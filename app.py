"""FastAPI application for cloud deployment.

Imports create_runtime from the factory, wraps it in FastAPI routes.
This file will move to its own cloud deployment repo when the split
is complete.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request

from basic_bot.chat import chat_with_claude
from basic_bot.config import (
    DEFAULT_MODEL,
    HISTORY_LIMIT,
    STORAGE_BACKEND,
    WINDOW_CEILING,
    WINDOW_FLOOR,
)
from basic_bot.factory import create_runtime
from basic_bot.fold import build_metadata, fold_rag, fold_summary, should_fold
from basic_bot.memory import get_messages
from basic_bot.models import MODELS

logger = logging.getLogger(__name__)

# Resolve agent path — the repo root (parent of this package)
AGENT_PATH = Path(__file__).parent.parent
runtime = create_runtime(AGENT_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("%s starting (collection: %s)", runtime.agent_id, runtime.agent_id)
    logger.info("Storage backend: %s", STORAGE_BACKEND)
    logger.info("Fold window: %d–%d messages", WINDOW_FLOOR, WINDOW_CEILING)
    logger.info("Default model: %s", DEFAULT_MODEL)
    logger.info(
        "Available models: %s",
        ", ".join(cfg["display_name"] for cfg in MODELS.values()),
    )
    logger.info("SDK: anthropic %s", anthropic.__version__)
    logger.info(
        "Tools: %d [%s]",
        len(runtime.tool_registry),
        ", ".join(runtime.tool_registry.keys()) or "none",
    )
    logger.info("History limit: %d messages", HISTORY_LIMIT)
    logger.info("=" * 50)

    if STORAGE_BACKEND == "firestore":
        try:
            from google.cloud import firestore
            db = firestore.Client()
            reg = dict(runtime.dashboard)
            reg["backendUrl"] = os.environ.get("BACKEND_URL", "")
            db.collection("agents").document(runtime.agent_id).set(reg)
            logger.info("Registered agent '%s' in Firestore", runtime.agent_id)
        except Exception:
            logger.exception("Failed to register agent — dashboard may be stale")

    yield
    logger.info("%s shutting down", runtime.agent_id)


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_id = data.get("session_id")
        message = data.get("message", "").strip()
        model_id = data.get("model", DEFAULT_MODEL)
        effort = data.get("effort")
        thinking = data.get("thinking", False)

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        logger.info(
            "Chat — user: %s, model: %s (%d chars)",
            user_id, model_id, len(message),
        )

        result = await chat_with_claude(
            runtime, user_id, message, model_id, effort, thinking,
        )

        metadata = build_metadata(result)
        seq = runtime.store.save_turn(
            user_id, message, result["reply"], metadata=metadata,
        )
        logger.info("Saved turn (seq %d–%d)", seq, seq + 1)

        fold_state = should_fold(runtime.store, user_id)
        if fold_state:
            chunk = fold_rag(runtime.store, user_id, fold_state)
            if chunk:
                asyncio.create_task(
                    asyncio.to_thread(
                        fold_summary,
                        runtime.store,
                        user_id,
                        fold_state["summary"],
                        chunk,
                    )
                )
                logger.info(
                    "Fold triggered for user %s — RAG done, summary in background",
                    user_id,
                )

        return {
            "response": result["reply"],
            "seq": seq,
            "model_used": result["model_used"],
            "display_name": result["display_name"],
            "effort": result["effort"],
            "thinking": result["thinking"],
            "fallback": result["fallback"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/history")
async def history(request: Request):
    user_id = request.query_params.get("session_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    limit_param = request.query_params.get("limit")
    if limit_param:
        try:
            limit = min(int(limit_param), WINDOW_CEILING)
        except ValueError:
            limit = HISTORY_LIMIT
    else:
        limit = HISTORY_LIMIT

    messages = get_messages(runtime.store, user_id, limit)

    return {
        "messages": [
            {
                "role": msg["role"],
                "content": msg["content"],
                "seq": msg["seq"],
                "metadata": msg.get("metadata"),
            }
            for msg in messages
        ],
        "count": len(messages),
    }


@app.get("/models")
async def models_endpoint():
    return {
        "models": {
            mid: {
                "display_name": cfg["display_name"],
                "effort_levels": cfg["effort_levels"],
                "thinking_type": cfg["thinking_type"],
            }
            for mid, cfg in MODELS.items()
        },
        "default": DEFAULT_MODEL,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": runtime.agent_id}


@app.get("/")
async def root():
    return {"agent": runtime.agent_id, "status": "running"}
