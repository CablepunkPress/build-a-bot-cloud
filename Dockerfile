FROM python:3.14-slim

# Install system dependencies (in case needed)
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Upgrade pip & install dependencies
RUN pip install --upgrade pip wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application package
COPY basic_bot/ basic_bot/
COPY dashboard.json .

# Document the port
EXPOSE 8080

# Start server with exec for proper signal handling
CMD exec uvicorn basic_bot.app:app --host 0.0.0.0 --port ${PORT:-8080}