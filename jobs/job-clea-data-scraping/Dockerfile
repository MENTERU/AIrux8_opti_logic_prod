# Use the official Python image
FROM python:3.13-slim

# Set non-interactive mode for APT
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    python3-dev \
    build-essential \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set the working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install Python dependencies using uv sync (creates .venv)
# Use --frozen if uv.lock exists, otherwise sync will generate it
RUN if [ -f uv.lock ]; then uv sync --frozen; else uv sync; fi

# Install Playwright browsers (required for the scraper)
RUN uv run playwright install chromium && \
    uv run playwright install-deps chromium

# Copy the application code (this layer will be rebuilt if code changes)
COPY . .

# Expose the port
EXPOSE 8080

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]






