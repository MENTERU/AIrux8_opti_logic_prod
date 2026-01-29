# syntax=docker/dockerfile:1.4

# ===== STAGE 1: Builder =====
FROM python:3.13-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install build tools, git and ssh client
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      openssh-client \
      build-essential \
      python3-dev \
      libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure pip up-to-date
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Ensure GitHub host key is trusted (prevents Host key verification failed)
# and create .ssh dir for root (ssh client inside builder uses /root/.ssh)
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    chmod 644 /root/.ssh/known_hosts

# Install package from private repo + requirements using BuildKit SSH mount
# NOTE: this requires you to build with --ssh (local) or --ssh forwarded (CI)
RUN --mount=type=ssh pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# ===== STAGE 2: Runtime =====
FROM python:3.13-slim
WORKDIR /app

# Copy only installed packages and app code
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

EXPOSE 8080

ENTRYPOINT ["python", "main.py"]
