# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ARG BUILD_MODE=full
ENV BUILD_MODE=${BUILD_MODE}

# System dependencies
RUN apt-get update && \
    if [ "$BUILD_MODE" = "cleaner" ]; then \
        apt-get install -y --no-install-recommends curl; \
    else \
        apt-get install -y --no-install-recommends ffmpeg curl; \
    fi && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt requirements-cleaner.txt ./
RUN if [ "$BUILD_MODE" = "cleaner" ]; then \
        pip install --no-cache-dir -r requirements-cleaner.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy source
COPY . .

# Create required directories
RUN mkdir -p tmp logs

# ── Runtime ───────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENABLE_TRANSCRIPTION=true \
    WHISPER_MODEL=large-v3 \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    UPLOAD_DIR=tmp \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=info

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ═══════════════════════════════════════════════════════════════════════════════
# GPU variant — uncomment the FROM line below and comment out the one above
# to build a CUDA-enabled image.
# ═══════════════════════════════════════════════════════════════════════════════
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base
# RUN apt-get update && apt-get install -y python3.11 python3-pip ffmpeg curl \
#     && rm -rf /var/lib/apt/lists/*
# ENV WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16
