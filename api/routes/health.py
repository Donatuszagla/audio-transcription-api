"""
api/routes/health.py
────────────────────
Health and readiness endpoints.
"""

from __future__ import annotations

import platform
import time

from fastapi import APIRouter

router = APIRouter(tags=["health"])

_START_TIME = time.time()


@router.get("/health", summary="Liveness check")
async def health():
    """Returns 200 if the server is running."""
    return {
        "status":   "ok",
        "uptime_s": round(time.time() - _START_TIME, 1),
        "python":   platform.python_version(),
    }


@router.get("/ready", summary="Readiness check — verifies model is loaded")
async def ready():
    """Returns 200 when the Whisper model is in memory."""
    from services.transcription_service import _model
    model_loaded = _model is not None
    return {
        "ready":        model_loaded,
        "model_loaded": model_loaded,
    }
