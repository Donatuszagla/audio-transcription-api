"""
main.py
───────
FastAPI application entry point.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import logging
import logging.config
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health_router, transcription_router
from config import get_settings
from services.job_store import job_store

# ── Logging ───────────────────────────────────────────────────────────────────

def _configure_logging(level: str) -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "logs/app.log",
                    "maxBytes": 10 * 1024 * 1024,   # 10 MB
                    "backupCount": 5,
                    "formatter": "default",
                },
            },
            "root": {
                "level": level.upper(),
                "handlers": ["console", "file"],
            },
        }
    )


# ── Rate limiting (simple sliding-window in memory) ───────────────────────────
# For production: replace with slowapi + Redis.

_rate_store: dict[str, list[float]] = {}
_rate_lock  = asyncio.Lock()


async def _check_rate_limit(client_ip: str, limit: int, window: int = 60) -> bool:
    now = time.time()
    async with _rate_lock:
        hits = _rate_store.get(client_ip, [])
        hits = [t for t in hits if now - t < window]
        if len(hits) >= limit:
            return False
        hits.append(now)
        _rate_store[client_ip] = hits
    return True


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    _configure_logging(cfg.log_level)
    logger = logging.getLogger(__name__)

    # Ensure temp directory exists
    Path(cfg.upload_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Sermon Transcription API starting up")
    logger.info("Whisper model : %s", cfg.whisper_model)
    logger.info("Device        : %s", cfg.whisper_device)
    logger.info("Upload dir    : %s", Path(cfg.upload_dir).resolve())
    logger.info("Max upload    : %d MB", cfg.max_upload_mb)
    logger.info("=" * 60)

    # Start job TTL eviction loop
    eviction_task = asyncio.create_task(
        job_store.start_eviction_loop(interval_seconds=300)
    )

    # Optional: pre-warm model at startup (removes cold-start on first request)
    # Uncomment for production — adds ~30-60 s to startup time:
    # logger.info("Pre-warming Whisper model…")
    # from services.transcription_service import get_model
    # await get_model()
    # logger.info("Model ready.")

    yield

    eviction_task.cancel()
    logger.info("API shut down.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title        = "Sermon Transcription API",
        description  = (
            "Self-hosted audio transcription powered by OpenAI Whisper large-v3. "
            "Optimised for long-form speech with automatic text cleaning."
        ),
        version      = "1.0.0",
        lifespan     = lifespan,
        docs_url     = "/docs",
        redoc_url    = "/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # Tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting middleware ───────────────────────────────────────────────
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path.startswith("/transcribe"):
            client_ip = request.client.host if request.client else "unknown"
            allowed   = await _check_rate_limit(
                client_ip, cfg.rate_limit_per_minute
            )
            if not allowed:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": (
                            f"Rate limit exceeded: "
                            f"{cfg.rate_limit_per_minute} requests/minute allowed."
                        )
                    },
                )
        return await call_next(request)

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        t0       = time.perf_counter()
        response = await call_next(request)
        elapsed  = time.perf_counter() - t0
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logging.getLogger(__name__).exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check server logs."},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(transcription_router)

    return app


app = create_app()


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run(
        "main:app",
        host    = cfg.host,
        port    = cfg.port,
        workers = cfg.workers,
        reload  = False,
        log_level = cfg.log_level,
    )
