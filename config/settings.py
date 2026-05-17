"""
config/settings.py
──────────────────
All configuration is driven by environment variables.
Copy .env.example → .env and adjust before running.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Whisper ──────────────────────────────────────────────────────────────
    whisper_model: Literal[
        "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
    ] = "large-v3"

    # Device: "cuda" | "cpu" | "auto"  (auto = cuda if available, else cpu)
    whisper_device: str = "auto"

    # Compute type for faster-whisper (float16 on GPU, int8 on CPU recommended)
    whisper_compute_type: str = "auto"  # auto resolves at startup

    # Beam size (higher = slightly better accuracy, slower)
    whisper_beam_size: int = 5

    # VAD (Voice Activity Detection) filter — removes silence automatically
    whisper_vad_filter: bool = True

    # ── File handling ────────────────────────────────────────────────────────
    upload_dir: str = "tmp"
    max_upload_mb: int = 2048          # 2 GB ceiling — enough for 4+ hr audio
    allowed_extensions: set[str] = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}

    # ── Text cleaning ────────────────────────────────────────────────────────
    clean_output_default: bool = True
    min_segment_confidence: float = 0.0   # drop segments below this avg log-prob

    # ── Job / queue ──────────────────────────────────────────────────────────
    job_ttl_seconds: int = 3600           # purge job results after 1 h
    max_concurrent_jobs: int = 2

    # ── Rate limiting ────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 10

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    workers: int = 1                      # keep 1 — Whisper model is shared


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
