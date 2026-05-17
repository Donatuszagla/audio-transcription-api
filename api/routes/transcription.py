"""
api/routes/transcription.py
────────────────────────────
Transcription endpoints:

  POST /transcribe       → submit job, returns job_id immediately
  GET  /status/{job_id} → poll for result
  GET  /jobs            → list all active jobs (admin/debug)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from config import get_settings
from services import job_store, transcribe_file, JobStatus
from utils import preprocess_audio, validate_extension, get_duration_seconds

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcription"])

CHUNK_SIZE = 1024 * 1024  # 1 MB streaming chunks


# ── Dependency: concurrency guard ─────────────────────────────────────────────

def check_capacity():
    cfg     = get_settings()
    active  = job_store.active_job_count()
    if active >= cfg.max_concurrent_jobs:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Server busy: {active} job(s) already running. "
                f"Max concurrent: {cfg.max_concurrent_jobs}. Try again later."
            ),
        )


# ── Background task ───────────────────────────────────────────────────────────

async def _run_transcription_job(
    job_id: str,
    raw_path: Path,
    language: str | None,
    clean_output: bool,
) -> None:
    """
    Full pipeline run in a BackgroundTask:
      1. Preprocess audio (ffmpeg)
      2. Transcribe (Whisper)
      3. Update job store
      4. Clean up temp files
    """
    job = job_store.get(job_id)
    if job is None:
        logger.error("Job %s not found — cannot process", job_id)
        return

    processed_path: Path | None = None
    cfg = get_settings()

    try:
        job.mark_processing()
        job_store.update(job)

        # ── Step 1: Preprocess ────────────────────────────────────────────────
        logger.info("[%s] Preprocessing audio…", job_id)
        loop = asyncio.get_running_loop()
        processed_path = await loop.run_in_executor(
            None, preprocess_audio, raw_path, cfg.upload_dir
        )

        audio_duration = get_duration_seconds(processed_path)
        job.duration_seconds = audio_duration
        job_store.update(job)

        # ── Step 2: Transcribe ────────────────────────────────────────────────
        logger.info("[%s] Starting transcription (%.1f s audio)…", job_id, audio_duration)
        result = await transcribe_file(
            processed_path,
            language=language,
            clean_output=clean_output,
        )

        job.mark_done(result)
        job_store.update(job)
        logger.info("[%s] Transcription complete.", job_id)

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("[%s] Transcription failed: %s", job_id, error_msg)
        job.mark_failed(error_msg)
        job_store.update(job)

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        for path in [raw_path, processed_path]:
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                    logger.debug("[%s] Deleted temp file: %s", job_id, path)
                except OSError as e:
                    logger.warning("[%s] Could not delete temp file %s: %s", job_id, path, e)


# ── POST /transcribe ──────────────────────────────────────────────────────────

@router.post(
    "/transcribe",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an audio file for transcription",
    response_description="Job accepted — poll /status/{job_id} for result",
)
async def submit_transcription(
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(check_capacity)],
    file: UploadFile = File(..., description="Audio file (mp3, wav, m4a, flac, aac, ogg)"),
    language: str | None = Form(
        default="en",
        description="ISO-639-1 language code (e.g. 'en'). Omit for auto-detection.",
    ),
    clean_output: bool = Form(
        default=True,
        description="Apply filler-word removal, deduplication, and paragraph merging.",
    ),
):
    cfg = get_settings()

    # ── Validate extension ────────────────────────────────────────────────────
    try:
        validate_extension(file.filename or "", cfg.allowed_extensions)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    # ── Stream upload to disk ─────────────────────────────────────────────────
    raw_filename = f"{uuid.uuid4().hex}_{Path(file.filename or 'audio').name}"
    raw_path     = Path(cfg.upload_dir) / raw_filename
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    max_bytes   = cfg.max_upload_mb * 1024 * 1024

    try:
        with raw_path.open("wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    f.close()
                    raw_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {cfg.max_upload_mb} MB.",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    logger.info(
        "Received upload: %s (%.1f MB)",
        file.filename, total_bytes / 1_048_576,
    )

    # ── Create job ────────────────────────────────────────────────────────────
    job = job_store.create(file.filename or "unknown")

    # ── Enqueue background task ───────────────────────────────────────────────
    background_tasks.add_task(
        _run_transcription_job,
        job.job_id,
        raw_path,
        language,
        clean_output,
    )

    return {
        "job_id":   job.job_id,
        "status":   job.status.value,
        "filename": file.filename,
        "message":  "Job accepted. Poll GET /status/{job_id} for results.",
    }


# ── GET /status/{job_id} ──────────────────────────────────────────────────────

@router.get(
    "/status/{job_id}",
    summary="Poll transcription job status",
)
async def get_status(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have expired.",
        )
    return job.to_dict()


# ── GET /jobs (debug/admin) ───────────────────────────────────────────────────

@router.get("/jobs", summary="List all tracked jobs (debug)")
async def list_jobs():
    """Returns all jobs in the store. For admin/debug use only."""
    with job_store._lock:
        jobs = [j.to_dict() for j in job_store._store.values()]
    return {"count": len(jobs), "jobs": jobs}
