"""
api/routes/document.py
───────────────────────
Document cleaning endpoints:

  POST /document/upload           → upload .docx, returns job_id
  POST /document/clean/{job_id}   → start background cleaning pipeline
  GET  /document/status/{job_id}  → poll for progress and result
  GET  /document/jobs             → list all document jobs (admin/debug)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from config import get_settings
from services.doc_job_store import doc_job_store, CleanJobStatus
from services.cleaning_pipeline import run_cleaning_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/document", tags=["document"])

CHUNK_SIZE = 1024 * 1024  # 1 MB streaming chunks


# ── POST /document/upload ──────────────────────────────────────────────────────

@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload a Word document (.doc / .docx) for cleaning",
    response_description="Returns job_id — use it to start cleaning and poll status",
)
async def upload_document(
    file: UploadFile = File(..., description="Word document (.doc or .docx)"),
):
    cfg = get_settings()

    # ── Validate extension ────────────────────────────────────────────────────
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()

    if ext not in cfg.allowed_doc_extensions:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(cfg.allowed_doc_extensions))}"
            ),
        )

    # ── Stream file to disk ───────────────────────────────────────────────────
    upload_dir = Path(cfg.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name   = f"{uuid.uuid4().hex}_{Path(filename).name}"
    file_path   = upload_dir / safe_name
    total_bytes = 0
    max_bytes   = cfg.max_doc_upload_mb * 1024 * 1024

    try:
        with file_path.open("wb") as fh:
            while chunk := await file.read(CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    fh.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {cfg.max_doc_upload_mb} MB.",
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    logger.info(
        "Document uploaded: %s (%.2f MB)", filename, total_bytes / 1_048_576
    )

    # ── Create job ────────────────────────────────────────────────────────────
    job = doc_job_store.create(filename=filename, file_path=str(file_path))

    return {
        "job_id":   job.job_id,
        "filename": filename,
        "status":   job.status.value,
        "message":  "Upload successful. POST /document/clean/{job_id} to start cleaning.",
    }


# ── POST /document/clean/{job_id} ─────────────────────────────────────────────

@router.post(
    "/clean/{job_id}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start the LLM cleaning pipeline for an uploaded document",
    response_description="Job accepted — poll GET /document/status/{job_id} for progress",
)
async def start_cleaning(job_id: str, background_tasks: BackgroundTasks):
    job = doc_job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have expired or never existed.",
        )

    if job.status not in (CleanJobStatus.QUEUED, CleanJobStatus.FAILED):
        # Already running or done — return current state
        return job.to_dict()

    # Reset pass1 cache for retries
    job.pass1_chunks = []

    # Check that the file still exists (it gets deleted on completion)
    if not Path(job.file_path).exists():
        raise HTTPException(
            status_code=410,
            detail=(
                "The uploaded file no longer exists (it may have been processed and cleaned up). "
                "Please re-upload the document."
            ),
        )

    background_tasks.add_task(run_cleaning_pipeline, job_id)

    logger.info("Cleaning job %s started", job_id)
    return {
        "job_id":  job_id,
        "status":  "queued",
        "message": f"Cleaning started. Poll GET /document/status/{job_id} for progress.",
    }


# ── GET /document/status/{job_id} ─────────────────────────────────────────────

@router.get(
    "/status/{job_id}",
    summary="Poll document cleaning job status and progress",
)
async def get_document_status(job_id: str):
    job = doc_job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have expired.",
        )
    return job.to_dict()


# ── GET /document/jobs (admin/debug) ──────────────────────────────────────────

@router.get("/jobs", summary="List all document cleaning jobs (debug)")
async def list_document_jobs():
    """Returns all document jobs in the store. For admin/debug use only."""
    with doc_job_store._lock:
        jobs = [j.to_dict() for j in doc_job_store._store.values()]
    return {"count": len(jobs), "jobs": jobs}
