"""
services/doc_job_store.py
──────────────────────────
Thread-safe in-memory store for document cleaning jobs.

Mirrors the design of job_store.py (audio transcription) but with richer
status fields to support chunk-level progress reporting.

For production at scale: replace _store dict with a Redis backend —
the public interface stays identical.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CleanJobStatus(str, Enum):
    QUEUED     = "queued"
    EXTRACTING = "extracting"   # parsing DOCX
    CLEANING   = "cleaning"     # Pass 1 — LLM cleanup
    REWRITING  = "rewriting"    # Pass 2 — LLM book-rewrite
    MERGING    = "merging"      # final cohesion LLM call
    DONE       = "done"
    FAILED     = "failed"


class CleanJob:
    """
    Represents a single document cleaning job.

    Progress tracking fields
    ------------------------
    chunk_total   : total number of chunks in this document
    chunk_current : index of the chunk currently being processed (1-based)
    phase         : human-readable phase label shown in the UI
    pass1_chunks  : cached Pass-1 results (enables Pass-2 retry without re-running Pass-1)
    """

    __slots__ = (
        "job_id", "status", "filename", "file_path",
        "created_at", "updated_at",
        "chunk_total", "chunk_current", "phase",
        "pass1_chunks",
        "result_text", "error",
    )

    def __init__(self, filename: str, file_path: str) -> None:
        self.job_id: str        = str(uuid.uuid4())
        self.status: CleanJobStatus = CleanJobStatus.QUEUED
        self.filename: str      = filename
        self.file_path: str     = file_path
        self.created_at: float  = time.time()
        self.updated_at: float  = time.time()

        # Progress
        self.chunk_total: int   = 0
        self.chunk_current: int = 0
        self.phase: str         = "Queued"

        # Intermediate / final results
        self.pass1_chunks: list[str] = []
        self.result_text: str | None = None
        self.error: str | None       = None

    # ── State transitions ──────────────────────────────────────────────────────

    def mark_extracting(self) -> None:
        self.status = CleanJobStatus.EXTRACTING
        self.phase  = "Extracting document text…"
        self._touch()

    def mark_cleaning(self, chunk_total: int, chunk_current: int = 1) -> None:
        self.status        = CleanJobStatus.CLEANING
        self.chunk_total   = chunk_total
        self.chunk_current = chunk_current
        self.phase         = f"Pass 1 — Cleaning (chunk {chunk_current} of {chunk_total})"
        self._touch()

    def mark_rewriting(self, chunk_current: int) -> None:
        self.status        = CleanJobStatus.REWRITING
        self.chunk_current = chunk_current
        self.phase         = f"Pass 2 — Rewriting (chunk {chunk_current} of {self.chunk_total})"
        self._touch()

    def mark_merging(self) -> None:
        self.status = CleanJobStatus.MERGING
        self.phase  = "Final merge — making text cohesive…"
        self._touch()

    def mark_done(self, result_text: str) -> None:
        self.status      = CleanJobStatus.DONE
        self.result_text = result_text
        self.phase       = "Done"
        self._touch()

    def mark_failed(self, error: str) -> None:
        self.status = CleanJobStatus.FAILED
        self.error  = error
        self.phase  = "Failed"
        self._touch()

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        elapsed = round(self.updated_at - self.created_at, 1)
        # Overall progress percentage (0–100)
        pct = _compute_pct(self.status, self.chunk_current, self.chunk_total)
        return {
            "job_id":          self.job_id,
            "status":          self.status.value,
            "filename":        self.filename,
            "phase":           self.phase,
            "chunk_current":   self.chunk_current,
            "chunk_total":     self.chunk_total,
            "progress_pct":    pct,
            "created_at":      self.created_at,
            "elapsed_seconds": elapsed,
            "result_text":     self.result_text,
            "error":           self.error,
        }

    def _touch(self) -> None:
        self.updated_at = time.time()


# ── Progress percentage helper ────────────────────────────────────────────────

def _compute_pct(
    status: CleanJobStatus,
    chunk_current: int,
    chunk_total: int,
) -> int:
    """
    Map pipeline state to an overall 0-100 progress percentage.

    Phases:
      QUEUED      ->  0
      EXTRACTING  ->  2
      CLEANING    ->  5  + 40 * (chunk_current / chunk_total)   [5–45]
      REWRITING   ->  45 + 45 * (chunk_current / chunk_total)   [45–90]
      MERGING     ->  90
      DONE        ->  100
      FAILED      ->  (current value, doesn't matter)
    """
    if status == CleanJobStatus.QUEUED:
        return 0
    if status == CleanJobStatus.EXTRACTING:
        return 2
    if status == CleanJobStatus.CLEANING:
        ratio = chunk_current / chunk_total if chunk_total else 0
        return int(5 + 40 * ratio)
    if status == CleanJobStatus.REWRITING:
        ratio = chunk_current / chunk_total if chunk_total else 0
        return int(45 + 45 * ratio)
    if status == CleanJobStatus.MERGING:
        return 90
    if status == CleanJobStatus.DONE:
        return 100
    return 0


# ── Job store ─────────────────────────────────────────────────────────────────

class DocJobStore:
    """
    Thread-safe in-memory store for CleanJob instances.

    Replace _store with a Redis client for multi-process / multi-host setups.
    """

    def __init__(self, ttl_seconds: int = 7200) -> None:  # 2-hour TTL for long docs
        self._store: dict[str, CleanJob] = {}
        self._lock  = threading.Lock()
        self._ttl   = ttl_seconds

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(self, filename: str, file_path: str) -> CleanJob:
        job = CleanJob(filename=filename, file_path=file_path)
        with self._lock:
            self._store[job.job_id] = job
        logger.info("CleanJob created: %s (%s)", job.job_id, filename)
        return job

    def get(self, job_id: str) -> CleanJob | None:
        with self._lock:
            return self._store.get(job_id)

    def update(self, job: CleanJob) -> None:
        with self._lock:
            self._store[job.job_id] = job

    # ── TTL eviction ──────────────────────────────────────────────────────────

    def evict_expired(self) -> int:
        now = time.time()
        evicted = 0
        with self._lock:
            expired = [
                jid for jid, j in self._store.items()
                if now - j.created_at > self._ttl
            ]
            for jid in expired:
                del self._store[jid]
                evicted += 1
        if evicted:
            logger.info("Evicted %d expired clean job(s)", evicted)
        return evicted

    async def start_eviction_loop(self, interval_seconds: int = 300) -> None:
        """Background coroutine: purge old jobs every `interval_seconds`."""
        while True:
            await asyncio.sleep(interval_seconds)
            self.evict_expired()

    def active_job_count(self) -> int:
        with self._lock:
            return sum(
                1 for j in self._store.values()
                if j.status in (CleanJobStatus.QUEUED, CleanJobStatus.EXTRACTING,
                                CleanJobStatus.CLEANING, CleanJobStatus.REWRITING,
                                CleanJobStatus.MERGING)
            )


# Module-level singleton
doc_job_store = DocJobStore()
