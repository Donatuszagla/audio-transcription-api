"""
services/job_store.py
─────────────────────
Thread-safe in-memory job store with automatic TTL expiry.

For production at scale, swap the _store dict for a Redis backend
(use redis-py / aioredis) — the interface stays identical.
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


class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


class Job:
    __slots__ = (
        "job_id", "status", "created_at", "updated_at",
        "result", "error", "filename", "duration_seconds",
    )

    def __init__(self, filename: str) -> None:
        self.job_id: str = str(uuid.uuid4())
        self.status: JobStatus = JobStatus.QUEUED
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.filename: str = filename
        self.duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        elapsed = round(self.updated_at - self.created_at, 1)
        return {
            "job_id":          self.job_id,
            "status":          self.status.value,
            "filename":        self.filename,
            "created_at":      self.created_at,
            "elapsed_seconds": elapsed,
            "result":          self.result,
            "error":           self.error,
        }

    def _touch(self) -> None:
        self.updated_at = time.time()

    def mark_processing(self) -> None:
        self.status = JobStatus.PROCESSING
        self._touch()

    def mark_done(self, result: dict[str, Any]) -> None:
        self.status = JobStatus.DONE
        self.result = result
        self._touch()

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.error  = error
        self._touch()


class JobStore:
    """
    Simple thread-safe dict-backed job store.

    Replace _store with a Redis client for multi-process / multi-host setups:

        import redis.asyncio as redis
        r = redis.from_url("redis://localhost")
        await r.set(job_id, json.dumps(job_dict), ex=ttl_seconds)
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: dict[str, Job] = {}
        self._lock  = threading.Lock()
        self._ttl   = ttl_seconds

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(self, filename: str) -> Job:
        job = Job(filename)
        with self._lock:
            self._store[job.job_id] = job
        logger.info("Job created: %s (%s)", job.job_id, filename)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._store.get(job_id)

    def update(self, job: Job) -> None:
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
            logger.info("Evicted %d expired job(s)", evicted)
        return evicted

    async def start_eviction_loop(self, interval_seconds: int = 300) -> None:
        """Background coroutine: purge old jobs every `interval_seconds`."""
        while True:
            await asyncio.sleep(interval_seconds)
            self.evict_expired()

    # ── Concurrency guard ─────────────────────────────────────────────────────

    def active_job_count(self) -> int:
        with self._lock:
            return sum(
                1 for j in self._store.values()
                if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
            )


# Module-level singleton — imported by other modules
job_store = JobStore()
