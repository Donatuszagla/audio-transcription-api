"""
services/cleaning_pipeline.py
──────────────────────────────
Two-pass LLM cleaning pipeline for sermon documents.

Pipeline stages:
  1. EXTRACTING  — parse .doc/.docx with mammoth
  2. Pre-clean   — regex filler/tongues removal (no LLM)
  3. Chunk        — split into 1000-word chunks
  4. CLEANING     — Pass 1: clean each chunk sequentially
  5. REWRITING    — Pass 2: book-rewrite each chunk sequentially
  6. MERGING      — final cohesion LLM call on full assembled text
  7. DONE

All progress updates are written to the CleanJob object in the doc_job_store
so the frontend polling endpoint always has fresh data.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────

_PASS1_SYSTEM = """\
You are an expert sermon editor. Your job is to clean a raw sermon transcript excerpt.

Rules:
- Remove ALL repetitions, verbal stutters, filler words, and redundant phrases
- Remove prayers, tongues, and worship interjections that interrupt the teaching
- Preserve EVERY teaching point, scripture reference, and theological content
- Do NOT summarize, shorten, or paraphrase the teaching content
- Do NOT add anything that was not in the original
- Fix obvious grammar errors while keeping the speaker's voice
- Return ONLY the cleaned text — no explanations, no headers"""

_PASS2_SYSTEM = """\
You are a professional book editor specialising in Christian teaching books.
You are rewriting a cleaned sermon excerpt into a polished book chapter section.

Rules:
- Organise the content into clear, well-structured paragraphs
- Ensure logical flow — ideas should connect smoothly
- Improve grammar, sentence structure, and clarity
- Where an explanation is brief or unclear, expand it slightly for a reader who is not present in the room
- Maintain the original meaning, theology, and teaching intent exactly
- Use a warm, authoritative, pastoral tone — not academic
- Do NOT add new theological content or scripture references not in the original
- Return ONLY the rewritten text — no explanations, no headers, no titles"""

_MERGE_SYSTEM = """\
You are a senior book editor. The following text is a sermon book chapter
assembled from multiple independently-rewritten sections.

Your task:
- Read the full text and make it feel like a single cohesive chapter
- Smooth transitions between sections so they flow naturally
- Remove any redundancy or repeated points that appear across section boundaries
- Ensure the opening paragraph introduces the chapter topic clearly
- Ensure the closing paragraph provides a satisfying conclusion or application
- Preserve ALL theological content and teaching points
- Return ONLY the final, cohesive chapter text — no explanations, no headers"""


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_cleaning_pipeline(job_id: str) -> None:
    """
    Full two-pass cleaning pipeline. Runs as a FastAPI BackgroundTask.

    Reads file path from the CleanJob; updates job status at every step.
    On failure, marks job as FAILED with a descriptive error message.
    Cleans up the uploaded temp file on completion (success or failure).
    """
    from services.doc_job_store import doc_job_store
    from utils.docx_parser import extract_text_from_docx
    from utils.doc_pre_cleaner import pre_clean_text
    from utils.chunker import chunk_text
    from services.llm_client import call_llm
    from config import get_settings

    cfg = get_settings()
    job = doc_job_store.get(job_id)
    if job is None:
        logger.error("CleanJob %s not found — cannot process", job_id)
        return

    file_path = Path(job.file_path)

    try:
        # ── Stage 1: Extract ──────────────────────────────────────────────────
        job.mark_extracting()
        doc_job_store.update(job)
        logger.info("[%s] Extracting text from %s", job_id, file_path.name)

        loop = asyncio.get_running_loop()
        raw_text = await loop.run_in_executor(
            None, extract_text_from_docx, file_path
        )

        # ── Pre-clean (no LLM, fast) ──────────────────────────────────────────
        logger.info("[%s] Pre-cleaning text (%d chars)", job_id, len(raw_text))
        pre_cleaned = await loop.run_in_executor(None, pre_clean_text, raw_text)

        # ── Chunk ─────────────────────────────────────────────────────────────
        chunks = await loop.run_in_executor(
            None, chunk_text, pre_cleaned, cfg.llm_chunk_max_words
        )
        if not chunks:
            raise RuntimeError("Document is empty after pre-cleaning.")

        total = len(chunks)
        logger.info("[%s] Split into %d chunks", job_id, total)

        # ── Stage 2: Pass 1 — Clean ───────────────────────────────────────────
        pass1_results: list[str] = []

        for i, chunk in enumerate(chunks, start=1):
            job.mark_cleaning(chunk_total=total, chunk_current=i)
            doc_job_store.update(job)
            logger.info("[%s] Pass 1 chunk %d/%d", job_id, i, total)

            prompt = (
                f"Clean the following sermon transcript excerpt:\n\n"
                f"---\n{chunk}\n---"
            )
            cleaned_chunk = await call_llm(prompt, system=_PASS1_SYSTEM)
            pass1_results.append(cleaned_chunk)

        # Cache Pass-1 results on the job (allows Pass-2 retry without re-running Pass-1)
        job.pass1_chunks = pass1_results
        doc_job_store.update(job)

        # Directly use Pass 1 results as the final text as per user request
        final_text = "\n\n".join(pass1_results)

        # ── Stage 3: Pass 2 — Rewrite ─────────────────────────────────────────
        # pass2_results: list[str] = []
        #
        # for i, p1_chunk in enumerate(pass1_results, start=1):
        #     job.mark_rewriting(chunk_current=i)
        #     doc_job_store.update(job)
        #     logger.info("[%s] Pass 2 chunk %d/%d", job_id, i, total)
        #
        #     prompt = (
        #         f"Rewrite the following cleaned sermon excerpt as a book chapter section:\n\n"
        #         f"---\n{p1_chunk}\n---"
        #     )
        #     rewritten_chunk = await call_llm(prompt, system=_PASS2_SYSTEM)
        #     pass2_results.append(rewritten_chunk)
        #
        # ── Stage 4: Merge ────────────────────────────────────────────────────
        # job.mark_merging()
        # doc_job_store.update(job)
        # logger.info("[%s] Final merge pass", job_id)
        #
        # assembled = "\n\n".join(pass2_results)
        #
        # if total > 1:
        #     # Only run merge LLM call if there are multiple chunks;
        #     # single-chunk docs are already cohesive.
        #     merge_prompt = (
        #         f"Make the following assembled chapter text cohesive:\n\n"
        #         f"---\n{assembled}\n---"
        #     )
        #     final_text = await call_llm(merge_prompt, system=_MERGE_SYSTEM)
        # else:
        #     final_text = assembled

        # ── Done ──────────────────────────────────────────────────────────────
        job.mark_done(final_text)
        doc_job_store.update(job)
        logger.info("[%s] Cleaning pipeline complete (%d chars output)", job_id, len(final_text))

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("[%s] Cleaning pipeline failed: %s", job_id, error_msg)
        job.mark_failed(error_msg)
        doc_job_store.update(job)

    finally:
        # ── Clean up uploaded temp file ───────────────────────────────────────
        if file_path.exists():
            try:
                file_path.unlink()
                logger.debug("[%s] Deleted temp file: %s", job_id, file_path)
            except OSError as e:
                logger.warning("[%s] Could not delete temp file %s: %s", job_id, file_path, e)
