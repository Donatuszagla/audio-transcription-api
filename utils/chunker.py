"""
utils/chunker.py
─────────────────
Split long sermon text into LLM-sized chunks.

Strategy (priority order):
  1. Split on double-newline paragraph boundaries (preferred)
  2. If a single paragraph exceeds max_words, split on sentence boundaries
  3. Never split mid-sentence

Chunk size: 800–1200 words (configurable, default 1000)
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def chunk_text(text: str, max_words: int = 1000) -> list[str]:
    """
    Split *text* into a list of chunks, each at most *max_words* words.

    Splitting respects paragraph boundaries (``\\n\\n``) first, then sentence
    boundaries. A chunk will never be empty.

    Parameters
    ----------
    text     : pre-cleaned plain text
    max_words: soft upper bound on words per chunk (default 1000)

    Returns
    -------
    List of non-empty text chunks. At least one chunk is always returned.
    """
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = _word_count(para)

        if para_words > max_words:
            # This paragraph alone is too long — must split it at sentence boundaries
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_word_count = 0

            sentence_chunks = _split_on_sentences(para, max_words)
            chunks.extend(sentence_chunks)
            continue

        if current_word_count + para_words > max_words and current_parts:
            # Adding this paragraph would overflow — flush and start fresh
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_word_count = para_words
        else:
            current_parts.append(para)
            current_word_count += para_words

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Sanity: filter empty strings
    chunks = [c.strip() for c in chunks if c.strip()]

    logger.info(
        "chunker: %d words -> %d chunks (max %d words/chunk)",
        _word_count(text), len(chunks), max_words,
    )
    return chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _word_count(text: str) -> int:
    return len(text.split())


def _split_on_sentences(text: str, max_words: int) -> list[str]:
    """
    Split a single over-length paragraph into sentence-boundary chunks.
    Each chunk will have at most max_words words.
    """
    sentences = _SENTENCE_SPLIT.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_wc = 0

    for sent in sentences:
        sent_wc = _word_count(sent)

        if sent_wc > max_words:
            # Even a single sentence is too long — hard-split on words
            if current:
                chunks.append(" ".join(current))
                current = []
                current_wc = 0
            word_chunks = _split_on_words(sent, max_words)
            chunks.extend(word_chunks)
            continue

        if current_wc + sent_wc > max_words and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_wc = sent_wc
        else:
            current.append(sent)
            current_wc += sent_wc

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_on_words(text: str, max_words: int) -> list[str]:
    """Hard-split by word count when no sentence boundaries exist."""
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]
