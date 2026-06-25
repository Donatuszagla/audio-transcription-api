"""
utils/doc_pre_cleaner.py
─────────────────────────
Sermon-specific pre-cleaning applied BEFORE LLM processing.

Goals:
  - Cut token count (saves cost and speeds up LLM calls)
  - Remove noise that would confuse the LLM rewrite
  - Preserve ALL teaching content

This is NOT the LLM cleaning step — it is a cheap regex/rule layer that runs
in milliseconds and strips obvious sermon noise.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

logger = logging.getLogger(__name__)

# ── Sermon filler phrases ─────────────────────────────────────────────────────
# These are whole-phrase matches (case-insensitive, word-boundary aware).
# Order matters: longer phrases first to avoid partial matches.

_SERMON_FILLERS: list[str] = [
    # Worship interjections
    "praise the lord",
    "thank you jesus",
    "thank you lord",
    "glory to god",
    "hallelujah",
    "glory hallelujah",
    "to god be the glory",
    "blessed be the lord",
    "bless the lord",
    # Audience acknowledgements
    "you understand me",
    "you understand",
    "do you understand",
    "are you with me",
    "are you following",
    "can you hear me",
    "somebody say amen",
    "say amen",
    "can i get an amen",
    "somebody shout",
    # Verbal tics
    "in other words",
    "i mean i mean",
    "i mean",
    "you know what i mean",
    "you know",
    "like i said",
    "as i said",
    "as i was saying",
    "okay okay",
    "right right",
    "so so",
    "basically basically",
    "basically",
    "you see",
    "listen listen",
    "look look",
    # Prayer transitions
    "let us pray",
    "let's pray",
    "father we thank you",
    "lord we thank you",
    "in the name of jesus",
    "in jesus name",
    "in jesus' name",
    # Tongues / glossolalia (phonetic patterns) — handled separately via regex
]

# Build compiled patterns — longest first to avoid partial-match issues
_FILLER_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(phrase) + r"\b[,.]?", re.IGNORECASE)
    for phrase in sorted(_SERMON_FILLERS, key=len, reverse=True)
]

# Simple filler words (single word)
_SINGLE_FILLERS = {
    "amen", "hallelujah", "uhh", "uhm", "umm", "um", "uh", "hmm", "hm",
    "er", "err", "ah", "ahh",
}
_SINGLE_FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _SINGLE_FILLERS) + r")\b[,.]?\s*",
    re.IGNORECASE,
)

# Tongues / glossolalia pattern — sequences of repeated consonant-heavy syllables
# e.g. "shundai shundai", "randara", "kum ba ya" in non-standard contexts
_TONGUES_PATTERN = re.compile(
    r"\b(?:[a-z]{2,4}(?:da|ra|sha|la|na|ka|ta|ma){1,3}\s*){2,}\b",
    re.IGNORECASE,
)

# Repeated "amen amen amen…"
_REPEATED_AMEN = re.compile(r"\b(amen\s*){2,}", re.IGNORECASE)

# Excessive ellipsis / dashes used as pause markers
_PAUSE_MARKERS = re.compile(r"[.…]{3,}|-{2,}")

# Multiple blank lines
_MULTI_BLANK = re.compile(r"\n{3,}")

# Trailing whitespace on each line
_LINE_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)


def pre_clean_text(text: str) -> str:
    """
    Apply sermon-specific pre-cleaning to raw extracted text.

    Steps (in order):
    1. Remove repeated "amen" clusters
    2. Remove glossolalia / tongues patterns
    3. Remove multi-word filler phrases
    4. Remove single filler words
    5. Normalise pause markers
    6. Collapse repeated consecutive sentences
    7. Normalise whitespace

    Parameters
    ----------
    text : raw text extracted from the document

    Returns
    -------
    Pre-cleaned text ready for LLM processing.
    """
    original_len = len(text)

    text = _repeated_amen(text)
    text = _remove_tongues(text)
    text = _remove_phrase_fillers(text)
    text = _remove_single_fillers(text)
    text = _normalise_pauses(text)
    text = _collapse_repeated_sentences(text)
    text = _normalise_whitespace(text)

    removed_pct = max(0.0, (1 - len(text) / original_len) * 100) if original_len else 0
    logger.info(
        "Pre-clean: %d -> %d chars (%.1f%% removed)",
        original_len, len(text), removed_pct,
    )
    return text


# ── Internal helpers ──────────────────────────────────────────────────────────

def _repeated_amen(text: str) -> str:
    return _REPEATED_AMEN.sub("", text)


def _remove_tongues(text: str) -> str:
    return _TONGUES_PATTERN.sub("", text)


def _remove_phrase_fillers(text: str) -> str:
    for pattern in _FILLER_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def _remove_single_fillers(text: str) -> str:
    return _SINGLE_FILLER_PATTERN.sub(" ", text)


def _normalise_pauses(text: str) -> str:
    # Long ellipsis / dashes become a comma (preserves grammatical rhythm)
    return _PAUSE_MARKERS.sub(",", text)


def _collapse_repeated_sentences(text: str) -> str:
    """
    Remove consecutive duplicate sentences within the same paragraph.
    A sentence is defined as text ending with . ! ?
    """
    paragraphs = text.split("\n\n")
    result: list[str] = []
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        seen: list[str] = []
        for sent in sentences:
            normalised = re.sub(r"\s+", " ", sent.strip().lower())
            if not seen or normalised != re.sub(r"\s+", " ", seen[-1].strip().lower()):
                seen.append(sent)
        result.append(" ".join(seen))
    return "\n\n".join(result)


def _normalise_whitespace(text: str) -> str:
    text = _LINE_TRAILING.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()
