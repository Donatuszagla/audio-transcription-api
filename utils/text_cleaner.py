"""
utils/text_cleaner.py
─────────────────────
Post-processing pipeline applied to raw Whisper output.

Steps (in order):
  1.  Strip segment-level artifacts (timestamps, bracket noise, music notes)
  2.  Remove filler words / hesitation tokens
  3.  Collapse repeated words or short repeated phrases
  4.  Fix punctuation spacing and strip doubled punctuation
  5.  Normalise casing (sentence-level capitalisation)
  6.  Merge short segments into readable paragraphs
  7.  Final whitespace normalisation
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)

# ── Filler words ──────────────────────────────────────────────────────────────
# Matched as whole words, case-insensitive, surrounded by word boundaries.
_FILLERS = {
    "um", "uh", "uhh", "hmm", "hm", "mhm", "uh-huh",
    "er", "err", "ah", "ahh", "oh-uh", "umm",
    "like",   # only stripped when it's a verbal tic (handled contextually below)
}

_FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _FILLERS) + r")\b[,.]?\s*",
    re.IGNORECASE,
)

# ── Whisper hallucination / artifact patterns ─────────────────────────────────
# These appear when Whisper processes silence or noise.
_ARTIFACT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\[.*?\]"),                          # [Music], [Applause], etc.
    re.compile(r"\(.*?\)"),                          # (inaudible), (crosstalk)
    re.compile(r"♪[^♪]*♪?"),                        # ♪ music markers ♪
    re.compile(r"♫[^♫]*♫?"),
    re.compile(r"<[^>]+>"),                          # <laugh>, <noise>
    re.compile(r"\btranscribed by\b.*", re.IGNORECASE),
    re.compile(r"\bsubtitles by\b.*", re.IGNORECASE),
    re.compile(r"\bwww\.\S+"),                       # stray URLs
]

# ── Punctuation normalisation ─────────────────────────────────────────────────
_MULTI_SPACE     = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,;:.!?])")
_DOUBLED_PUNCT   = re.compile(r"([.!?,;])\1+")
_TRAILING_COMMA  = re.compile(r",\s*$", re.MULTILINE)
_ELLIPSIS_CLEAN  = re.compile(r"\.{2,}")            # normalize ... → …
_DASH_SPACE      = re.compile(r"\s*-{2,}\s*")       # em-dash normalise


# ── Repeated word / phrase collapser ─────────────────────────────────────────
# Matches 1–6 word phrases repeated 2+ times consecutively.
_REPEAT_PATTERN  = re.compile(
    r"\b((?:\w+\s+){0,5}\w+)\s+(?:\1\s+)+",
    re.IGNORECASE,
)


@dataclass
class Segment:
    text: str
    start: float = 0.0
    end: float = 0.0
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


# ── Public entry point ────────────────────────────────────────────────────────

def clean_segments(
    segments: Sequence[Segment],
    min_confidence: float = 0.0,
    merge_gap_seconds: float = 2.5,
    max_paragraph_words: int = 120,
) -> tuple[str, list[dict]]:
    """
    Clean a list of Whisper segments and return:
      - full cleaned transcript string
      - list of cleaned segment dicts (for the API response)
    """
    cleaned_segments: list[dict] = []

    for seg in segments:
        # ── Drop low-confidence / hallucinated segments ───────────────────────
        if seg.no_speech_prob > 0.85:
            logger.debug("Dropped silent segment (%.2f ns_prob): %r", seg.no_speech_prob, seg.text[:60])
            continue

        if seg.avg_logprob < -1.5:
            logger.debug("Dropped low-confidence segment (%.2f logprob): %r", seg.avg_logprob, seg.text[:60])
            continue

        text = _clean_segment_text(seg.text)
        if not text:
            continue

        cleaned_segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
        })

    # ── Merge into paragraphs ─────────────────────────────────────────────────
    paragraphs = _merge_into_paragraphs(
        cleaned_segments,
        gap_seconds=merge_gap_seconds,
        max_words=max_paragraph_words,
    )

    full_text = "\n\n".join(paragraphs)
    full_text = _final_pass(full_text)

    return full_text, cleaned_segments


def clean_text(raw: str) -> str:
    """
    Clean a plain text string (no segment metadata).
    Used when only raw text is available (e.g., tiny/base models).
    """
    text = _strip_artifacts(raw)
    text = _remove_fillers(text)
    text = _collapse_repeats(text)
    text = _fix_punctuation(text)
    text = _normalise_casing(text)
    return text.strip()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_segment_text(text: str) -> str:
    text = text.strip()
    text = _strip_artifacts(text)
    text = _remove_fillers(text)
    text = _collapse_repeats(text)
    text = _fix_punctuation(text)
    text = _normalise_casing(text)
    return text.strip()


def _strip_artifacts(text: str) -> str:
    for pattern in _ARTIFACT_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def _remove_fillers(text: str) -> str:
    return _FILLER_PATTERN.sub(" ", text).strip()


def _collapse_repeats(text: str) -> str:
    # Iteratively collapse until stable (handles triple+ repetitions)
    prev = None
    while prev != text:
        prev = text
        text = _REPEAT_PATTERN.sub(r"\1 ", text)
    return text


def _fix_punctuation(text: str) -> str:
    text = _DOUBLED_PUNCT.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _TRAILING_COMMA.sub(".", text)
    text = _ELLIPSIS_CLEAN.sub("…", text)
    text = _DASH_SPACE.sub(" — ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _normalise_casing(text: str) -> str:
    """Capitalise after sentence-ending punctuation."""
    if not text:
        return text

    # Split on sentence boundaries, capitalise each piece
    parts = re.split(r"([.!?…]\s+)", text)
    result: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # text fragment
            result.append(part[:1].upper() + part[1:] if part else part)
        else:           # separator
            result.append(part)
    text = "".join(result)

    # Always capitalise the very first character
    if text:
        text = text[0].upper() + text[1:]

    return text


def _merge_into_paragraphs(
    segments: list[dict],
    gap_seconds: float,
    max_words: int,
) -> list[str]:
    """
    Group consecutive segments into paragraphs.

    A new paragraph starts when:
      - The gap between segment end and next segment start > gap_seconds, OR
      - The running word count exceeds max_words
    """
    if not segments:
        return []

    paragraphs: list[str] = []
    current_words: list[str] = []
    current_word_count = 0
    prev_end = segments[0]["start"]

    for seg in segments:
        gap = seg["start"] - prev_end
        word_count = len(seg["text"].split())

        start_new = (
            gap > gap_seconds
            or (current_word_count + word_count > max_words and current_words)
        )

        if start_new and current_words:
            para = " ".join(current_words)
            paragraphs.append(_ensure_ends_with_period(para))
            current_words = []
            current_word_count = 0

        current_words.append(seg["text"])
        current_word_count += word_count
        prev_end = seg["end"]

    if current_words:
        para = " ".join(current_words)
        paragraphs.append(_ensure_ends_with_period(para))

    return paragraphs


def _ensure_ends_with_period(text: str) -> str:
    text = text.rstrip()
    if text and text[-1] not in ".!?…":
        text += "."
    return text


def _final_pass(text: str) -> str:
    """Last-mile cleanup on the full assembled transcript."""
    # Remove blank lines > 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix any stray double spaces introduced by paragraph joining
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()
