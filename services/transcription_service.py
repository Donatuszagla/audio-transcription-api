"""
services/transcription_service.py
──────────────────────────────────
Whisper model lifecycle and transcription orchestration.

Uses `faster-whisper` (CTranslate2 backend) for speed + memory efficiency.
faster-whisper is drop-in compatible with openai-whisper models but runs
2–4× faster and uses ~50 % less VRAM.

If you need speaker diarisation, integrate WhisperX on top:
    import whisperx
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN)
    diarize_segments = diarize_model(audio_path)
    result = whisperx.assign_word_speakers(diarize_segments, transcript)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Device resolution ─────────────────────────────────────────────────────────

def _resolve_device() -> tuple[str, str]:
    """Return (device, compute_type) based on available hardware."""
    from config import get_settings
    cfg = get_settings()

    device = cfg.whisper_device
    compute = cfg.whisper_compute_type

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"

    logger.info("Whisper device: %s  compute: %s", device, compute)
    return device, compute


# ── Model singleton ───────────────────────────────────────────────────────────

_model = None
_model_lock = asyncio.Lock()


async def get_model():
    """
    Lazy-load the Whisper model. Thread/coroutine-safe singleton.

    The model is loaded once at first transcription request and kept in memory.
    For multi-worker setups, load model at startup in lifespan() instead.
    """
    global _model
    if _model is not None:
        return _model

    async with _model_lock:
        if _model is not None:   # double-checked locking
            return _model

        from config import get_settings
        from faster_whisper import WhisperModel

        cfg     = get_settings()
        device, compute = _resolve_device()

        logger.info(
            "Loading Whisper model '%s' on %s (%s) — this may take a minute…",
            cfg.whisper_model, device, compute,
        )
        t0 = time.time()

        # Run blocking model load in executor so we don't block the event loop
        loop = asyncio.get_running_loop()
        _model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                cfg.whisper_model,
                device=device,
                compute_type=compute,
                download_root=os.environ.get("WHISPER_CACHE", None),
                num_workers=1,
            ),
        )
        logger.info("Model loaded in %.1f s", time.time() - t0)
    return _model


# ── Core transcription function ───────────────────────────────────────────────

async def transcribe_file(
    audio_path: str | Path,
    *,
    language: str | None = None,
    clean_output: bool = True,
    beam_size: int | None = None,
) -> dict[str, Any]:
    """
    Transcribe a preprocessed WAV file.

    Parameters
    ----------
    audio_path : path to the 16 kHz mono WAV produced by preprocess_audio()
    language   : ISO-639-1 code ("en", "es"…) or None for auto-detect
    clean_output : apply text cleaning pipeline
    beam_size  : override default beam_size from config

    Returns
    -------
    dict with keys: text, language, duration, segments, model
    """
    from config import get_settings
    from utils import clean_segments, Segment, get_duration_seconds

    cfg   = get_settings()
    model = await get_model()

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    beam = beam_size or cfg.whisper_beam_size

    logger.info("Transcribing %s (beam_size=%d, lang=%s)", audio_path.name, beam, language or "auto")
    t0 = time.time()

    # Run blocking Whisper inference in executor
    loop = asyncio.get_running_loop()

    def _run_whisper():
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=language if language else None,
            beam_size=beam,
            vad_filter=cfg.whisper_vad_filter,
            vad_parameters={
                "min_silence_duration_ms": 800,
                "speech_pad_ms": 200,
            },
            word_timestamps=False,
            condition_on_previous_text=True,   # improves coherence for long audio
            # Compression ratio threshold: drop segments that look repeated
            compression_ratio_threshold=2.4,
            # Log prob threshold: drop very uncertain segments
            log_prob_threshold=-1.0,
            # No-speech threshold: high → keep more speech-y content
            no_speech_threshold=0.6,
        )
        # Materialise the generator (required before leaving executor)
        return list(segments_gen), info

    raw_segments, info = await loop.run_in_executor(None, _run_whisper)

    elapsed = time.time() - t0
    duration = get_duration_seconds(audio_path)
    rtf = elapsed / duration if duration else 0   # real-time factor

    logger.info(
        "Whisper done: %.1f s audio in %.1f s (RTF %.2fx), %d segments, lang=%s",
        duration, elapsed, rtf, len(raw_segments), info.language,
    )

    # ── Convert to our Segment dataclass ─────────────────────────────────────
    segs = [
        Segment(
            text=s.text,
            start=s.start,
            end=s.end,
            no_speech_prob=s.no_speech_prob,
            avg_logprob=s.avg_logprob,
        )
        for s in raw_segments
    ]

    # ── Clean output ──────────────────────────────────────────────────────────
    if clean_output:
        full_text, cleaned_segs = clean_segments(
            segs,
            min_confidence=cfg.min_segment_confidence,
        )
    else:
        full_text   = " ".join(s.text.strip() for s in segs)
        cleaned_segs = [
            {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs
        ]

    return {
        "text":     full_text,
        "language": info.language,
        "duration": round(duration, 2),
        "segments": cleaned_segs,
        "model":    cfg.whisper_model,
        "meta": {
            "inference_seconds": round(elapsed, 1),
            "real_time_factor":  round(rtf, 2),
            "segment_count":     len(cleaned_segs),
        },
    }
