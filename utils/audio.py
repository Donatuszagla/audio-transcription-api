"""
utils/audio.py
──────────────
FFmpeg-based audio preprocessing.

Pipeline applied before Whisper sees the file:
  1. Decode any format → 16 kHz mono PCM WAV   (Whisper's native input)
  2. Loudness normalisation  (EBU R128, target -23 LUFS)
  3. High-pass filter @ 80 Hz                   (remove low rumble / AC hum)
  4. Strip leading / trailing silence
  5. Validate the result is non-empty

Every step calls ffmpeg as a subprocess — no Python audio libraries needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it: apt-get install ffmpeg"
        )


def _run(cmd: list[str], *, description: str = "") -> subprocess.CompletedProcess:
    """Run a subprocess, raise on failure with captured stderr."""
    logger.debug("ffmpeg cmd [%s]: %s", description, " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed during '{description}':\n{result.stderr[-2000:]}"
        )
    return result


def get_duration_seconds(path: str | Path) -> float:
    """Return audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def validate_extension(filename: str, allowed: set[str]) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )


# ── main preprocessing function ───────────────────────────────────────────────

def preprocess_audio(input_path: str | Path, output_dir: str | Path) -> Path:
    """
    Full preprocessing pipeline.

    Returns
    -------
    Path
        Path to a 16 kHz mono WAV file ready for Whisper.

    Raises
    ------
    RuntimeError
        If the file is corrupt, silent, or ffmpeg is unavailable.
    """
    _require_ffmpeg()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = uuid.uuid4().hex
    stage1 = output_dir / f"{stem}_stage1.wav"
    stage2 = output_dir / f"{stem}_stage2.wav"

    # ── Stage 1: decode + resample + mono + high-pass + strip silence ─────────
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            # Audio filters:
            # highpass=f=80        → remove sub-80 Hz rumble
            # silenceremove        → strip leading and trailing silence (>1 s @ -50 dBFS)
            # aresample=16000      → resample to 16 kHz
            # pan=mono             → downmix to mono
            "-af",
            (
                "highpass=f=80,"
                "silenceremove=start_periods=1:start_silence=1:start_threshold=-50dB"
                ":stop_periods=-1:stop_silence=1:stop_threshold=-50dB,"
                "aresample=16000,"
                "pan=mono|c0=0.5*c0+0.5*c1"        # safe stereo→mono
            ),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            "-vn",                                  # strip any video stream
            str(stage1),
        ],
        description="decode+resample+silence-strip",
    )

    # ── Stage 2: loudness normalization (EBU R128 two-pass) ───────────────────
    # First pass: measure integrated loudness
    measure = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(stage1),
            "-af", "loudnorm=I=-23:TP=-2:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    stderr = measure.stderr

    # Parse loudnorm JSON from stderr
    import json, re  # noqa: E401
    match = re.search(r"\{[^{}]+\}", stderr, re.DOTALL)
    if match:
        try:
            stats = json.loads(match.group())
            il  = stats.get("input_i",  "-23.0")
            lra = stats.get("input_lra", "11.0")
            tp  = stats.get("input_tp",  "-2.0")
            thr = stats.get("input_thresh", "-33.0")
            ofs = stats.get("target_offset", "0.0")
            norm_filter = (
                f"loudnorm=I=-23:TP=-2:LRA=11:linear=true:"
                f"measured_I={il}:measured_LRA={lra}:"
                f"measured_TP={tp}:measured_thresh={thr}:"
                f"offset={ofs}"
            )
        except (json.JSONDecodeError, KeyError):
            norm_filter = "loudnorm=I=-23:TP=-2:LRA=11"
    else:
        norm_filter = "loudnorm=I=-23:TP=-2:LRA=11"

    _run(
        [
            "ffmpeg", "-y",
            "-i", str(stage1),
            "-af", norm_filter,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(stage2),
        ],
        description="loudness-normalize",
    )

    # Cleanup intermediate
    stage1.unlink(missing_ok=True)

    # ── Sanity check: duration must be > 0.5 s ────────────────────────────────
    duration = get_duration_seconds(stage2)
    if duration < 0.5:
        stage2.unlink(missing_ok=True)
        raise RuntimeError(
            "Preprocessed audio is too short or entirely silent. "
            "Check your source file."
        )

    logger.info("Preprocessing complete: %.1f s -> %s", duration, stage2.name)
    return stage2


# ── async wrapper ─────────────────────────────────────────────────────────────

async def preprocess_audio_async(
    input_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Non-blocking wrapper around preprocess_audio."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, preprocess_audio, input_path, output_dir)
