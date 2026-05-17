# Sermon Transcription API

A production-ready, self-hosted audio transcription API powered by **OpenAI Whisper large-v3** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Optimised for long-form speech (multi-hour sermons) with automatic audio preprocessing, filler-word removal, and clean paragraph output.

---

## Quick Start

### Option A — Local (bare metal)

**Prerequisites**

- Python 3.10+
- ffmpeg (required for audio preprocessing)

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

**Install & run**

```bash
git clone <repo>
cd audio-transcription-api

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env              # edit if needed

uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

The first request downloads the Whisper model (~3 GB for large-v3). Subsequent starts are instant.

---

### Option B — Docker (recommended for production)

```bash
docker compose up --build
```

GPU support: uncomment the GPU service in `docker-compose.yml` and ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

---

## API Reference

### `POST /transcribe`

Submit an audio file for transcription. Returns a `job_id` immediately.

| Parameter      | Type    | Default  | Description                                    |
|----------------|---------|----------|------------------------------------------------|
| `file`         | File    | required | Audio file (mp3, wav, m4a, flac, aac, ogg)    |
| `language`     | string  | auto     | ISO-639-1 code e.g. `en`, `es`, `fr`          |
| `clean_output` | boolean | `true`   | Apply filler removal + paragraph merging       |

**Response (202 Accepted)**

```json
{
  "job_id": "3f2a...",
  "status": "queued",
  "filename": "sermon.mp3",
  "message": "Job accepted. Poll GET /status/{job_id} for results."
}
```

---

### `GET /status/{job_id}`

Poll for job result.

**Possible `status` values:** `queued` → `processing` → `done` | `failed`

**Response when done**

```json
{
  "job_id": "3f2a...",
  "status": "done",
  "filename": "sermon.mp3",
  "elapsed_seconds": 312.4,
  "result": {
    "text": "Good morning, brothers and sisters. Today we will explore...",
    "language": "en",
    "duration": 5402.3,
    "segments": [
      { "start": 0.0, "end": 4.2, "text": "Good morning, brothers and sisters." },
      ...
    ],
    "model": "large-v3",
    "meta": {
      "inference_seconds": 298.1,
      "real_time_factor": 0.055,
      "segment_count": 284
    }
  },
  "error": null
}
```

---

### `GET /health`

```json
{ "status": "ok", "uptime_s": 3600.1, "python": "3.11.0" }
```

### `GET /ready`

Returns `200` when the Whisper model is loaded, `200` with `ready: false` before.

---

## Example cURL Requests

### Submit a job

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@/path/to/sermon.mp3" \
  -F "language=en" \
  -F "clean_output=true"
```

### Poll for result

```bash
curl http://localhost:8000/status/3f2a9c1d-...
```

### Submit without language hint (auto-detect)

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@/path/to/sermon.wav"
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## Text Cleaning Pipeline

When `clean_output=true` (default), the following transformations are applied:

| Step | What it does |
|---|---|
| Artifact removal | Strips `[Music]`, `(inaudible)`, `♪`, `<laugh>` markers |
| Filler removal | Removes `um`, `uh`, `hmm`, `er`, `ah`, and variants |
| Repeat collapse | Collapses `the the the` → `the`, repeated phrases |
| Punctuation fix | Removes doubled punctuation, fixes spacing |
| Casing normalisation | Capitalises after `.!?` |
| Paragraph merging | Groups segments separated by < 2.5 s gaps into paragraphs |
| Final pass | Collapses blank lines, trims whitespace |

Low-confidence and near-silent Whisper segments are **dropped entirely** rather than guessed.

---

## Audio Preprocessing Pipeline

Every file goes through ffmpeg before Whisper sees it:

1. **Decode** any format → 16 kHz mono PCM WAV (Whisper's native input)
2. **High-pass filter** @ 80 Hz — removes AC hum / low rumble
3. **Silence stripping** — removes leading/trailing silence > 1 s @ −50 dBFS
4. **Loudness normalisation** — EBU R128 two-pass, target −23 LUFS
5. **Duration validation** — rejects < 0.5 s files

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `WHISPER_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `WHISPER_COMPUTE_TYPE` | `auto` | `float16` (GPU) / `int8` (CPU) / `float32` |
| `WHISPER_BEAM_SIZE` | `5` | Higher = slightly better accuracy, slower |
| `WHISPER_VAD_FILTER` | `true` | Skip silent segments before transcription |
| `MAX_UPLOAD_MB` | `2048` | Maximum file size (2 GB) |
| `MAX_CONCURRENT_JOBS` | `2` | Simultaneous transcription jobs |
| `RATE_LIMIT_PER_MINUTE` | `10` | Requests to `/transcribe` per IP/minute |
| `JOB_TTL_SECONDS` | `3600` | How long to keep job results |

---

## Performance Notes

| Model | VRAM / RAM | Speed (CPU) | Speed (GPU, A10) | Quality |
|---|---|---|---|---|
| `tiny` | ~1 GB | ~32× RT | — | Low |
| `small` | ~2 GB | ~8× RT | — | Medium |
| `medium` | ~5 GB | ~3× RT | ~20× RT | Good |
| `large-v3` | ~10 GB | ~1× RT | ~8× RT | Best |

**Real-time factor (RTF):** How fast the system processes audio relative to its duration.  
`RTF 0.12` = 1 hour of audio processed in ~7 minutes.

A 2-hour sermon on `large-v3`:
- **GPU (A10/A100):** ~14–20 minutes
- **CPU (8-core):** ~90–120 minutes

---

## Scaling for Large Sermon Uploads

### Immediate (single server)
- Run with 1 Uvicorn worker (Whisper model is shared, more workers = OOM)
- Increase `MAX_CONCURRENT_JOBS` only if you have multiple GPUs
- Mount a fast NVMe drive for `UPLOAD_DIR`

### Horizontal scaling (multi-server)
Replace FastAPI `BackgroundTasks` with **Celery + Redis**:

```
# Worker
celery -A celery_worker.celery_app worker --loglevel=info --concurrency=1

# API server just enqueues, workers pick up tasks
```

Files must live on shared storage (S3, NFS, or object store).

### GPU (strongly recommended for production)
- Any NVIDIA GPU with ≥ 10 GB VRAM (A10, RTX 3090, A100)
- Set `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`
- 8–12× faster than CPU

### Speaker Diarisation (optional upgrade)
To add speaker labels, replace faster-whisper with **WhisperX**:

```bash
pip install whisperx
```

Then in `services/transcription_service.py`:
```python
import whisperx
model    = whisperx.load_model("large-v3", device, compute_type=compute_type)
audio    = whisperx.load_audio(str(audio_path))
result   = model.transcribe(audio)
diarize  = whisperx.DiarizationPipeline(use_auth_token="hf_...")
segments = diarize(audio)
result   = whisperx.assign_word_speakers(segments, result)
```

---

## Project Structure

```
audio-transcription-api/
├── main.py                          ← FastAPI app, middleware, lifespan
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── config/
│   └── settings.py                  ← Pydantic settings (env vars)
├── api/
│   └── routes/
│       ├── health.py                ← GET /health, /ready
│       └── transcription.py        ← POST /transcribe, GET /status/{id}
├── services/
│   ├── transcription_service.py    ← Whisper model lifecycle + inference
│   └── job_store.py                ← In-memory job tracker with TTL
├── utils/
│   ├── audio.py                    ← ffmpeg preprocessing pipeline
│   └── text_cleaner.py             ← Filler removal, paragraph merging
├── logs/                           ← Rotating log files
└── tmp/                            ← Temp upload/processing files
```





