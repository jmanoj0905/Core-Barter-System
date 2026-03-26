import logging
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import httpx
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("audio-pipeline")
logging.basicConfig(level=logging.INFO)

SEMANTIC_URL = "http://localhost:8002"
WARNING_ENGINE_URL = "http://localhost:8003"
BACKEND_URL = "http://localhost:8000"

# Toxicity thresholds (OpenAI Moderation API returns scores 0.0-1.0)
TOXICITY_FLAG_THRESHOLD = 0.7
TOXICITY_BLOCK_THRESHOLD = 0.9

# Trigger transcription after this many seconds of buffered audio
BUFFER_THRESHOLD_SECONDS = 15.0

# ---------------------------------------------------------------------------
# In-memory state: (barter_id, user_id) -> buffer dict
# ---------------------------------------------------------------------------

buffers: dict[tuple[int, int], dict] = {}


def _new_buffer() -> dict:
    return {
        "chunks": [],              # list of raw bytes
        "header_chunk": None,      # first chunk contains webm header — always prepend
        "accumulated_seconds": 0.0,
        "segment_counter": 0,
        "wall_start": time.time(),
    }


# ---------------------------------------------------------------------------
# ML model + HTTP client (loaded once at startup)
# ---------------------------------------------------------------------------

whisper_model: WhisperModel | None = None
http_client: httpx.AsyncClient | None = None
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, http_client
    logger.info("Loading faster-whisper base model (CTranslate2 int8) ...")
    whisper_model = WhisperModel("base", compute_type="int8")
    logger.info("faster-whisper model loaded.")
    http_client = httpx.AsyncClient(timeout=30.0)
    yield
    await http_client.aclose()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Audio Pipeline", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def convert_to_wav(webm_bytes: bytes) -> str:
    """Write webm bytes to a temp file, convert to 16kHz mono WAV, return WAV path."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(webm_bytes)
        webm_path = f.name

    wav_path = webm_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", webm_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                wav_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    finally:
        os.unlink(webm_path)

    return wav_path


def transcribe(wav_path: str) -> str:
    """Run faster-whisper on a WAV file, return transcript text."""
    try:
        segments, _info = whisper_model.transcribe(wav_path, language="en")
        text = " ".join(seg.text for seg in segments).strip()
        return text
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


async def post_segment(barter_id: int, user_id: int, text: str,
                        duration: float, ts_start: float, ts_end: float):
    payload = {
        "barter_id": barter_id,
        "user_id": user_id,
        "text": text,
        "duration_seconds": duration,
        "timestamp_start": ts_start,
        "timestamp_end": ts_end,
    }
    try:
        resp = await http_client.post(f"{SEMANTIC_URL}/ingest/segment", json=payload)
        resp.raise_for_status()
        logger.info(
            "Segment posted: barter=%d user=%d dur=%.1fs",
            barter_id, user_id, duration,
        )
    except Exception as e:
        logger.error("Failed to POST segment to semantic engine: %s", e)


async def check_toxicity(text: str) -> dict | None:
    """Check transcript against Mistral Moderation API. Returns flagged info or None."""
    try:
        response = await http_client.post(
            "https://api.mistral.ai/v1/moderations",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={"model": "mistral-moderation-latest", "input": text},
        )
        response.raise_for_status()
        result = response.json()["results"][0]
        if not result["flagged"]:
            return None
        scores = result.get("category_scores", {})
        flagged_categories = {
            cat: round(score, 4)
            for cat, score in scores.items()
            if score >= TOXICITY_FLAG_THRESHOLD
        }
        return {
            "flagged": True,
            "categories": flagged_categories,
            "hard_block": any(s >= TOXICITY_BLOCK_THRESHOLD for s in flagged_categories.values()),
        }
    except Exception as e:
        logger.error("Mistral Moderation API call failed: %s", e)
        return None  # fail-open: don't block on API errors


async def post_safety_warning(barter_id: int, user_id: int, warning_type: str, details: dict):
    """Forward a safety alert (toxicity/NSFW) to the warning engine."""
    payload = {
        "barter_id": barter_id,
        "user_id": user_id,
        "warning_type": warning_type,
        "details": details,
    }
    try:
        resp = await http_client.post(f"{WARNING_ENGINE_URL}/safety/alert", json=payload)
        resp.raise_for_status()
        logger.info("Safety warning posted: barter=%d type=%s", barter_id, warning_type)
    except Exception as e:
        logger.error("Failed to POST safety warning: %s", e)


async def process_buffer(barter_id: int, user_id: int, buf: dict):
    """Concatenate buffered chunks, transcribe, check toxicity, post segment if non-empty."""
    if not buf["chunks"]:
        return

    audio_bytes = b"".join(buf["chunks"])
    accumulated = buf["accumulated_seconds"]
    ts_start = buf["wall_start"]
    ts_end = ts_start + accumulated

    buf["segment_counter"] += 1
    logger.info(
        "Processing buffer: barter=%d user=%d seg=%d size=%d bytes dur=%.1fs",
        barter_id, user_id, buf["segment_counter"], len(audio_bytes), accumulated,
    )

    try:
        wav_path = convert_to_wav(audio_bytes)
        text = transcribe(wav_path)
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return

    if not text:
        logger.info("Empty transcription, skipping segment.")
        return

    logger.info("Transcript: %s", text[:120])

    # Safety check: run toxicity detection on transcript
    toxicity_result = await check_toxicity(text)
    if toxicity_result:
        logger.warning("Toxicity detected in barter=%d user=%d: %s",
                        barter_id, user_id, toxicity_result["categories"])
        await post_safety_warning(barter_id, user_id, "toxicity", toxicity_result)

    # Save transcript to backend DB
    try:
        await http_client.post(
            f"{BACKEND_URL}/session/{barter_id}/transcript",
            json={
                "barter_id": barter_id,
                "user_id": user_id,
                "text": text,
                "duration_seconds": accumulated,
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
            },
        )
    except Exception as e:
        logger.error("Failed to save transcript segment: %s", e)

    # Forward to semantic analysis regardless (toxicity doesn't block topic analysis)
    await post_segment(barter_id, user_id, text, accumulated, ts_start, ts_end)


def reset_buffer(buf: dict):
    # Keep header_chunk — it must be prepended to every new segment
    buf["chunks"] = [buf["header_chunk"]] if buf["header_chunk"] else []
    buf["accumulated_seconds"] = 0.0
    buf["wall_start"] = time.time()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/audio/{barter_id}/{user_id}")
async def audio_ws(barter_id: int, user_id: int, ws: WebSocket):
    await ws.accept()
    key = (barter_id, user_id)
    buffers[key] = _new_buffer()
    buf = buffers[key]

    logger.info("Audio WS connected: barter=%d user=%d", barter_id, user_id)

    try:
        while True:
            chunk = await ws.receive_bytes()
            if not chunk:
                continue

            if buf["header_chunk"] is None:
                buf["header_chunk"] = chunk  # save webm container header
            buf["chunks"].append(chunk)
            buf["accumulated_seconds"] = time.time() - buf["wall_start"]

            if buf["accumulated_seconds"] >= BUFFER_THRESHOLD_SECONDS:
                await process_buffer(barter_id, user_id, buf)
                reset_buffer(buf)

    except (WebSocketDisconnect, RuntimeError):
        logger.info("Audio WS disconnected: barter=%d user=%d", barter_id, user_id)
        if buf["chunks"]:
            await process_buffer(barter_id, user_id, buf)
        if key in buffers:
            del buffers[key]


# ---------------------------------------------------------------------------
# Session end endpoint (called by backend when both users confirm)
# ---------------------------------------------------------------------------


@app.post("/session/{barter_id}/end")
async def end_session(barter_id: int):
    """Flush all buffers for this barter session and notify semantic engine."""
    flushed_users = []
    keys_to_delete = [k for k in buffers if k[0] == barter_id]

    for key in keys_to_delete:
        user_id = key[1]
        buf = buffers[key]
        if buf["chunks"]:
            await process_buffer(barter_id, user_id, buf)
            flushed_users.append(user_id)
        del buffers[key]

    try:
        resp = await http_client.post(f"{SEMANTIC_URL}/session/{barter_id}/end")
        resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to notify semantic engine of session end: %s", e)

    logger.info("Session %d ended, flushed users: %s", barter_id, flushed_users)
    return {"status": "ended", "barter_id": barter_id, "flushed_users": flushed_users}


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"service": "audio-pipeline", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
