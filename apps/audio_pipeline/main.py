import asyncio
import logging
import os
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import httpx
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Terminal colours ────────────────────────────────────────────────────────
_R = "\033[0;31m";  _G = "\033[0;32m";  _Y = "\033[1;33m"
_C = "\033[0;36m";  _M = "\033[0;35m";  _B = "\033[1;34m"
_W = "\033[1;37m";  _NC = "\033[0m";    _BOLD = "\033[1m"

def _banner(msg): print(f"\n{_B}{_BOLD}{'─'*56}{_NC}\n  {_W}{_BOLD}{msg}{_NC}\n{_B}{_BOLD}{'─'*56}{_NC}", flush=True)
def _ok(msg):     print(f"  {_G}✓{_NC}  {msg}", flush=True)
def _info(msg):   print(f"  {_C}→{_NC}  {msg}", flush=True)
def _warn(msg):   print(f"  {_Y}⚠{_NC}  {_Y}{msg}{_NC}", flush=True)
def _tx(msg):     print(f"  {_M}🎙 {_NC} {msg}", flush=True)
def _safety(msg): print(f"  {_R}🚨 {_BOLD}SAFETY{_NC}  {_R}{msg}{_NC}", flush=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("audio-pipeline")

SEMANTIC_URL      = os.getenv("SEMANTIC_URL",      "http://localhost:8002")
WARNING_ENGINE_URL = os.getenv("WARNING_ENGINE_URL", "http://localhost:8003")
BACKEND_URL       = os.getenv("BACKEND_URL",       "http://localhost:8000")
DEEPGRAM_API_KEY  = os.getenv("DEEPGRAM_API_KEY",  "")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.getenv("AWS_REGION",            "ap-south-1")
AWS_S3_BUCKET         = os.getenv("AWS_S3_BUCKET",         "")

# Toxicity thresholds (OpenAI Moderation API returns scores 0.0-1.0)
TOXICITY_FLAG_THRESHOLD = 0.7
TOXICITY_BLOCK_THRESHOLD = 0.9

# Trigger transcription after this many seconds of buffered audio
BUFFER_THRESHOLD_SECONDS = 5.0

# ---------------------------------------------------------------------------
# STT backend — runtime-switchable: "whisper" | "deepgram" | "aws"
# ---------------------------------------------------------------------------

_VALID_BACKENDS = {"whisper", "deepgram", "aws"}
current_stt_backend: str = os.getenv("STT_BACKEND", "whisper")

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
    _banner("Audio Pipeline  ·  Port 8001")
    _info(f"STT backend: {current_stt_backend}")
    _info("Loading faster-whisper  large-v3  (CTranslate2 int8) …")
    whisper_model = WhisperModel("large-v3", compute_type="int8")
    _ok("faster-whisper model ready")
    if current_stt_backend == "deepgram" and not DEEPGRAM_API_KEY:
        _warn("DEEPGRAM_API_KEY not set — Deepgram calls will fail")
    http_client = httpx.AsyncClient(timeout=30.0)
    _ok("Service online — waiting for audio streams")
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


_HALLUCINATION_PHRASES = {
    "thank you", "thank you.", "thank you for watching",
    "thank you for watching.", "thanks for watching.", "you", "you.",
    "bye", "bye.", "bye bye", "bye bye.", ".",
}

def transcribe(wav_path: str) -> str:
    """Run faster-whisper on a WAV file, return transcript text."""
    try:
        segments, _info = whisper_model.transcribe(
            wav_path,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(seg.text for seg in segments).strip()
        if text.lower() in _HALLUCINATION_PHRASES:
            return ""
        return text
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


async def transcribe_deepgram(wav_path: str) -> str:
    """Send a WAV file to Deepgram prerecorded API, return transcript."""
    try:
        with open(wav_path, "rb") as f:
            audio_data = f.read()
        resp = await http_client.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&language=en&punctuate=true",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            content=audio_data,
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript.strip()
    except Exception as e:
        logger.error("Deepgram transcription failed: %s", e)
        return ""
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


async def transcribe_aws(wav_path: str) -> str:
    """Upload WAV to S3, run AWS Transcribe job, return transcript, delete from S3."""
    import boto3
    job_name = f"barter-{uuid.uuid4().hex}"
    s3_key = f"tmp/{job_name}.wav"
    s3_uri = f"s3://{AWS_S3_BUCKET}/{s3_key}"

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    s3 = session.client("s3")
    transcribe_client = session.client("transcribe")

    try:
        s3.upload_file(wav_path, AWS_S3_BUCKET, s3_key)

        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat="wav",
            LanguageCode="en-IN",
            Settings={"ShowAlternatives": False},
        )

        # Poll until complete (max 60s for short segments)
        for _ in range(60):
            await asyncio.sleep(1)
            resp = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
            if status == "COMPLETED":
                break
            if status == "FAILED":
                reason = resp["TranscriptionJob"].get("FailureReason", "unknown")
                logger.error("AWS Transcribe job failed: %s", reason)
                return ""

        transcript_uri = resp["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        import urllib.request, json as _json
        with urllib.request.urlopen(transcript_uri) as r:
            data = _json.loads(r.read())
        text = data["results"]["transcripts"][0]["transcript"].strip()
        return text

    except Exception as e:
        logger.error("AWS Transcribe failed: %s", e)
        return ""
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        try:
            s3.delete_object(Bucket=AWS_S3_BUCKET, Key=s3_key)
        except Exception:
            pass


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
        _ok(f"Segment → semantic  barter={barter_id}  user={user_id}  dur={duration:.1f}s")
    except Exception as e:
        logger.error("Failed to POST segment to semantic engine: %s", e)


async def check_toxicity(text: str) -> dict | None:
    """Check transcript against Mistral Moderation API. Returns flagged info or None."""
    if not MISTRAL_API_KEY:
        return None
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
        _safety(f"Warning posted  barter={barter_id}  type={warning_type}")
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
    seg_n = buf["segment_counter"]
    _info(f"Transcribing  barter={barter_id}  user={user_id}  seg={seg_n}  ({len(audio_bytes)//1024}KB  {accumulated:.1f}s) …")

    try:
        wav_path = convert_to_wav(audio_bytes)
        if current_stt_backend == "deepgram":
            text = await transcribe_deepgram(wav_path)
        elif current_stt_backend == "aws":
            text = await transcribe_aws(wav_path)
        elif current_stt_backend == "whisper":
            text = transcribe(wav_path)
        else:
            logger.error("STT backend '%s' not implemented — skipping segment", current_stt_backend)
            return
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return

    if not text:
        _warn(f"Empty transcription  barter={barter_id}  user={user_id}  — skipping")
        return

    preview = text[:100] + ("…" if len(text) > 100 else "")
    _tx(f"barter={barter_id}  user={user_id}  \"{preview}\"")

    # Safety check: run toxicity detection on transcript
    toxicity_result = await check_toxicity(text)
    if toxicity_result:
        cats = ", ".join(toxicity_result.get("categories", {}).keys())
        _safety(f"Toxicity detected  barter={barter_id}  user={user_id}  [{cats}]")
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
# STT config endpoints
# ---------------------------------------------------------------------------

from fastapi import HTTPException
from pydantic import BaseModel

class SttConfigRequest(BaseModel):
    backend: str

@app.get("/stt/config")
async def get_stt_config():
    return {"backend": current_stt_backend, "available": sorted(_VALID_BACKENDS)}

@app.post("/stt/config")
async def set_stt_config(body: SttConfigRequest):
    global current_stt_backend
    if body.backend not in _VALID_BACKENDS:
        raise HTTPException(400, f"Unknown backend '{body.backend}'. Choose from: {sorted(_VALID_BACKENDS)}")
    current_stt_backend = body.backend
    _ok(f"STT backend → {current_stt_backend}")
    return {"backend": current_stt_backend}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/audio/{barter_id}/{user_id}")
async def audio_ws(barter_id: int, user_id: int, ws: WebSocket):
    await ws.accept()
    key = (barter_id, user_id)
    buffers[key] = _new_buffer()
    buf = buffers[key]

    _ok(f"Audio stream connected  barter={barter_id}  user={user_id}  (buffering {BUFFER_THRESHOLD_SECONDS:.0f}s windows)")

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
        _info(f"Audio stream closed  barter={barter_id}  user={user_id}")
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

    _ok(f"Session {barter_id} ended — flushed users: {flushed_users}")
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
