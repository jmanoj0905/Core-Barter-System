import logging
import re
import time
from contextlib import asynccontextmanager

import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# ── Terminal colours ────────────────────────────────────────────────────────
_R = "\033[0;31m";  _G = "\033[0;32m";  _Y = "\033[1;33m"
_C = "\033[0;36m";  _M = "\033[0;35m";  _B = "\033[1;34m"
_W = "\033[1;37m";  _NC = "\033[0m";    _BOLD = "\033[1m";  _DIM = "\033[2m"

def _banner(msg):  print(f"\n{_B}{_BOLD}{'─'*56}{_NC}\n  {_W}{_BOLD}{msg}{_NC}\n{_B}{_BOLD}{'─'*56}{_NC}", flush=True)
def _ok(msg):      print(f"  {_G}✓{_NC}  {msg}", flush=True)
def _info(msg):    print(f"  {_C}→{_NC}  {msg}", flush=True)
def _warn(msg):    print(f"  {_Y}⚠{_NC}  {_Y}{msg}{_NC}", flush=True)

_CLASS_COLOR = {
    "correct":        f"{_G}✓ CORRECT       {_NC}",
    "weakly_correct": f"{_Y}~ WEAKLY CORRECT{_NC}",
    "incorrect":      f"{_R}✗ INCORRECT     {_NC}",
    "out_of_scope":   f"{_M}⊘ OUT OF SCOPE  {_NC}",
}

def _window(barter_id, window_id, classification, similarity, preview):
    label = _CLASS_COLOR.get(classification, classification)
    bar_len = int(similarity * 20)
    bar = f"{_G}{'█' * bar_len}{_DIM}{'░' * (20 - bar_len)}{_NC}"
    print(f"\n  {_BOLD}[Barter {barter_id}  Window #{window_id}]{_NC}", flush=True)
    print(f"  {label}  sim={_BOLD}{similarity:.3f}{_NC}  [{bar}]", flush=True)
    print(f"  {_DIM}\"{preview[:80]}…\"{_NC}", flush=True)

def _engagement(barter_id, score, ratio_pct, questions, acks):
    bar_len = int(score * 20)
    bar = f"{_C}{'█' * bar_len}{_DIM}{'░' * (20 - bar_len)}{_NC}"
    print(f"  {_C}📊{_NC}  Barter {barter_id}  learner engagement={_BOLD}{score:.2f}{_NC}  [{bar}]  ratio={ratio_pct:.0f}%  q={questions}  ack={acks}", flush=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("semantic-analysis")

import os
WARNING_ENGINE_URL = os.getenv("WARNING_ENGINE_URL", "http://localhost:8003")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Cosine similarity thresholds
UPPER = 0.55  # >= UPPER → correct
LOWER = 0.35  # LOWER..UPPER → weakly_correct, < LOWER → incorrect

# Filler words to strip before embedding
FILLER_WORDS = {
    "uh", "um", "er", "ah", "like", "you know", "i mean",
    "basically", "literally", "actually", "so", "well", "okay",
}

# Window triggers when accumulated audio >= this many seconds
WINDOW_DURATION_THRESHOLD = 25.0

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ContractRequest(BaseModel):
    barter_id: int
    topic: str
    scope: str
    teacher_user_id: int = 1
    learner_user_id: int = 2


class SegmentRequest(BaseModel):
    barter_id: int
    user_id: int
    text: str
    duration_seconds: float
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

# barter_id -> contract info + cached topic embedding
contracts: dict[int, dict] = {}

# barter_id -> window buffer (teacher segments only)
buffers: dict[int, dict] = {}

# barter_id -> learner engagement tracking
engagement_state: dict[int, dict] = {}


def _new_buffer() -> dict:
    return {
        "segments": [],          # list of {"text": str, "duration": float, "ts_start": float, "ts_end": float}
        "accumulated_seconds": 0.0,
        "window_counter": 0,
    }


def _new_engagement() -> dict:
    return {
        "teacher_speaking_seconds": 0.0,
        "learner_speaking_seconds": 0.0,
        "learner_question_count": 0,
        "learner_acknowledgment_count": 0,
        "learner_segment_count": 0,
        "last_score": 0.5,
        "last_alert_time": 0.0,
    }


# ---------------------------------------------------------------------------
# ML model (loaded once at startup)
# ---------------------------------------------------------------------------

model: SentenceTransformer | None = None
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, http_client
    _banner("Semantic Analysis  ·  Port 8002")
    _info("Loading sentence-transformers  all-MiniLM-L6-v2  (384-dim, 22M params) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    _ok("Model ready  —  UPPER=0.55  LOWER=0.35  window=15s")
    http_client = httpx.AsyncClient(timeout=10.0)
    _ok("Service online — waiting for session contracts")
    yield
    await http_client.aclose()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Semantic Analysis", lifespan=lifespan)

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


def clean_text(text: str) -> str:
    """Strip filler words and extra whitespace."""
    tokens = text.lower().split()
    cleaned = [t for t in tokens if t not in FILLER_WORDS]
    return " ".join(cleaned).strip()


def embed(text: str):
    return model.encode(text, convert_to_tensor=True)


def cosine_sim(a, b) -> float:
    return float(util.cos_sim(a, b).item())


def classify(similarity: float) -> str:
    if similarity >= UPPER:
        return "correct"
    if similarity >= LOWER:
        return "weakly_correct"
    return "incorrect"


# ---------------------------------------------------------------------------
# Engagement scoring helpers
# ---------------------------------------------------------------------------

QUESTION_PATTERNS = re.compile(
    r'\b(how|what|why|when|where|which|can you|could you|is it|are there)\b|\?'
)
ACKNOWLEDGMENT_WORDS = {
    "uh-huh", "okay", "right", "yes", "yeah", "got it",
    "i see", "makes sense", "interesting", "cool",
}

ENGAGEMENT_ALERT_COOLDOWN = 120.0  # seconds between alerts


def count_questions(text: str) -> int:
    return len(QUESTION_PATTERNS.findall(text.lower()))


def count_acknowledgments(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for phrase in ACKNOWLEDGMENT_WORDS if phrase in text_lower)


def calculate_engagement_score(state: dict) -> float:
    total_speaking = state["teacher_speaking_seconds"] + state["learner_speaking_seconds"]
    if total_speaking < 30.0:
        return 0.5  # not enough data yet

    ratio = state["learner_speaking_seconds"] / total_speaking

    # Speaking ratio score: 20-40% ideal
    if 0.20 <= ratio <= 0.40:
        ratio_score = 1.0
    elif ratio < 0.10:
        ratio_score = 0.2
    elif ratio < 0.20:
        ratio_score = 0.5 + (ratio - 0.10) * 5.0
    else:
        ratio_score = max(0.4, 1.0 - (ratio - 0.40) * 2.0)

    # Question frequency: 1 per 3 min = good
    minutes = total_speaking / 60.0
    question_rate = state["learner_question_count"] / max(minutes, 1.0)
    question_score = min(1.0, question_rate / 0.33)

    # Acknowledgment presence
    ack_per_segment = state["learner_acknowledgment_count"] / max(state["learner_segment_count"], 1)
    ack_score = min(1.0, ack_per_segment * 2.0)

    # Combined: 50% ratio, 30% questions, 20% acknowledgments
    combined = (ratio_score * 0.5) + (question_score * 0.3) + (ack_score * 0.2)
    return round(combined, 3)


async def update_engagement_score(barter_id: int, segment: SegmentRequest):
    if barter_id not in engagement_state:
        engagement_state[barter_id] = _new_engagement()

    state = engagement_state[barter_id]
    state["learner_speaking_seconds"] += segment.duration_seconds
    state["learner_segment_count"] += 1
    state["learner_question_count"] += count_questions(segment.text)
    state["learner_acknowledgment_count"] += count_acknowledgments(segment.text)

    score = calculate_engagement_score(state)
    state["last_score"] = score

    # Alert if low engagement, with cooldown
    if score < 0.3 and (time.time() - state["last_alert_time"] > ENGAGEMENT_ALERT_COOLDOWN):
        await post_engagement_alert(barter_id, score)
        state["last_alert_time"] = time.time()

    total = state["teacher_speaking_seconds"] + state["learner_speaking_seconds"]
    ratio_pct = (state["learner_speaking_seconds"] / max(total, 1)) * 100

    _engagement(barter_id, score, ratio_pct,
                state["learner_question_count"],
                state["learner_acknowledgment_count"])


async def post_engagement_alert(barter_id: int, score: float):
    payload = {
        "barter_id": barter_id,
        "alert_type": "low_engagement",
        "engagement_score": score,
    }
    try:
        resp = await http_client.post(
            f"{WARNING_ENGINE_URL}/engagement/alert", json=payload
        )
        resp.raise_for_status()
        _warn(f"Low engagement alert  barter={barter_id}  score={score:.2f}")
    except Exception as e:
        logger.error("Failed to POST engagement alert: %s", e)


# ---------------------------------------------------------------------------
# Topic analysis helpers
# ---------------------------------------------------------------------------


async def post_window_result(barter_id: int, window_id: int, classification: str,
                              similarity: float,
                              ts_start: float, ts_end: float, text_preview: str):
    payload = {
        "barter_id": barter_id,
        "window_id": window_id,
        "classification": classification,
        "similarity_score": round(similarity, 4),
        "timestamp_start": ts_start,
        "timestamp_end": ts_end,
        "text_preview": text_preview[:200],
    }
    try:
        resp = await http_client.post(f"{WARNING_ENGINE_URL}/window/result", json=payload)
        resp.raise_for_status()
        pass  # visual output already shown via _window()
    except Exception as e:
        logger.error("Failed to POST window result to warning engine: %s", e)


async def flush_buffer(barter_id: int):
    """Process whatever segments remain in the buffer as a final window."""
    buf = buffers.get(barter_id)
    if not buf or not buf["segments"]:
        return

    contract = contracts.get(barter_id)
    if not contract:
        return

    await process_window(barter_id, buf, contract)
    buf["segments"] = []
    buf["accumulated_seconds"] = 0.0


async def process_window(barter_id: int, buf: dict, contract: dict):
    buf["window_counter"] += 1
    window_id = buf["window_counter"]

    combined_text = " ".join(s["text"] for s in buf["segments"])
    cleaned = clean_text(combined_text)

    if not cleaned:
        logger.info("Window %d barter %d: empty after cleaning, skipping", window_id, barter_id)
        return

    ts_start = buf["segments"][0]["ts_start"]
    ts_end = buf["segments"][-1]["ts_end"]

    window_embedding = embed(cleaned)
    similarity = cosine_sim(window_embedding, contract["topic_embedding"])

    classification = classify(similarity)
    _window(barter_id, window_id, classification, similarity, combined_text)

    await post_window_result(
        barter_id=barter_id,
        window_id=window_id,
        classification=classification,
        similarity=similarity,
        ts_start=ts_start,
        ts_end=ts_end,
        text_preview=combined_text,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"service": "semantic-analysis", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/session/{barter_id}/contract")
async def register_contract(barter_id: int, req: ContractRequest):
    """Store contract and pre-compute topic embedding."""
    topic_text = f"{req.topic}. {req.scope}"
    topic_embedding = embed(topic_text)

    contracts[barter_id] = {
        "topic": req.topic,
        "scope": req.scope,
        "topic_embedding": topic_embedding,
        "teacher_user_id": req.teacher_user_id,
        "learner_user_id": req.learner_user_id,
    }
    buffers[barter_id] = _new_buffer()
    engagement_state[barter_id] = _new_engagement()

    _ok(f"Contract registered  barter={barter_id}  topic=\"{req.topic}\"  teacher={req.teacher_user_id}  learner={req.learner_user_id}")
    return {"status": "registered", "barter_id": barter_id}


async def _fetch_contract_from_backend(barter_id: int) -> dict | None:
    """Fetch contract from backend DB if not in memory."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{BACKEND_URL}/session/{barter_id}/contract")
            if resp.status_code == 200:
                data = resp.json()
                # Compute topic embedding
                topic_text = f"{data['topic']}. {data['scope'] or ''}"
                topic_embedding = embed(topic_text)
                return {
                    "topic": data["topic"],
                    "scope": data["scope"] or "",
                    "topic_embedding": topic_embedding,
                    "teacher_user_id": data["teacher_user_id"],
                    "learner_user_id": data["learner_user_id"],
                }
    except Exception as e:
        logger.warning(f"Failed to fetch contract from backend: {e}")
    return None


@app.post("/ingest/segment")
async def ingest_segment(req: SegmentRequest):
    """Receive a transcript segment. Route to topic analysis (teacher) or engagement (learner)."""
    barter_id = req.barter_id

    # Try to get contract from memory, or fetch from backend
    if barter_id not in contracts:
        contract = await _fetch_contract_from_backend(barter_id)
        if contract:
            contracts[barter_id] = contract
            buffers[barter_id] = _new_buffer()
            engagement_state[barter_id] = _new_engagement()
            _ok(f"Contract loaded from backend  barter={barter_id}  topic=\"{contract['topic']}\"")
        else:
            raise HTTPException(status_code=404, detail="Contract not registered for this barter session")

    contract = contracts[barter_id]

    if not req.text.strip():
        return {"status": "skipped", "reason": "empty transcript"}

    # Ensure engagement state exists (for teacher speaking time tracking)
    if barter_id not in engagement_state:
        engagement_state[barter_id] = _new_engagement()

    # Route based on role
    if req.user_id == contract["teacher_user_id"]:
        # Teacher: full topic relevance pipeline
        engagement_state[barter_id]["teacher_speaking_seconds"] += req.duration_seconds

        if barter_id not in buffers:
            buffers[barter_id] = _new_buffer()

        buf = buffers[barter_id]
        buf["segments"].append({
            "text": req.text,
            "duration": req.duration_seconds,
            "ts_start": req.timestamp_start,
            "ts_end": req.timestamp_end,
        })
        buf["accumulated_seconds"] += req.duration_seconds

        _info(f"Teacher buffer  barter={barter_id}  {buf['accumulated_seconds']:.1f}s / {WINDOW_DURATION_THRESHOLD:.0f}s")

        if buf["accumulated_seconds"] >= WINDOW_DURATION_THRESHOLD:
            await process_window(barter_id, buf, contract)
            buf["segments"] = []
            buf["accumulated_seconds"] = 0.0

        return {
            "status": "buffered",
            "role": "teacher",
            "accumulated_seconds": round(buf["accumulated_seconds"], 1),
            "window_count": buf["window_counter"],
        }

    elif req.user_id == contract["learner_user_id"]:
        # Learner: engagement scoring only
        await update_engagement_score(barter_id, req)

        return {
            "status": "engagement_scored",
            "role": "learner",
            "engagement_score": engagement_state[barter_id]["last_score"],
        }

    else:
        logger.warning("Barter %d: unknown user_id %d, skipping", barter_id, req.user_id)
        return {"status": "skipped", "reason": "unknown user_id"}


@app.post("/session/{barter_id}/end")
async def end_session(barter_id: int):
    """Flush remaining buffer, notify warning engine, then post engagement summary."""
    if barter_id not in buffers:
        logger.warning("Barter %d: no buffer found on end", barter_id)
    else:
        await flush_buffer(barter_id)
        del buffers[barter_id]

    # Save engagement state before cleanup (needed after warning engine call)
    eng_state = engagement_state.pop(barter_id, None)

    if barter_id in contracts:
        del contracts[barter_id]

    # Notify warning engine to finalise + send drift summary to backend
    # This creates the verdict row that the engagement summary will attach to
    try:
        resp = await http_client.post(f"{WARNING_ENGINE_URL}/session/{barter_id}/end")
        resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to notify warning engine of session end: %s", e)

    # Post engagement summary to backend (after verdict exists from drift summary)
    if eng_state:
        try:
            await http_client.post(
                f"{BACKEND_URL}/session/{barter_id}/engagement-summary",
                json={
                    "learner_engagement_score": eng_state["last_score"],
                    "learner_speaking_seconds": eng_state["learner_speaking_seconds"],
                    "teacher_speaking_seconds": eng_state["teacher_speaking_seconds"],
                    "learner_question_count": eng_state["learner_question_count"],
                    "learner_acknowledgment_count": eng_state["learner_acknowledgment_count"],
                    "learner_segment_count": eng_state["learner_segment_count"],
                },
            )
        except Exception as e:
            logger.error("Failed to POST engagement summary: %s", e)

    _ok(f"Session {barter_id} ended — buffer flushed, engagement summary posted")
    return {"status": "ended", "barter_id": barter_id}
