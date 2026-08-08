import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Terminal colours ────────────────────────────────────────────────────────
_R = "\033[0;31m";  _G = "\033[0;32m";  _Y = "\033[1;33m"
_C = "\033[0;36m";  _M = "\033[0;35m";  _B = "\033[1;34m"
_W = "\033[1;37m";  _NC = "\033[0m";    _BOLD = "\033[1m";  _DIM = "\033[2m"

def _banner(msg):  print(f"\n{_B}{_BOLD}{'─'*56}{_NC}\n  {_W}{_BOLD}{msg}{_NC}\n{_B}{_BOLD}{'─'*56}{_NC}", flush=True)
def _ok(msg):      print(f"  {_G}✓{_NC}  {msg}", flush=True)
def _info(msg):    print(f"  {_C}→{_NC}  {msg}", flush=True)
def _warn(msg):    print(f"  {_Y}⚠{_NC}  {_Y}{msg}{_NC}", flush=True)
def _silent(msg):  print(f"  {_DIM}·  {msg}{_NC}", flush=True)

_SEVERITY_FMT = {
    "mild":   f"{_Y}⚠  MILD WARNING   {_NC}",
    "strong": f"{_M}⚠⚠ STRONG WARNING {_NC}",
    "severe": f"{_R}{_BOLD}🚨 SEVERE WARNING {_NC}",
}

def _warning(barter_id, severity, reason, consecutive):
    label = _SEVERITY_FMT.get(severity, severity.upper())
    print(f"\n  {_BOLD}[Barter {barter_id}  consecutive={consecutive}]{_NC}", flush=True)
    print(f"  {label}  {reason}", flush=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("warning-engine")

import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class WindowResultRequest(BaseModel):
    barter_id: int
    window_id: int
    classification: str  # correct | weakly_correct | incorrect
    similarity_score: float
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    text_preview: str = ""


class SafetyAlertRequest(BaseModel):
    barter_id: int
    user_id: int
    warning_type: str  # "toxicity" or "nsfw"
    details: dict = {}


class EngagementAlertRequest(BaseModel):
    barter_id: int
    alert_type: str = "low_engagement"
    engagement_score: float


class SessionInitRequest(BaseModel):
    teacher_user_id: int = 1
    learner_user_id: int = 2


# ---------------------------------------------------------------------------
# In-Memory State
# ---------------------------------------------------------------------------

sessions: dict[int, dict] = {}


def _new_state() -> dict:
    return {
        "total_windows": 0,
        "incorrect_windows": 0,
        "consecutive_incorrect": 0,
        "max_consecutive_incorrect": 0,
        "total_drift_incidents": 0,
        "warning_history": [],
        "terminated": False,
    }


# ---------------------------------------------------------------------------
# HTTP Client (lifespan)
# ---------------------------------------------------------------------------

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    _banner("Warning Engine  ·  Port 8003")
    _info("Escalation ladder: 1→silent  2→mild  3–4→strong  5+→severe")
    http_client = httpx.AsyncClient(timeout=5.0)
    _ok("Service online — waiting for window results")
    yield
    await http_client.aclose()


async def post_to_backend(path: str, payload: dict) -> dict | None:
    try:
        resp = await http_client.post(f"{BACKEND_URL}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Backend call failed %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Warning Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Warning Decision Engine
# ---------------------------------------------------------------------------


async def run_warning_decision(
    barter_id: int, state: dict, request: WindowResultRequest
):
    # Guard: session already terminated
    if state["terminated"]:
        logger.info("Session %d terminated, ignoring window %d", barter_id, request.window_id)
        return {"action": "ignored", "reason": "session already terminated"}

    # Update counters
    state["total_windows"] += 1

    if request.classification == "incorrect":
        state["consecutive_incorrect"] += 1
        state["incorrect_windows"] += 1
        state["total_drift_incidents"] += 1
    else:
        state["consecutive_incorrect"] = 0

    state["max_consecutive_incorrect"] = max(
        state["max_consecutive_incorrect"], state["consecutive_incorrect"]
    )

    # Build the payload forwarded to backend for every window
    window_payload = {
        "barter_id": barter_id,
        "window_id": request.window_id,
        "classification": request.classification,
        "similarity_score": request.similarity_score,
        "text_preview": request.text_preview,
        "timestamp_start": request.timestamp_start,
        "timestamp_end": request.timestamp_end,
    }

    # Determine warning severity based on consecutive off-topic count
    severity = None
    reason = ""
    count = state["consecutive_incorrect"]

    if count >= 3:
        severity = "severe"
        reason = f"{count} consecutive off-topic windows — severe drift detected"
    elif count == 2:
        severity = "strong"
        reason = "2 consecutive off-topic windows"
    elif count == 1:
        await post_to_backend("/window/result", window_payload)
        _silent(f"Barter {barter_id}  window #{request.window_id}  incorrect (1 consecutive — silent)")
        return {"action": "silent", "consecutive_incorrect": 1}

    if severity is None:
        await post_to_backend("/window/result", window_payload)
        return {"action": "none", "consecutive_incorrect": 0}

    # Forward window result to backend for storage + broadcast
    await post_to_backend("/window/result", window_payload)

    # Record warning in local history
    warning_entry = {
        "severity": severity,
        "reason": reason,
        "window_ids": str(request.window_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    state["warning_history"].append(warning_entry)

    # POST warning to backend
    await post_to_backend("/warnings/log", {
        "barter_id": barter_id,
        "severity": severity,
        "reason": reason,
        "window_ids": str(request.window_id),
        "timestamp": warning_entry["timestamp"],
    })

    _warning(barter_id, severity, reason, state["consecutive_incorrect"])

    return {
        "action": "warning",
        "severity": severity,
        "reason": reason,
        "consecutive_incorrect": state["consecutive_incorrect"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"service": "warning-engine", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/session/{barter_id}/init")
async def init_session(barter_id: int, req: SessionInitRequest | None = None):
    if barter_id in sessions:
        logger.info("Session %d already initialized, skipping", barter_id)
        return {"status": "already_initialized", "barter_id": barter_id}

    state = _new_state()
    if req:
        state["teacher_user_id"] = req.teacher_user_id
        state["learner_user_id"] = req.learner_user_id
    sessions[barter_id] = state
    _ok(f"Session {barter_id} initialized")
    return {"status": "initialized", "barter_id": barter_id}


@app.post("/window/result")
async def receive_window_result(request: WindowResultRequest):
    barter_id = request.barter_id

    # Auto-init if session not found
    if barter_id not in sessions:
        logger.warning("Session %d not initialized, auto-initializing", barter_id)
        sessions[barter_id] = _new_state()

    state = sessions[barter_id]
    result = await run_warning_decision(barter_id, state, request)
    return result


@app.post("/safety/alert")
async def receive_safety_alert(request: SafetyAlertRequest):
    """Receive safety alerts (toxicity/NSFW) — immediate strong or severe warning."""
    barter_id = request.barter_id
    hard_block = request.details.get("hard_block", False)
    severity = "severe" if hard_block else "strong"
    categories = request.details.get("categories", {})
    cat_names = ", ".join(categories.keys()) if categories else request.warning_type

    reason = f"Safety violation ({request.warning_type}): {cat_names}"

    # Record in session state if it exists
    if barter_id in sessions:
        state = sessions[barter_id]
        warning_entry = {
            "severity": severity,
            "reason": reason,
            "window_ids": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state["warning_history"].append(warning_entry)

    # POST warning to backend for storage + WebSocket broadcast
    await post_to_backend("/warnings/log", {
        "barter_id": barter_id,
        "severity": severity,
        "reason": reason,
        "window_ids": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    _warning(barter_id, severity, reason, 0)
    return {"action": "safety_warning", "severity": severity, "reason": reason}


@app.post("/engagement/alert")
async def receive_engagement_alert(request: EngagementAlertRequest):
    """Low engagement from learner — softer alert, not a topic warning."""
    barter_id = request.barter_id
    reason = f"Low learner engagement (score: {request.engagement_score:.1%})"

    if barter_id in sessions:
        state = sessions[barter_id]
        warning_entry = {
            "severity": "mild",
            "reason": reason,
            "window_ids": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state["warning_history"].append(warning_entry)

    await post_to_backend("/warnings/log", {
        "barter_id": barter_id,
        "severity": "mild",
        "reason": reason,
        "window_ids": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    _warn(f"Barter {barter_id}  low learner engagement — {request.engagement_score:.1%}")
    return {"action": "engagement_alert"}


@app.post("/session/{barter_id}/end")
async def end_session(barter_id: int):
    if barter_id not in sessions:
        logger.warning("Session %d not found or already ended", barter_id)
        return {"status": "error", "detail": "Session not found or already ended"}

    state = sessions[barter_id]
    total = state["total_windows"]
    incorrect = state["incorrect_windows"]
    percent_incorrect = round((incorrect / total) * 100, 2) if total > 0 else 0.0

    drift_summary = {
        "barter_id": barter_id,
        "total_windows": total,
        "incorrect_windows": incorrect,
        "percent_incorrect": percent_incorrect,
        "max_consecutive_incorrect": state["max_consecutive_incorrect"],
        "total_drift_incidents": state["total_drift_incidents"],
        "warning_count": len(state["warning_history"]),
        "warnings": state["warning_history"],
        "terminated_early": state["terminated"],
    }

    await post_to_backend(f"/session/{barter_id}/drift-summary", drift_summary)

    print(f"\n  {_BOLD}[Barter {barter_id}  SESSION END]{_NC}", flush=True)
    _info(f"Windows: {total} total  {incorrect} incorrect  ({percent_incorrect:.1f}% off-topic)")
    _info(f"Warnings issued: {len(state['warning_history'])}  max_consecutive: {state['max_consecutive_incorrect']}")
    _ok(f"Drift summary posted to backend")

    del sessions[barter_id]
    return {"status": "ended", "drift_summary": drift_summary}
