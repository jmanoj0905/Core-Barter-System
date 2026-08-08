# Implementation Plan — Core Barter System

> Actionable, file-level implementation guide. Maps every design decision to specific code changes.
>
> **Scope**: This repo (core-barter-system) builds the Session Quality Monitor + Session Safety Monitor POC.
>
> **Baseline**: Stages 1–7 of `roadmap.md` are substantially complete. This plan covers all remaining work to reach demo-readiness per `DESIGN_DECISIONS.md`.
>
> Last updated: 2026-03-18 (Phases A, B & C completed)

---

## Current State vs Target State

### What exists and works (Stages 1–7)

| Component | Status | Key files |
|---|---|---|
| Backend Core (8000) | Working | `backend/app/routes.py`, `models.py`, `schemas.py` |
| Audio Pipeline (8001) | **Updated** — faster-whisper (int8), 15s buffer, toxicity check | `audio_pipeline/main.py` |
| Semantic Analysis (8002) | Working (Sentence-BERT, keyword scope) | `semantic_analysis/main.py` |
| Warning Engine (8003) | **Updated** — no auto-terminate, total_drift_incidents, /safety/alert | `warning_engine/main.py` |
| React Frontend | **Updated** — 3 screens + video frame capture for NSFW | `frontend/src/screens/*.jsx` |
| Backend Core (8000) safety | **NEW** — nudenet NSFW detection module | `backend/app/safety.py` |
| PostgreSQL | 6 tables + seed data | `backend/migrations/001-003` |
| Docker Compose | 5 containers defined | `docker-compose.yml` |

### What needs to change (per DESIGN_DECISIONS.md)

| Change | Priority | Effort |
|---|---|---|
| ~~Replace Whisper with faster-whisper~~ | ~~P0~~ | **DONE** |
| ~~Reduce audio buffer 25s → 15s~~ | ~~P0~~ | **DONE** |
| ~~Remove auto-terminate from warning engine~~ | ~~P0~~ | **DONE** |
| ~~Add total_drift_incidents tracking~~ | ~~P0~~ | **DONE** |
| ~~Add OpenAI Moderation API (toxicity)~~ | ~~P1~~ | **DONE** |
| ~~Add nudenet NSFW detection (video frames)~~ | ~~P1~~ | **DONE** |
| ~~Add teacher vs learner asymmetric monitoring~~ | ~~P1~~ | **DONE** |
| ~~Add learner engagement scoring~~ | ~~P1~~ | **DONE** |
| Add LLM scope generation (GPT-4o-mini) | P1 | 2–3 hrs |
| Add credit_balance to User + credit adjustment | P2 | 1–2 hrs |
| Add admin dashboard (React) | P2 | 4–6 hrs |
| Add admin API endpoints | P2 | 2–3 hrs |
| Upgrade scope violation to embedding similarity | P2 | 1–2 hrs |
| Add OpenAI Whisper API fallback | P3 | 1–2 hrs |
| Admin WebSocket (full monitoring data) | P2 | 2–3 hrs |
| Docker Compose integration | P3 | 1–2 hrs |
| End-to-end testing | P0 | 3–4 hrs |

**Completed: ~14 hours (Phases A + B + C)**
**Total estimated remaining work: 16–29 hours (Phases D–J)**

---

## Phase A — Critical Fixes (P0) ✅ COMPLETED

> All four critical fixes implemented and verified.

| Task | Files Modified | Summary |
|---|---|---|
| A.1 Remove auto-terminate | `warning_engine/main.py` | Removed `if severity == "severe": state["terminated"] = True` block and the terminate POST call. Updated severe reason from "auto-terminating session" → "severe drift detected". |
| A.2 Add total_drift_incidents | `warning_engine/main.py`, `backend/app/schemas.py` | Added `total_drift_incidents: 0` to `_new_state()`, increment on every off-topic window (never resets). Added to drift summary payload. Added field to `DriftSummaryRequest`. |
| A.3 Replace with faster-whisper | `audio_pipeline/main.py`, `audio_pipeline/requirements.txt` | Replaced `openai-whisper` with `faster-whisper`. Model: `WhisperModel("base", compute_type="int8")`. Updated `transcribe()` to use iterator API `(segments, info)`. |
| A.4 Reduce buffer 25s → 15s | `audio_pipeline/main.py` | Changed `BUFFER_THRESHOLD_SECONDS` from 25.0 to 15.0. |

---

## Phase B — Session Safety Monitor (P1) ✅ COMPLETED

> Toxicity detection (OpenAI Moderation API) + NSFW detection (nudenet) fully implemented.

| Task | Files Modified/Created | Summary |
|---|---|---|
| B.1 Toxicity detection | `audio_pipeline/main.py`, `audio_pipeline/requirements.txt`, `warning_engine/main.py` | Added `check_toxicity()` using `AsyncOpenAI().moderations.create()`. Thresholds: 0.7 flag, 0.9 hard block. Integrated into `process_buffer()` after transcription, before semantic analysis. Added `post_safety_warning()` to forward to warning engine. Added `POST /safety/alert` endpoint in warning engine with `SafetyAlertRequest` model. Fail-open: returns None on API errors. |
| B.2 NSFW detection | `backend/app/safety.py` (NEW), `backend/app/main.py`, `backend/app/routes.py`, `backend/app/schemas.py`, `backend/requirements.txt`, `frontend/src/screens/LiveSession.jsx` | Created nudenet wrapper module (`init_detector()`, `check_frame()`). NSFW classes: exposed body parts, confidence ≥0.6, hard block ≥0.9. Added `POST /safety/check-frame` endpoint. NudeNet initialized in backend lifespan with graceful fallback. Frontend captures video at 0.1 FPS (10s interval), sends base64 JPEG. Video tracks cleaned up in halt/confirm/terminate paths. Added `FrameCheckRequest` schema. |

**Safety alert pipeline**: Toxicity/NSFW → `POST /safety/alert` (warning engine) → `POST /warnings/log` (backend) → WebSocket broadcast to frontend.

**Dependencies added**: `openai`, `python-dotenv` (audio_pipeline); `nudenet` (backend). **Env var**: `OPENAI_API_KEY`.

---

## Phase C — Asymmetric Monitoring (P1) ✅ COMPLETED

> Teacher gets topic relevance. Learner gets engagement scoring. Per Decision §8.
>
> All tasks implemented and verified with 6 integration tests.

| Task | Files Modified/Created | Summary |
|---|---|---|
| C.1 Role tracking | `backend/app/models.py`, `backend/app/schemas.py`, `backend/app/routes.py`, `backend/migrations/004_teacher_learner.sql` | Added `teacher_user_id`/`learner_user_id` to SessionContract model + migration. Role defaults (teacher=1, learner=2) for backward compatibility. Roles propagated to semantic analysis contract and warning engine init. |
| C.2 Role-aware semantic analysis | `semantic_analysis/main.py` | Rewrote `ingest_segment()` to route by role: teacher segments → topic relevance pipeline (buffer/window/classify), learner segments → engagement scoring only. Unknown user_ids gracefully skipped. |
| C.3 Engagement scoring | `semantic_analysis/main.py`, `warning_engine/main.py`, `backend/app/routes.py` | Added `calculate_engagement_score()`: 50% speaking ratio + 30% question frequency + 20% acknowledgment detection. 30s minimum data threshold. Low engagement (<0.3) fires mild warning via `/engagement/alert` with 120s cooldown. Engagement summary posted to backend at session end and stored in verdict's `drift_summary` JSON. |

**Bug fix during testing**: Reordered `end_session()` in semantic analysis — engagement summary POST moved after warning engine notification so the verdict row exists when the engagement data is attached.

### C.1 Add role tracking to session contract

**File**: `backend/app/models.py`

The session contract needs to identify who is the teacher for this session. In the demo, user1 (Alice) is always the teacher.

**Changes**:
1. Add to `SessionContract` model:
   ```python
   teacher_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
   learner_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
   ```
2. Add migration `backend/migrations/004_teacher_learner.sql`:
   ```sql
   ALTER TABLE session_contracts ADD COLUMN teacher_user_id INTEGER REFERENCES users(id);
   ALTER TABLE session_contracts ADD COLUMN learner_user_id INTEGER REFERENCES users(id);
   UPDATE session_contracts SET teacher_user_id = 1, learner_user_id = 2;
   ALTER TABLE session_contracts ALTER COLUMN teacher_user_id SET NOT NULL;
   ALTER TABLE session_contracts ALTER COLUMN learner_user_id SET NOT NULL;
   ```
3. Update `SessionCreateRequest` schema to include `teacher_user_id` and `learner_user_id`.
4. Update `POST /session/create` to store these values.
5. When notifying semantic analysis (`POST /session/{id}/contract`), include `teacher_user_id` and `learner_user_id`.

### C.2 Modify semantic analysis for role-aware monitoring

**File**: `semantic_analysis/main.py`

Currently, all segments are buffered by `barter_id` regardless of who is speaking. We need to separate teacher and learner pipelines.

**Changes**:
1. Update contract storage to include roles:
   ```python
   contracts[barter_id] = {
       "topic": req.topic,
       "scope": req.scope,
       "allowed_concepts": req.allowed_concepts,
       "disallowed_concepts": req.disallowed_concepts,
       "topic_embedding": topic_embedding,
       "teacher_user_id": req.teacher_user_id,    # NEW
       "learner_user_id": req.learner_user_id,    # NEW
   }
   ```
2. Change buffer key from `barter_id` to `(barter_id, user_id)`:
   ```python
   buffers: dict[tuple[int, int], dict] = {}
   ```
3. In `ingest_segment()`, check the user role:
   ```python
   contract = contracts[barter_id]
   key = (barter_id, req.user_id)

   if req.user_id == contract["teacher_user_id"]:
       # Teacher: full topic relevance pipeline (existing logic)
       # Buffer, window, classify, post to warning engine
       ...
   elif req.user_id == contract["learner_user_id"]:
       # Learner: engagement scoring only (new logic)
       await update_engagement_score(barter_id, req)
   ```

### C.3 Add learner engagement scoring

**File**: `semantic_analysis/main.py` (add new section)

Per Decision §8, engagement = combined signal from:
- Speaking ratio (learner talks 20–40% = healthy, <10% = passive)
- Question frequency (detected from transcript: "?", "how", "what", "why" patterns)
- Audio activity (presence of acknowledgment sounds: "uh-huh", "okay", "right")

**Changes**:
1. Add engagement state per session:
   ```python
   engagement_state: dict[int, dict] = {}  # barter_id -> engagement tracking

   def _new_engagement() -> dict:
       return {
           "teacher_speaking_seconds": 0.0,
           "learner_speaking_seconds": 0.0,
           "learner_question_count": 0,
           "learner_acknowledgment_count": 0,
           "learner_segment_count": 0,
           "last_score": 0.0,
       }
   ```
2. Add question and acknowledgment detection:
   ```python
   QUESTION_PATTERNS = re.compile(r'\b(how|what|why|when|where|which|can you|could you|is it|are there)\b|\?')
   ACKNOWLEDGMENT_WORDS = {"uh-huh", "okay", "right", "yes", "yeah", "got it", "i see", "makes sense"}

   def count_questions(text: str) -> int:
       return len(QUESTION_PATTERNS.findall(text.lower()))

   def count_acknowledgments(text: str) -> int:
       text_lower = text.lower()
       return sum(1 for word in ACKNOWLEDGMENT_WORDS if word in text_lower)
   ```
3. Add engagement scoring function:
   ```python
   def calculate_engagement_score(state: dict) -> float:
       total_speaking = state["teacher_speaking_seconds"] + state["learner_speaking_seconds"]
       if total_speaking == 0:
           return 0.5  # no data yet

       # Speaking ratio: 20-40% is ideal (score 1.0), <10% is poor (0.2), >50% is noisy (0.6)
       ratio = state["learner_speaking_seconds"] / total_speaking
       if 0.20 <= ratio <= 0.40:
           ratio_score = 1.0
       elif ratio < 0.10:
           ratio_score = 0.2
       elif ratio < 0.20:
           ratio_score = 0.5 + (ratio - 0.10) * 5.0  # linear 0.5→1.0
       else:
           ratio_score = max(0.4, 1.0 - (ratio - 0.40) * 2.0)  # penalize over-talking

       # Question frequency: at least 1 question per 3 minutes of session = good
       minutes = total_speaking / 60.0
       question_rate = state["learner_question_count"] / max(minutes, 1.0)
       question_score = min(1.0, question_rate / 0.33)  # 1 per 3 min = 1.0

       # Acknowledgment: at least some engagement signals
       ack_per_segment = state["learner_acknowledgment_count"] / max(state["learner_segment_count"], 1)
       ack_score = min(1.0, ack_per_segment * 2.0)

       # Combined: 50% ratio, 30% questions, 20% acknowledgments
       combined = (ratio_score * 0.5) + (question_score * 0.3) + (ack_score * 0.2)
       return round(combined, 3)
   ```
4. Add engagement update function (called for learner segments):
   ```python
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

       # Post engagement score to warning engine if low
       if score < 0.3:
           await post_engagement_alert(barter_id, score)

       logger.info("Barter %d learner engagement: %.2f (ratio=%.0f%%, questions=%d, acks=%d)",
                    barter_id, score,
                    state["learner_speaking_seconds"] / max(
                        state["teacher_speaking_seconds"] + state["learner_speaking_seconds"], 1) * 100,
                    state["learner_question_count"],
                    state["learner_acknowledgment_count"])
   ```
5. Also track teacher speaking time (when teacher segments come in, add duration to engagement state):
   ```python
   # In the teacher branch of ingest_segment:
   if barter_id not in engagement_state:
       engagement_state[barter_id] = _new_engagement()
   engagement_state[barter_id]["teacher_speaking_seconds"] += req.duration_seconds
   ```
6. Add engagement alert post function:
   ```python
   async def post_engagement_alert(barter_id: int, score: float):
       payload = {
           "barter_id": barter_id,
           "alert_type": "low_engagement",
           "engagement_score": score,
       }
       try:
           resp = await http_client.post(f"{WARNING_ENGINE_URL}/engagement/alert", json=payload)
           resp.raise_for_status()
       except Exception as e:
           logger.error("Failed to POST engagement alert: %s", e)
   ```

**Warning Engine** — add engagement alert endpoint:
```python
@app.post("/engagement/alert")
async def receive_engagement_alert(request: EngagementAlertRequest):
    """Low engagement from learner — softer alert, not a topic warning."""
    await post_to_backend("/warnings/log", {
        "barter_id": request.barter_id,
        "severity": "mild",
        "reason": f"Low learner engagement (score: {request.engagement_score:.1%})",
        "window_ids": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"action": "engagement_alert"}
```

**Test**: Simulate a session where teacher talks for 5 minutes and learner says nothing → verify low engagement alert. Then simulate healthy engagement (learner asks questions) → verify no alert.

---

## Phase D — LLM Scope Generation (P1)

> Per Decision §10: GPT-4o-mini generates allowed/disallowed concepts from a topic.

### D.1 Add scope generation endpoint

**File**: `backend/app/routes.py` (add endpoint)
**New dependency**: `openai` in `backend/requirements.txt`
**New env var**: `OPENAI_API_KEY`

**Changes**:
1. Add to `backend/requirements.txt`:
   ```
   openai
   ```
2. Add endpoint:
   ```python
   from openai import AsyncOpenAI
   import json

   openai_client: AsyncOpenAI | None = None  # initialize in lifespan

   @router.post("/scope/generate")
   async def generate_scope(request: ScopeGenerateRequest):
       """Use GPT-4o-mini to generate allowed/disallowed concepts for a topic."""
       prompt = f"""Given the teaching topic: "{request.topic}"
   Level: {request.level or "beginner"}

   Generate a JSON object with:
   1. "allowed_concepts": list of 5-10 concepts that are within scope for teaching this topic
   2. "disallowed_concepts": list of 5-10 concepts that are outside scope (advanced topics, unrelated subjects, controversial topics)
   3. "scope_description": one sentence describing the acceptable scope

   Return ONLY valid JSON, no markdown."""

       response = await openai_client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": prompt}],
           response_format={"type": "json_object"},
           max_tokens=500,
           temperature=0.7,
       )

       scope_data = json.loads(response.choices[0].message.content)
       return scope_data
   ```
3. Add schema:
   ```python
   class ScopeGenerateRequest(BaseModel):
       topic: str
       level: str = "beginner"
   ```
4. Update frontend Setup screen to call this endpoint and pre-fill the concepts fields.

**Frontend integration** (`frontend/src/screens/Setup.jsx`):
```javascript
const generateScope = async () => {
    const resp = await fetch('http://localhost:8000/scope/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ topic, level: 'beginner' }),
    });
    const data = await resp.json();
    setAllowedConcepts(data.allowed_concepts.join(', '));
    setDisallowedConcepts(data.disallowed_concepts.join(', '));
    setScope(data.scope_description);
};
```

**Test**: POST `{"topic": "Python basics for beginners"}` → verify structured JSON with reasonable concepts.

---

## Phase E — Credits & Escrow (P2)

> Per Decision §12: Minimal credit demonstration — balance goes up/down based on verdict.

### E.1 Add credit_balance to User model

**File**: `backend/app/models.py`

**Changes**:
1. Add field to User:
   ```python
   credit_balance: Mapped[int] = mapped_column(Integer, default=50, nullable=False)
   ```
2. Add migration `backend/migrations/005_credits.sql`:
   ```sql
   ALTER TABLE users ADD COLUMN credit_balance INTEGER NOT NULL DEFAULT 50;
   ```

### E.2 Add credit adjustment after verdict

**File**: `backend/app/routes.py`

Add to `update_trust()` endpoint (or create a separate `/credits/{barter_id}/settle` endpoint):

```python
# Credit adjustment based on verdict
CREDIT_REWARDS = {"SUCCESSFUL": 15, "PARTIAL": 5, "DISPUTE": -10}
credit_delta = CREDIT_REWARDS.get(verdict.verdict_type, 0)

u1.credit_balance = max(0, u1.credit_balance + credit_delta)
u2.credit_balance = max(0, u2.credit_balance + credit_delta)
```

Return credit changes in the response:
```python
return {
    "user_1_trust": {...},
    "user_2_trust": {...},
    "user_1_credits": {"before": u1_credit_before, "after": u1.credit_balance, "delta": credit_delta},
    "user_2_credits": {"before": u2_credit_before, "after": u2.credit_balance, "delta": credit_delta},
}
```

**Frontend** (`frontend/src/screens/PostSession.jsx`):
Display credit changes: "Before: 50 credits → After: 65 credits (+15 for successful session)"

**Test**: Run a full session → SUCCESSFUL verdict → verify credits increase by 15. Run a DISPUTE → verify credits decrease by 10.

---

## Phase F — Admin Dashboard (P2)

> Per Decision §11: Real-time admin view with full controls.

### F.1 Admin API endpoints

**File**: `backend/app/routes.py` (add new section)

**New endpoints**:
```python
# --- Admin Endpoints ---

@router.post("/admin/session/{barter_id}/warn")
async def admin_manual_warning(barter_id: int, req: AdminWarningRequest, db=Depends(get_db)):
    """Admin sends a custom warning to both users."""
    warning = Warning(
        barter_session_id=barter_id,
        severity="strong",
        message=f"[ADMIN] {req.message}",
        window_ids="",
    )
    db.add(warning)
    await db.commit()
    await manager.broadcast(barter_id, {
        "warning_id": warning.id, "barter_id": barter_id,
        "severity": "strong", "reason": f"[ADMIN] {req.message}",
        "window_ids": "", "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"warning_id": warning.id}


@router.post("/admin/session/{barter_id}/pause")
async def admin_pause_session(barter_id: int, db=Depends(get_db)):
    """Pause monitoring and notify users."""
    # Broadcast pause notification
    await manager.broadcast(barter_id, {
        "type": "session_control", "action": "pause",
        "message": "Session paused by administrator",
    })
    return {"status": "paused"}


@router.post("/admin/session/{barter_id}/resume")
async def admin_resume_session(barter_id: int, db=Depends(get_db)):
    await manager.broadcast(barter_id, {
        "type": "session_control", "action": "resume",
        "message": "Session resumed by administrator",
    })
    return {"status": "resumed"}


@router.post("/admin/session/{barter_id}/adjust-thresholds")
async def admin_adjust_thresholds(barter_id: int, req: ThresholdAdjustRequest):
    """Adjust similarity thresholds mid-session."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        await client.post(f"http://localhost:8002/session/{barter_id}/thresholds", json={
            "upper": req.upper,
            "lower": req.lower,
        })
    return {"status": "updated", "upper": req.upper, "lower": req.lower}


@router.get("/admin/session/{barter_id}/live")
async def admin_session_live_data(barter_id: int, db=Depends(get_db)):
    """Get full live session data for admin dashboard."""
    # Fetch session, contract, all windows, all warnings
    session = (await db.execute(select(BarterSession).where(BarterSession.id == barter_id))).scalar_one_or_none()
    contract = (await db.execute(select(SessionContract).where(SessionContract.barter_session_id == barter_id))).scalar_one_or_none()
    windows = (await db.execute(select(WindowResult).where(WindowResult.barter_session_id == barter_id).order_by(WindowResult.id))).scalars().all()
    warnings = (await db.execute(select(Warning).where(Warning.barter_session_id == barter_id).order_by(Warning.id))).scalars().all()

    return {
        "session": {"id": session.id, "status": session.status, "started_at": session.started_at.isoformat() if session.started_at else None} if session else None,
        "contract": {"topic": contract.topic, "scope": contract.scope} if contract else None,
        "windows": [
            {"id": w.id, "number": w.window_number, "classification": w.classification,
             "similarity": w.cosine_similarity, "text": w.text_content}
            for w in windows
        ],
        "warnings": [
            {"id": w.id, "severity": w.severity, "message": w.message, "created_at": w.created_at.isoformat()}
            for w in warnings
        ],
    }
```

**Semantic Analysis** — add threshold adjustment endpoint:
```python
@app.post("/session/{barter_id}/thresholds")
async def adjust_thresholds(barter_id: int, req: ThresholdRequest):
    """Allow admin to adjust UPPER/LOWER thresholds mid-session."""
    global UPPER, LOWER
    UPPER = req.upper
    LOWER = req.lower
    logger.info("Thresholds adjusted: UPPER=%.2f, LOWER=%.2f", UPPER, LOWER)
    return {"status": "updated"}
```

### F.2 Admin WebSocket for live monitoring data

**File**: `backend/app/websocket.py`

Add a separate admin WebSocket that sends all data (transcripts, window results, engagement scores):
```python
@app.websocket("/ws/admin/{barter_id}")
async def admin_ws(barter_id: int, ws: WebSocket):
    await ws.accept()
    manager.connect_admin(barter_id, ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        manager.disconnect_admin(barter_id, ws)
```

Extend `ConnectionManager` to have separate admin connections that receive richer payloads (transcript text, similarity scores, engagement data).

### F.3 Store window results in backend DB

**File**: `backend/app/routes.py`

Currently window results only exist in the semantic analysis service's memory. For the admin dashboard, they need to be stored.

Add endpoint:
```python
@router.post("/windows/store")
async def store_window_result(req: WindowStoreRequest, db=Depends(get_db)):
    window = WindowResult(
        barter_session_id=req.barter_id,
        window_number=req.window_id,
        classification=req.classification,
        cosine_similarity=req.similarity_score,
        text_content=req.text_preview,
    )
    db.add(window)
    await db.commit()
    # Also broadcast to admin WebSocket
    await manager.broadcast_admin(req.barter_id, {
        "type": "window_result", **req.model_dump()
    })
    return {"window_id": window.id}
```

Update semantic analysis to also POST to this endpoint after processing each window.

### F.4 Admin Dashboard React page

**File**: `frontend/src/screens/AdminDashboard.jsx` (NEW)

Layout:
```
┌────────────────────────────────────┐
│          ADMIN DASHBOARD           │
├──────────────┬─────────────────────┤
│  TEACHER     │  LEARNER            │
│  Transcript  │  Transcript         │
│  [text...]   │  [text...]          │
│  Sim: 0.72   │  Engagement: 0.85   │
├──────────────┴─────────────────────┤
│  WARNING LOG                       │
│  [timestamp] [severity] [message]  │
├────────────────────────────────────┤
│  TIMELINE                          │
│  ■■■□□■■■□□■■■  (green/yellow/red) │
├────────────────────────────────────┤
│  CONTROLS                          │
│  [Manual Warn] [Pause] [Terminate] │
│  Thresholds: Upper [0.55] Lower [0.35] [Update] │
└────────────────────────────────────┘
```

**Key components**:
- WebSocket connection to `ws://localhost:8000/ws/admin/{barter_id}`
- Poll `GET /admin/session/{barter_id}/live` every 5 seconds for full state
- Buttons call admin API endpoints
- Timeline visualization using colored blocks per window classification

**Test**: Open admin dashboard alongside 2 user tabs → verify live data appears, admin controls work.

---

## Phase G — Scope Violation Upgrade (P2)

> Per Decision §6: Upgrade from keyword-only to embedding similarity.

### G.1 Embedding-based scope violation detection

**File**: `semantic_analysis/main.py`

**Changes**:
1. Pre-compute disallowed concept embeddings at contract registration:
   ```python
   contracts[barter_id] = {
       ...
       "disallowed_embeddings": [embed(concept) for concept in req.disallowed_concepts] if req.disallowed_concepts else [],
   }
   ```
2. Update `check_scope_violation()`:
   ```python
   def check_scope_violation(text: str, disallowed_concepts: list[str],
                              text_embedding=None, disallowed_embeddings=None) -> bool:
       # Method 1: Keyword matching (fast, catches exact matches)
       text_lower = text.lower()
       for concept in disallowed_concepts:
           if concept.lower() in text_lower:
               return True

       # Method 2: Embedding similarity (catches semantic violations)
       if text_embedding is not None and disallowed_embeddings:
           for emb in disallowed_embeddings:
               sim = cosine_sim(text_embedding, emb)
               if sim >= 0.60:  # threshold for semantic scope violation
                   return True

       return False
   ```
3. In `process_window()`, pass the embeddings:
   ```python
   scope_violation = check_scope_violation(
       combined_text,
       contract["disallowed_concepts"],
       text_embedding=window_embedding,
       disallowed_embeddings=contract.get("disallowed_embeddings", []),
   )
   ```

**Test**: Topic is "basic math". Disallowed: ["calculus", "advanced math"]. Input: "let's talk about derivatives and integrals" → should trigger scope violation even without exact keyword "calculus".

---

## Phase H — OpenAI Whisper API Fallback (P3)

> Per Decision §3: Cloud fallback for demo if faster-whisper latency is unsatisfactory.

### H.1 Add fallback STT

**File**: `audio_pipeline/main.py`

**Changes**:
1. Add env var `STT_MODE` (default: "local", options: "local", "cloud"):
   ```python
   STT_MODE = os.getenv("STT_MODE", "local")  # "local" or "cloud"
   ```
2. Add cloud transcription function:
   ```python
   async def transcribe_cloud(wav_path: str) -> str:
       """Use OpenAI Whisper API for cloud transcription."""
       try:
           with open(wav_path, "rb") as f:
               response = await openai_client.audio.transcriptions.create(
                   model="whisper-1",
                   file=f,
                   language="en",
               )
           return response.text.strip()
       finally:
           if os.path.exists(wav_path):
               os.unlink(wav_path)
   ```
3. In `process_buffer()`, choose based on mode:
   ```python
   wav_path = convert_to_wav(audio_bytes)
   if STT_MODE == "cloud":
       text = await transcribe_cloud(wav_path)
   else:
       text = transcribe(wav_path)
   ```

**Test**: Set `STT_MODE=cloud`, send audio → verify OpenAI API is called and returns transcript.

---

## Phase I — Database Migrations Summary

All new migrations needed (in order):

| Migration | File | Changes |
|---|---|---|
| 004 | `004_teacher_learner.sql` | Add `teacher_user_id`, `learner_user_id` to `session_contracts` |
| 005 | `005_credits.sql` | Add `credit_balance` to `users` (default 50) |

**Note**: Some changes (toxicity results, engagement scores) are stored in-memory only since they're transient per session. Only persist what's needed for verdicts and post-session review.

---

## Phase J — Dependency Updates Summary

### audio_pipeline/requirements.txt ✅ UPDATED
```
faster-whisper          # ✅ replaced openai-whisper (Phase A)
fastapi
uvicorn[standard]
httpx
openai                  # ✅ added for Moderation API (Phase B) + Whisper API fallback (Phase H)
python-dotenv           # ✅ added (Phase B)
```

### backend/requirements.txt ✅ PARTIALLY UPDATED
```
# Already added:
nudenet                 # ✅ added for NSFW detection (Phase B)
# Still needed (Phase D):
openai                  # for scope generation (GPT-4o-mini)
```
(Keep all existing dependencies)

### semantic_analysis/requirements.txt
No changes. Keep existing: `sentence-transformers`, `torch`, `httpx`, `fastapi`, `uvicorn`.

### warning_engine/requirements.txt
No changes. Keep existing: `fastapi`, `uvicorn`, `httpx`.

### Environment variables needed
```
OPENAI_API_KEY=sk-...    # Required for: Moderation API (free), scope generation, Whisper fallback
```

---

## Testing Plan

### Unit-level tests (per service)

| Test | Service | What to verify |
|---|---|---|
| Warning escalation (no auto-terminate) | 8003 | 5+ consecutive incorrect → severe warning, NOT termination |
| Total drift incidents tracking | 8003 | Counter increments on every off-topic, never resets |
| faster-whisper transcription | 8001 | Model loads, transcription returns text, latency < 5s |
| 15s buffer threshold | 8001 | Transcription triggers at 15s not 25s |
| Toxicity detection | 8001 | Hate speech flagged, clean text passes |
| NSFW detection | 8000 | NSFW image flagged, safe image passes |
| Teacher topic classification | 8002 | On-topic text → correct, off-topic → incorrect |
| Learner engagement scoring | 8002 | Passive learner → low score, active → high score |
| Embedding scope violation | 8002 | Semantically similar disallowed concept triggers violation |
| LLM scope generation | 8000 | Returns valid JSON with concepts |
| Credit adjustment | 8000 | SUCCESSFUL → +15, DISPUTE → -10 |

### Integration tests (cross-service)

| Test | Flow | What to verify |
|---|---|---|
| Full quality pipeline | Audio → STT → Semantic → Warning → Backend → WebSocket | Warning appears in browser within 8 seconds of off-topic speech |
| Safety pipeline | Toxic audio → STT → Moderation API → Warning → WebSocket | Immediate safety warning |
| NSFW pipeline | Video frame → Backend → nudenet → Warning → WebSocket | Safety warning on NSFW |
| Engagement pipeline | Silent learner → Engagement alert → Warning → WebSocket | Mild engagement alert |
| Verdict + credits | Session end → Verdict → Trust update → Credit settlement | Credits and trust both update correctly |
| Admin controls | Admin sends manual warning → Both users see it | WebSocket broadcast works |

### End-to-end demo test

1. Create session with LLM-generated scope
2. Both users join (2 browser tabs)
3. Admin opens dashboard (3rd tab)
4. Teacher speaks on-topic → green windows on admin
5. Teacher goes off-topic → red windows, warnings appear within 5–8 seconds
6. Teacher returns to topic → warnings de-escalate
7. Session ends → verdict → credits update → trust update
8. Admin sees full timeline

---

## Implementation Order (Recommended)

```
✅ Phase A (Critical Fixes) — DONE
  A.1 Remove auto-terminate           ✅
  A.2 Add total_drift_incidents        ✅
  A.3 Replace with faster-whisper      ✅
  A.4 Reduce buffer to 15s            ✅

✅ Phase B (Safety Monitor) — DONE
  B.1 Toxicity detection              ✅
  B.2 NSFW detection                  ✅

✅ Phase C (Asymmetric Monitoring) — DONE
  C.1 Role tracking                   ✅
  C.2 Role-aware semantic analysis    ✅
  C.3 Engagement scoring              ✅
  → All 6 integration tests passed

Week 1-2: Phase D (LLM Scope)
  D.1 Scope generation endpoint       [2 hrs]
  → Test with various topics

Week 2: Phase E + F (Credits + Admin Dashboard)
  E.1 Credit balance model            [1 hr]
  E.2 Credit adjustment logic         [1 hr]
  F.1 Admin API endpoints             [3 hrs]
  F.2 Admin WebSocket                 [2 hrs]
  F.3 Store window results            [1 hr]
  F.4 Admin React dashboard           [5 hrs]
  → Test admin dashboard end-to-end

Week 2-3: Phase G + H (Polish)
  G.1 Embedding scope violation       [2 hrs]
  H.1 Whisper API fallback            [2 hrs]
  → Full E2E testing + demo rehearsal

Week 3: Phase I + J (Integration)
  Run all migrations
  Update remaining dependencies
  Docker Compose verification
  Full demo rehearsal (3 browser tabs)
```

---

## File Change Summary

### Done (Phases A + B + C)

| File | Type | Status |
|---|---|---|
| `warning_engine/main.py` | MODIFY | ✅ Removed auto-terminate, added total_drift_incidents, `/safety/alert`, `/engagement/alert`, session init with roles |
| `audio_pipeline/main.py` | MODIFY | ✅ faster-whisper, 15s buffer, toxicity check via OpenAI Moderation API |
| `audio_pipeline/requirements.txt` | MODIFY | ✅ faster-whisper, openai, python-dotenv |
| `backend/app/safety.py` | NEW | ✅ nudenet NSFW detection module |
| `backend/app/main.py` | MODIFY | ✅ Initialize nudenet in lifespan |
| `backend/app/routes.py` | MODIFY | ✅ `/safety/check-frame`, role propagation to services, `/engagement-summary` endpoint |
| `backend/app/schemas.py` | MODIFY | ✅ FrameCheckRequest, DriftSummaryRequest, teacher/learner defaults in SessionCreateRequest |
| `backend/app/models.py` | MODIFY | ✅ teacher_user_id/learner_user_id on SessionContract |
| `backend/requirements.txt` | MODIFY | ✅ Added nudenet |
| `backend/migrations/004_teacher_learner.sql` | NEW | ✅ teacher/learner columns on session_contracts |
| `semantic_analysis/main.py` | MODIFY | ✅ Role-aware routing (teacher→topic, learner→engagement), engagement scoring, engagement alerts |
| `frontend/src/screens/LiveSession.jsx` | MODIFY | ✅ Video frame capture for NSFW (10s interval) |
| `.env.example` | MODIFY | ✅ Added OPENAI_API_KEY |

### Remaining (Phases D–J)

| File | Type | Changes |
|---|---|---|
| `audio_pipeline/main.py` | MODIFY | Whisper API fallback (Phase H) |
| `semantic_analysis/main.py` | MODIFY | Embedding scope violation, threshold adjustment (Phases G, F) |
| `backend/app/models.py` | MODIFY | Add credit_balance to User (Phase E) |
| `backend/app/schemas.py` | MODIFY | Add ScopeGenerate, AdminWarning, ThresholdAdjust, WindowStore schemas (Phases D, F) |
| `backend/app/routes.py` | MODIFY | Add admin endpoints, scope generation, credit settlement, window storage (Phases D, E, F) |
| `backend/app/main.py` | MODIFY | Initialize OpenAI client for scope generation (Phase D) |
| `backend/requirements.txt` | MODIFY | Add openai (Phase D) |
| `backend/migrations/005_credits.sql` | NEW | credit_balance column on users (Phase E) |
| `frontend/src/screens/Setup.jsx` | MODIFY | Add "Generate Scope" button calling LLM (Phase D) |
| `frontend/src/screens/PostSession.jsx` | MODIFY | Show credit changes (Phase E) |
| `frontend/src/screens/AdminDashboard.jsx` | NEW | Full admin monitoring dashboard (Phase F) |
| `frontend/src/App.jsx` | MODIFY | Add admin route (Phase F) |
