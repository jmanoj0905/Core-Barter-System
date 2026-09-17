# Video Engagement Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a video-based engagement signal (local MediaPipe vs cloud AWS Rekognition, comparable via a shared formula) that fuses with the existing speech-based engagement signal in `warning_engine`, without touching either scoring service's existing internals.

**Architecture:** New standalone `video_engagement` service (port 8004) mirrors `audio_pipeline`'s WebSocket-buffer-window shape. It computes a `video_attention_score` per 5s window per user from face-landmark sub-signals, under a runtime-switchable backend (`local` | `aws` | `both`). It POSTs raw results to `backend` (storage) and to `warning_engine` (fusion). `warning_engine` gains a small combine step that blends the video score with the speech `engagement_score` it already receives from `semantic_analysis` (which itself gets one new additive POST call, no internal changes). An offline `weight_search.py` script sweeps candidate weights for both the sub-signal formula and the fusion blend, using the existing speech score as ground-truth proxy, and records the chosen weights with rationale in `docs/video_engagement/design-choices.md`.

**Tech Stack:** Python 3.11, FastAPI, MediaPipe Face Mesh (local), boto3 Rekognition (cloud), httpx, pytest + pytest-asyncio, SQLAlchemy (SQLite), React (frontend capture only).

**Spec:** `docs/superpowers/specs/2026-09-17-video-engagement-design.md`

## Global Constraints

- Formula weights are never hardcoded as "the answer" — every weight (sub-signal triple, fusion blend) ships as an overridable env var with an explicit placeholder default, and the real value is picked by `weight_search.py`, documented in `docs/video_engagement/design-choices.md`.
- `semantic_analysis`'s `calculate_engagement_score()` and its internal state machine are not modified — only one new outbound POST call is added.
- `warning_engine`'s existing topic-adherence escalation logic (1/2/3+ off-topic → silent/strong/severe) is not touched. The engagement fusion is a fully separate, additive code path.
- Video score only fuses when its `user_id` matches the session's `learner_user_id` — mirrors the existing speech-engagement scope (learner only).
- No face detected in a frame → skip that frame, don't zero the score. Zero faces in a whole window → omit the window (no POST), don't post a zero/garbage score.
- Fail-open on cloud errors (Rekognition throttled/failed) — log and skip that backend's score for the window; in `both` mode the other backend's score still posts.
- New service, new tables: single-purpose, no cross-service DB access — `video_engagement` never touches SQLite directly, only via `backend`'s HTTP API, matching every other service in this repo.

---

## File Structure

```
apps/video_engagement/
  main.py              # FastAPI + WS ingest, buffering, backend switch, config endpoints
  scoring.py           # pure functions: sub-signal extraction + weighted formula (no I/O)
  weight_search.py     # offline script: grid-sweep weights against ground truth, write design doc
  requirements.txt
  Dockerfile
  tests/
    test_scoring.py
    test_main.py

apps/backend/app/
  models.py            # + VideoEngagementResult, EngagementScoreLog
  schemas.py            # + VideoEngagementRequest, EngagementLogRequest
  routes.py            # + /session/{id}/video-engagement (POST/GET), /session/{id}/engagement-log (POST), /session/{id}/engagement (GET), /session/{id}/engagement/history (GET)
apps/backend/migrations/
  007_video_engagement.sql
apps/backend/tests/
  conftest.py           # new — self-contained, doesn't depend on tests/conftest.py's stale paths
  test_video_engagement_routes.py

apps/warning_engine/main.py   # + fusion state, /engagement/update, /video-engagement/update
apps/warning_engine/tests/
  conftest.py
  test_engagement_fusion.py

apps/semantic_analysis/main.py   # + one additive POST call in update_engagement_score
apps/semantic_analysis/tests/
  conftest.py
  test_engagement_update_post.py

docs/video_engagement/design-choices.md   # written by weight_search.py, seeded with placeholder

docker-compose.yml     # + video_engagement service
apps/frontend/nginx.conf   # + /video/ WS proxy route
apps/frontend/src/screens/LiveSession.jsx   # + video-frame capture + WS send, mirrors audio capture
```

**Note on existing test infra:** `tests/conftest.py` at the repo root references `ROOT / "backend"`, `ROOT / "warning_engine"` etc. — paths that predate the `apps/` reorg and no longer exist (verified: `ls backend warning_engine` from repo root both fail). Those root-level e2e tests are currently broken independent of this work; fixing them is out of scope here. Every test added by this plan is self-contained inside its own service's `tests/` directory with its own `conftest.py`, using corrected `apps/...` paths, so none of it depends on the broken shared fixtures.

---

### Task 1: Backend — DB tables for video results and fused engagement log

**Files:**
- Modify: `apps/backend/app/models.py`
- Create: `apps/backend/migrations/007_video_engagement.sql`
- Create: `apps/backend/tests/conftest.py`
- Create: `apps/backend/tests/test_video_engagement_routes.py` (model-level part only in this task; route tests come in Task 2 — this task's test just verifies the tables exist and accept inserts)

**Interfaces:**
- Produces: `VideoEngagementResult` model (`barter_session_id, user_id, window_start, window_end, video_attention_score, backend_used, raw_signals_json, created_at`), `EngagementScoreLog` model (`barter_session_id, user_id, speech_engagement_score, video_attention_score, fused_engagement_score, created_at`) — Task 2's routes import both by these exact names from `app.models`.

- [ ] **Step 1: Write the failing test**

`apps/backend/tests/conftest.py`:

```python
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from app.models import Base  # noqa: E402

TEST_DB_URL = "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true"
test_engine = create_async_engine(TEST_DB_URL, echo=False)
test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)


async def override_get_db():
    async with test_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def backend_client():
    with patch("app.safety.init_detector", lambda: None):
        from app.main import app
        from app.database import get_db

        app.dependency_overrides[get_db] = override_get_db

        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with test_session_factory() as db:
            from app.models import User
            db.add(User(id=1, username="alice", trust_score=1.0))
            db.add(User(id=2, username="bob", trust_score=1.0))
            await db.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def db_session():
    async with test_session_factory() as session:
        yield session
```

`apps/backend/tests/test_video_engagement_routes.py`:

```python
import pytest
from app.models import VideoEngagementResult, EngagementScoreLog


@pytest.mark.asyncio
async def test_video_engagement_result_model_roundtrip(backend_client, db_session):
    row = VideoEngagementResult(
        barter_session_id=1,
        user_id=2,
        window_start=0.0,
        window_end=5.0,
        video_attention_score=0.72,
        backend_used="local",
        raw_signals='{"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.8}',
    )
    db_session.add(row)
    await db_session.commit()
    assert row.id is not None


@pytest.mark.asyncio
async def test_engagement_score_log_model_roundtrip(backend_client, db_session):
    row = EngagementScoreLog(
        barter_session_id=1,
        user_id=2,
        speech_engagement_score=0.6,
        video_attention_score=0.72,
        fused_engagement_score=0.65,
    )
    db_session.add(row)
    await db_session.commit()
    assert row.id is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/backend && python -m pytest tests/test_video_engagement_routes.py -v`
Expected: FAIL — `ImportError: cannot import name 'VideoEngagementResult' from 'app.models'`

- [ ] **Step 3: Add the models**

Add to `apps/backend/app/models.py` (after `WindowResult`):

```python
class VideoEngagementResult(Base):
    __tablename__ = "video_engagement_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    window_start: Mapped[float] = mapped_column(Float, nullable=False)
    window_end: Mapped[float] = mapped_column(Float, nullable=False)
    video_attention_score: Mapped[float] = mapped_column(Float, nullable=False)
    backend_used: Mapped[str] = mapped_column(String(20), nullable=False)
    raw_signals: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EngagementScoreLog(Base):
    __tablename__ = "engagement_score_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    speech_engagement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_attention_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    fused_engagement_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

Add `apps/backend/migrations/007_video_engagement.sql`:

```sql
-- Migration 007: Video Engagement
-- Adds video_engagement_results (raw per-window video-only scores) and
-- engagement_score_log (fused speech+video engagement history)

CREATE TABLE IF NOT EXISTS video_engagement_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    window_start FLOAT NOT NULL,
    window_end FLOAT NOT NULL,
    video_attention_score FLOAT NOT NULL,
    backend_used TEXT NOT NULL,
    raw_signals TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_video_engagement_barter ON video_engagement_results(barter_session_id);

CREATE TABLE IF NOT EXISTS engagement_score_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    user_id INTEGER REFERENCES users(id),
    speech_engagement_score FLOAT,
    video_attention_score FLOAT,
    fused_engagement_score FLOAT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_engagement_log_barter ON engagement_score_log(barter_session_id);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/backend && python -m pytest tests/test_video_engagement_routes.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/backend/app/models.py apps/backend/migrations/007_video_engagement.sql apps/backend/tests/conftest.py apps/backend/tests/test_video_engagement_routes.py
git commit -m "feat(backend): add video_engagement_results and engagement_score_log tables"
```

---

### Task 2: Backend — API endpoints for storing/exposing video + fused engagement results

**Files:**
- Modify: `apps/backend/app/schemas.py`
- Modify: `apps/backend/app/routes.py`
- Modify: `apps/backend/tests/test_video_engagement_routes.py` (add route tests)

**Interfaces:**
- Consumes: `VideoEngagementResult`, `EngagementScoreLog` from Task 1 (`app.models`).
- Produces: `POST /session/{barter_id}/video-engagement`, `GET /session/{barter_id}/video-engagement`, `POST /session/{barter_id}/engagement-log`, `GET /session/{barter_id}/engagement`, `GET /session/{barter_id}/engagement/history` — `video_engagement` service (Task 9) and `warning_engine` (Task 4) call these by these exact paths and payload shapes.

- [ ] **Step 1: Write the failing tests**

Append to `apps/backend/tests/test_video_engagement_routes.py`:

```python
@pytest.mark.asyncio
async def test_post_and_list_video_engagement(backend_client):
    resp = await backend_client.post("/session/1/video-engagement", json={
        "user_id": 2,
        "window_start": 0.0,
        "window_end": 5.0,
        "video_attention_score": 0.72,
        "backend_used": "local",
        "raw_signals": {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.8},
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "stored"

    resp = await backend_client.get("/session/1/video-engagement")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["video_attention_score"] == 0.72
    assert rows[0]["raw_signals"] == {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.8}


@pytest.mark.asyncio
async def test_post_engagement_log_and_get_latest(backend_client):
    await backend_client.post("/session/1/engagement-log", json={
        "user_id": 2,
        "speech_engagement_score": 0.6,
        "video_attention_score": None,
        "fused_engagement_score": 0.6,
    })
    resp = await backend_client.post("/session/1/engagement-log", json={
        "user_id": 2,
        "speech_engagement_score": 0.6,
        "video_attention_score": 0.72,
        "fused_engagement_score": 0.65,
    })
    assert resp.status_code == 200

    resp = await backend_client.get("/session/1/engagement")
    assert resp.status_code == 200
    latest = resp.json()
    assert latest["fused_engagement_score"] == 0.65

    resp = await backend_client.get("/session/1/engagement/history")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_get_engagement_no_data_returns_404(backend_client):
    resp = await backend_client.get("/session/999/engagement")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/backend && python -m pytest tests/test_video_engagement_routes.py -v`
Expected: FAIL — `404 Not Found` for `/session/1/video-engagement` (route doesn't exist yet)

- [ ] **Step 3: Add schemas and routes**

Add to `apps/backend/app/schemas.py`:

```python
class VideoEngagementRequest(BaseModel):
    user_id: int
    window_start: float
    window_end: float
    video_attention_score: float
    backend_used: str
    raw_signals: dict


class EngagementLogRequest(BaseModel):
    user_id: int | None = None
    speech_engagement_score: float | None = None
    video_attention_score: float | None = None
    fused_engagement_score: float
```

Add to `apps/backend/app/routes.py` (near the `WindowResult` section; import `json`, `VideoEngagementResult`, `EngagementScoreLog`, `VideoEngagementRequest`, `EngagementLogRequest` at top alongside existing imports):

```python
@router.post("/session/{barter_id}/video-engagement")
async def store_video_engagement(
    barter_id: int, req: VideoEngagementRequest, db: AsyncSession = Depends(get_db)
):
    row = VideoEngagementResult(
        barter_session_id=barter_id,
        user_id=req.user_id,
        window_start=req.window_start,
        window_end=req.window_end,
        video_attention_score=req.video_attention_score,
        backend_used=req.backend_used,
        raw_signals=json.dumps(req.raw_signals),
    )
    db.add(row)
    await db.commit()
    return {"status": "stored", "id": row.id}


@router.get("/session/{barter_id}/video-engagement")
async def list_video_engagement(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(VideoEngagementResult)
        .where(VideoEngagementResult.barter_session_id == barter_id)
        .order_by(VideoEngagementResult.window_start)
    )
    rows = result.scalars().all()
    return [
        {
            "user_id": r.user_id,
            "window_start": r.window_start,
            "window_end": r.window_end,
            "video_attention_score": r.video_attention_score,
            "backend_used": r.backend_used,
            "raw_signals": json.loads(r.raw_signals),
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@router.post("/session/{barter_id}/engagement-log")
async def store_engagement_log(
    barter_id: int, req: EngagementLogRequest, db: AsyncSession = Depends(get_db)
):
    row = EngagementScoreLog(
        barter_session_id=barter_id,
        user_id=req.user_id,
        speech_engagement_score=req.speech_engagement_score,
        video_attention_score=req.video_attention_score,
        fused_engagement_score=req.fused_engagement_score,
    )
    db.add(row)
    await db.commit()
    return {"status": "stored", "id": row.id}


@router.get("/session/{barter_id}/engagement")
async def get_latest_engagement(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(EngagementScoreLog)
        .where(EngagementScoreLog.barter_session_id == barter_id)
        .order_by(EngagementScoreLog.created_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="No engagement data for this session")
    return {
        "user_id": row.user_id,
        "speech_engagement_score": row.speech_engagement_score,
        "video_attention_score": row.video_attention_score,
        "fused_engagement_score": row.fused_engagement_score,
        "created_at": row.created_at.isoformat(),
    }


@router.get("/session/{barter_id}/engagement/history")
async def get_engagement_history(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(EngagementScoreLog)
        .where(EngagementScoreLog.barter_session_id == barter_id)
        .order_by(EngagementScoreLog.created_at)
    )
    rows = result.scalars().all()
    return [
        {
            "user_id": r.user_id,
            "speech_engagement_score": r.speech_engagement_score,
            "video_attention_score": r.video_attention_score,
            "fused_engagement_score": r.fused_engagement_score,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/backend && python -m pytest tests/test_video_engagement_routes.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/backend/app/schemas.py apps/backend/app/routes.py apps/backend/tests/test_video_engagement_routes.py
git commit -m "feat(backend): add video-engagement and engagement-log endpoints"
```

---

### Task 3: warning_engine — receive live speech engagement score (`/engagement/update`)

**Files:**
- Modify: `apps/warning_engine/main.py`
- Create: `apps/warning_engine/tests/conftest.py`
- Create: `apps/warning_engine/tests/test_engagement_fusion.py`

**Interfaces:**
- Produces: `sessions[barter_id]["speech_engagement_score"]`, `sessions[barter_id]["learner_user_id"]` (now always set, `None` if uninitialized), `POST /engagement/update` — Task 5 (`semantic_analysis`) calls this endpoint with `{barter_id, user_id, engagement_score}`.

- [ ] **Step 1: Write the failing test**

`apps/warning_engine/tests/conftest.py`:

```python
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest_asyncio.fixture
async def warning_client():
    spec = importlib.util.spec_from_file_location("warning_engine_main", APP_DIR / "main.py")
    we_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(we_main)

    mock_http = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = lambda: None
    mock_http.post.return_value = mock_response

    we_main.http_client = mock_http
    we_main.sessions.clear()

    transport = ASGITransport(app=we_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_http, we_main

    we_main.sessions.clear()
```

`apps/warning_engine/tests/test_engagement_fusion.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_engagement_update_stores_speech_score(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    resp = await client.post("/engagement/update", json={
        "barter_id": 1, "user_id": 2, "engagement_score": 0.6,
    })
    assert resp.status_code == 200
    assert we_main.sessions[1]["speech_engagement_score"] == 0.6


@pytest.mark.asyncio
async def test_engagement_update_auto_inits_session(warning_client):
    client, mock_http, we_main = warning_client
    resp = await client.post("/engagement/update", json={
        "barter_id": 42, "user_id": 2, "engagement_score": 0.4,
    })
    assert resp.status_code == 200
    assert 42 in we_main.sessions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/warning_engine && python -m pytest tests/test_engagement_fusion.py -v`
Expected: FAIL — `404 Not Found` for `/engagement/update`

- [ ] **Step 3: Implement**

In `apps/warning_engine/main.py`, modify `_new_state()`:

```python
def _new_state() -> dict:
    return {
        "total_windows": 0,
        "incorrect_windows": 0,
        "consecutive_incorrect": 0,
        "max_consecutive_incorrect": 0,
        "total_drift_incidents": 0,
        "warning_history": [],
        "terminated": False,
        "learner_user_id": None,
        "speech_engagement_score": None,
        "video_attention_score": None,
    }
```

Modify `init_session` to always set `learner_user_id` (not only `if req:`):

```python
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
```

Add new pydantic model near `EngagementAlertRequest`:

```python
class EngagementUpdateRequest(BaseModel):
    barter_id: int
    user_id: int
    engagement_score: float
```

Add new endpoint near `/engagement/alert`:

```python
@app.post("/engagement/update")
async def receive_engagement_update(request: EngagementUpdateRequest):
    """Live speech-based engagement score from semantic_analysis (every update, not just low alerts)."""
    barter_id = request.barter_id
    if barter_id not in sessions:
        sessions[barter_id] = _new_state()
    state = sessions[barter_id]
    state["speech_engagement_score"] = request.engagement_score
    return {"status": "updated"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/warning_engine && python -m pytest tests/test_engagement_fusion.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/warning_engine/main.py apps/warning_engine/tests/conftest.py apps/warning_engine/tests/test_engagement_fusion.py
git commit -m "feat(warning_engine): receive live speech engagement score"
```

---

### Task 4: warning_engine — video score fusion + persistence to backend

**Files:**
- Modify: `apps/warning_engine/main.py`
- Modify: `apps/warning_engine/tests/test_engagement_fusion.py`

**Interfaces:**
- Consumes: `sessions[barter_id]` from Task 3 (`speech_engagement_score`, `video_attention_score`, `learner_user_id` keys), `post_to_backend(path, payload)` (existing helper).
- Produces: `POST /video-engagement/update` — `video_engagement` service (Task 9) calls this with `{barter_id, user_id, video_attention_score}`. Fused result is POSTed to backend's `/session/{barter_id}/engagement-log` (Task 2).

- [ ] **Step 1: Write the failing test**

Append to `apps/warning_engine/tests/test_engagement_fusion.py`:

```python
@pytest.mark.asyncio
async def test_video_update_ignored_for_non_learner(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    resp = await client.post("/video-engagement/update", json={
        "barter_id": 1, "user_id": 1, "video_attention_score": 0.9,  # user 1 is the teacher
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    assert we_main.sessions[1]["video_attention_score"] is None


@pytest.mark.asyncio
async def test_fusion_combines_speech_and_video_for_learner(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})
    await client.post("/engagement/update", json={
        "barter_id": 1, "user_id": 2, "engagement_score": 0.6,
    })
    resp = await client.post("/video-engagement/update", json={
        "barter_id": 1, "user_id": 2, "video_attention_score": 0.8,
    })
    assert resp.status_code == 200

    # w_speech=0.7, w_video=0.3 defaults -> 0.7*0.6 + 0.3*0.8 = 0.66
    logged_call = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/engagement-log")]
    assert len(logged_call) == 1
    payload = logged_call[0].kwargs["json"]
    assert payload["fused_engagement_score"] == pytest.approx(0.66, abs=1e-6)
    assert payload["speech_engagement_score"] == 0.6
    assert payload["video_attention_score"] == 0.8


@pytest.mark.asyncio
async def test_fusion_falls_back_to_speech_only_when_no_video(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})
    await client.post("/engagement/update", json={
        "barter_id": 1, "user_id": 2, "engagement_score": 0.55,
    })

    logged_call = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/engagement-log")]
    assert len(logged_call) == 1
    assert logged_call[0].kwargs["json"]["fused_engagement_score"] == 0.55
    assert logged_call[0].kwargs["json"]["video_attention_score"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/warning_engine && python -m pytest tests/test_engagement_fusion.py -v`
Expected: FAIL — `test_video_update_ignored_for_non_learner` gets 404 (`/video-engagement/update` doesn't exist), and the speech-only test finds zero `/engagement-log` calls (fusion/logging not wired yet).

- [ ] **Step 3: Implement**

Add near the top of `apps/warning_engine/main.py`, with the other env vars:

```python
# Fusion weights — experimental placeholders, picked properly by
# apps/video_engagement/weight_search.py and recorded in
# docs/video_engagement/design-choices.md. Speech starts as the
# majority signal since it is already tuned; video is a minority
# signal until validated against it.
ENGAGEMENT_FUSION_W_SPEECH = float(os.getenv("ENGAGEMENT_FUSION_W_SPEECH", "0.7"))
ENGAGEMENT_FUSION_W_VIDEO = float(os.getenv("ENGAGEMENT_FUSION_W_VIDEO", "0.3"))
```

Add the video update model and a fusion helper, plus the endpoint:

```python
class VideoEngagementUpdateRequest(BaseModel):
    barter_id: int
    user_id: int
    video_attention_score: float


async def _recompute_and_log_fusion(barter_id: int, state: dict):
    speech = state.get("speech_engagement_score")
    video = state.get("video_attention_score")
    if speech is None and video is None:
        return
    if speech is not None and video is not None:
        fused = ENGAGEMENT_FUSION_W_SPEECH * speech + ENGAGEMENT_FUSION_W_VIDEO * video
    else:
        fused = speech if speech is not None else video
    fused = round(fused, 4)

    await post_to_backend(f"/session/{barter_id}/engagement-log", {
        "user_id": state.get("learner_user_id"),
        "speech_engagement_score": speech,
        "video_attention_score": video,
        "fused_engagement_score": fused,
    })


@app.post("/engagement/update")
async def receive_engagement_update(request: EngagementUpdateRequest):
    barter_id = request.barter_id
    if barter_id not in sessions:
        sessions[barter_id] = _new_state()
    state = sessions[barter_id]
    state["speech_engagement_score"] = request.engagement_score
    await _recompute_and_log_fusion(barter_id, state)
    return {"status": "updated"}


@app.post("/video-engagement/update")
async def receive_video_engagement_update(request: VideoEngagementUpdateRequest):
    barter_id = request.barter_id
    if barter_id not in sessions:
        sessions[barter_id] = _new_state()
    state = sessions[barter_id]

    learner_id = state.get("learner_user_id")
    if learner_id is not None and request.user_id != learner_id:
        return {"status": "ignored", "reason": "video score is not for the learner"}

    state["video_attention_score"] = request.video_attention_score
    await _recompute_and_log_fusion(barter_id, state)
    return {"status": "updated"}
```

(This replaces the plain `/engagement/update` endpoint added in Task 3 — same route, now also calling `_recompute_and_log_fusion`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/warning_engine && python -m pytest tests/test_engagement_fusion.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/warning_engine/main.py apps/warning_engine/tests/test_engagement_fusion.py
git commit -m "feat(warning_engine): fuse video + speech engagement, log to backend"
```

---

### Task 5: semantic_analysis — post live engagement score (additive only)

**Files:**
- Modify: `apps/semantic_analysis/main.py`
- Create: `apps/semantic_analysis/tests/conftest.py`
- Create: `apps/semantic_analysis/tests/test_engagement_update_post.py`

**Interfaces:**
- Consumes: `contracts[barter_id]["learner_user_id"]` (existing), `WARNING_ENGINE_URL` (existing env var).
- Produces: outbound `POST {WARNING_ENGINE_URL}/engagement/update` with `{barter_id, user_id, engagement_score}` on every `update_engagement_score` call — matches Task 3's endpoint contract exactly.

- [ ] **Step 1: Write the failing test**

`apps/semantic_analysis/tests/conftest.py`:

```python
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest_asyncio.fixture
async def semantic_client():
    spec = importlib.util.spec_from_file_location("semantic_analysis_main", APP_DIR / "main.py")
    sa_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sa_main)

    mock_http = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = lambda: None
    mock_http.post.return_value = mock_response

    sa_main.http_client = mock_http
    sa_main.contracts.clear()
    sa_main.engagement_state.clear()
    sa_main.contracts[1] = {
        "topic": "test", "scope": "test",
        "topic_embedding": sa_main.embed("test topic"),
        "teacher_user_id": 1, "learner_user_id": 2,
    }

    transport = ASGITransport(app=sa_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_http, sa_main
```

`apps/semantic_analysis/tests/test_engagement_update_post.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_learner_segment_posts_engagement_update(semantic_client):
    client, mock_http, sa_main = semantic_client

    resp = await client.post("/ingest/segment", json={
        "barter_id": 1,
        "user_id": 2,  # learner
        "text": "That makes sense, can you explain more?",
        "duration_seconds": 35.0,
        "timestamp_start": 0.0,
        "timestamp_end": 35.0,
    })
    assert resp.status_code == 200

    update_calls = [
        c for c in mock_http.post.call_args_list
        if c.args[0].endswith("/engagement/update")
    ]
    assert len(update_calls) == 1
    payload = update_calls[0].kwargs["json"]
    assert payload["barter_id"] == 1
    assert payload["user_id"] == 2
    assert isinstance(payload["engagement_score"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/semantic_analysis && python -m pytest tests/test_engagement_update_post.py -v`
Expected: FAIL — zero calls to `/engagement/update` (not implemented yet)

- [ ] **Step 3: Implement**

Add a small helper in `apps/semantic_analysis/main.py` near `post_engagement_alert`:

```python
async def post_engagement_update(barter_id: int, user_id: int, score: float):
    """Live score, posted on every update — not just low-engagement alerts."""
    try:
        await http_client.post(f"{WARNING_ENGINE_URL}/engagement/update", json={
            "barter_id": barter_id, "user_id": user_id, "engagement_score": score,
        })
    except Exception as e:
        logger.error("Failed to POST engagement update: %s", e)
```

In `update_engagement_score`, after `state["last_score"] = score` add:

```python
    learner_id = contracts.get(barter_id, {}).get("learner_user_id")
    if learner_id is not None:
        await post_engagement_update(barter_id, learner_id, score)
```

No other line in `calculate_engagement_score` or `update_engagement_score` changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/semantic_analysis && python -m pytest tests/test_engagement_update_post.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/semantic_analysis/main.py apps/semantic_analysis/tests/conftest.py apps/semantic_analysis/tests/test_engagement_update_post.py
git commit -m "feat(semantic_analysis): post live engagement score to warning_engine"
```

---

### Task 6: video_engagement — pure scoring formula module

**Files:**
- Create: `apps/video_engagement/scoring.py`
- Create: `apps/video_engagement/tests/test_scoring.py`

**Interfaces:**
- Produces: `sub_signals_from_mediapipe_landmarks(landmarks: dict[int, tuple[float, float]]) -> dict`, `sub_signals_from_rekognition_face_detail(face_detail: dict) -> dict`, `video_attention_score(sub_signals: dict, weights: tuple[float, float, float]) -> float`, `DEFAULT_WEIGHTS: tuple[float, float, float]` — Tasks 7, 8, 9, and 11 (`weight_search.py`) import all of these by these exact names from `scoring.py`.

- [ ] **Step 1: Write the failing test**

`apps/video_engagement/tests/test_scoring.py`:

```python
import pytest
from scoring import (
    DEFAULT_WEIGHTS,
    sub_signals_from_mediapipe_landmarks,
    sub_signals_from_rekognition_face_detail,
    video_attention_score,
)


def test_video_attention_score_weighted_combination():
    sub_signals = {"eyes_open": 1.0, "head_deviation": 0.0, "gaze_centered": 1.0}
    score = video_attention_score(sub_signals, weights=(0.4, 0.4, 0.2))
    assert score == pytest.approx(1.0)

    sub_signals_bad = {"eyes_open": 0.0, "head_deviation": 1.0, "gaze_centered": 0.0}
    score_bad = video_attention_score(sub_signals_bad, weights=(0.4, 0.4, 0.2))
    assert score_bad == pytest.approx(0.0)


def test_video_attention_score_clamped_to_0_1():
    score = video_attention_score(
        {"eyes_open": 2.0, "head_deviation": -1.0, "gaze_centered": 2.0},
        weights=(0.4, 0.4, 0.2),
    )
    assert 0.0 <= score <= 1.0


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS) == pytest.approx(1.0)


def test_mediapipe_eyes_open_high_when_eye_landmarks_wide():
    # Synthetic landmarks: right eye corners (33, 133) far apart in y from
    # top/bottom lid points (160,158 / 153,144) -> wide-open eye.
    landmarks = {
        # right eye (open)
        33: (0.30, 0.40), 160: (0.32, 0.36), 158: (0.34, 0.36),
        133: (0.36, 0.40), 153: (0.34, 0.44), 144: (0.32, 0.44),
        # left eye (open)
        362: (0.60, 0.40), 385: (0.62, 0.36), 387: (0.64, 0.36),
        263: (0.66, 0.40), 373: (0.64, 0.44), 380: (0.62, 0.44),
        # nose + cheeks for head pose, iris for gaze
        1: (0.48, 0.42), 234: (0.28, 0.42), 454: (0.68, 0.42),
        468: (0.48, 0.42), 473: (0.48, 0.42),
    }
    signals = sub_signals_from_mediapipe_landmarks(landmarks)
    assert signals["eyes_open"] > 0.5
    assert 0.0 <= signals["head_deviation"] <= 1.0
    assert 0.0 <= signals["gaze_centered"] <= 1.0


def test_mediapipe_missing_landmarks_raises():
    with pytest.raises(KeyError):
        sub_signals_from_mediapipe_landmarks({1: (0.5, 0.5)})


def test_rekognition_eyes_open_maps_boolean_to_float():
    face_detail = {
        "EyesOpen": {"Value": True, "Confidence": 99.0},
        "Pose": {"Yaw": 5.0, "Pitch": -3.0, "Roll": 1.0},
    }
    signals = sub_signals_from_rekognition_face_detail(face_detail)
    assert signals["eyes_open"] == 1.0
    assert 0.0 <= signals["head_deviation"] <= 1.0
    assert 0.0 <= signals["gaze_centered"] <= 1.0


def test_rekognition_eyes_closed_and_large_yaw():
    face_detail = {
        "EyesOpen": {"Value": False, "Confidence": 90.0},
        "Pose": {"Yaw": 80.0, "Pitch": 10.0, "Roll": 0.0},
    }
    signals = sub_signals_from_rekognition_face_detail(face_detail)
    assert signals["eyes_open"] == 0.0
    assert signals["head_deviation"] > 0.5
    assert signals["gaze_centered"] < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/video_engagement && python -m pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scoring'`

- [ ] **Step 3: Implement**

`apps/video_engagement/scoring.py`:

```python
"""Pure video-engagement scoring functions — no I/O, no ML model loading.

Formula and weights are experimental (see docs/video_engagement/design-choices.md).
DEFAULT_WEIGHTS is a placeholder until weight_search.py picks a real value.
"""

import math

# w_eyes, w_head, w_gaze — placeholder, pending weight_search.py
DEFAULT_WEIGHTS: tuple[float, float, float] = (0.4, 0.4, 0.2)

# MediaPipe Face Mesh landmark indices used (468/478-point mesh, refine_landmarks=True)
_RIGHT_EYE = {"p1": 33, "p2": 160, "p3": 158, "p4": 133, "p5": 153, "p6": 144}
_LEFT_EYE = {"p1": 362, "p2": 385, "p3": 387, "p4": 263, "p5": 373, "p6": 380}
_NOSE_TIP = 1
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473

_EAR_CLOSED = 0.15
_EAR_OPEN = 0.30


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _eye_aspect_ratio(landmarks: dict[int, tuple[float, float]], eye: dict[str, int]) -> float:
    p1, p2, p3, p4, p5, p6 = (landmarks[eye[k]] for k in ("p1", "p2", "p3", "p4", "p5", "p6"))
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def sub_signals_from_mediapipe_landmarks(
    landmarks: dict[int, tuple[float, float]],
) -> dict[str, float]:
    """Map raw Face Mesh landmarks to the three formula sub-signals, each 0-1."""
    right_ear = _eye_aspect_ratio(landmarks, _RIGHT_EYE)
    left_ear = _eye_aspect_ratio(landmarks, _LEFT_EYE)
    avg_ear = (right_ear + left_ear) / 2.0
    eyes_open = _clamp01((avg_ear - _EAR_CLOSED) / (_EAR_OPEN - _EAR_CLOSED))

    nose = landmarks[_NOSE_TIP]
    left_cheek = landmarks[_LEFT_CHEEK]
    right_cheek = landmarks[_RIGHT_CHEEK]
    span = right_cheek[0] - left_cheek[0]
    if span == 0:
        head_deviation = 1.0
    else:
        ratio = (nose[0] - left_cheek[0]) / span
        head_deviation = _clamp01(abs(ratio - 0.5) * 2.0)

    def _gaze_ratio(iris_key: int, eye: dict[str, int]) -> float:
        iris = landmarks[iris_key]
        p1, p4 = landmarks[eye["p1"]], landmarks[eye["p4"]]
        eye_span = p4[0] - p1[0]
        if eye_span == 0:
            return 0.5
        return (iris[0] - p1[0]) / eye_span

    left_gaze = _gaze_ratio(_LEFT_IRIS_CENTER, _LEFT_EYE)
    right_gaze = _gaze_ratio(_RIGHT_IRIS_CENTER, _RIGHT_EYE)
    avg_gaze_ratio = (left_gaze + right_gaze) / 2.0
    gaze_centered = _clamp01(1.0 - abs(avg_gaze_ratio - 0.5) * 2.0)

    return {
        "eyes_open": eyes_open,
        "head_deviation": head_deviation,
        "gaze_centered": gaze_centered,
    }


def sub_signals_from_rekognition_face_detail(face_detail: dict) -> dict[str, float]:
    """Map an AWS Rekognition DetectFaces FaceDetails[i] entry to the sub-signals.

    Rekognition has no iris/gaze data, so gaze_centered is approximated from
    yaw alone — a known limitation, documented in design-choices.md.
    """
    eyes_open = 1.0 if face_detail["EyesOpen"]["Value"] else 0.0

    pose = face_detail["Pose"]
    yaw, pitch = abs(pose["Yaw"]), abs(pose["Pitch"])
    head_deviation = _clamp01((yaw / 90.0 + pitch / 90.0) / 2.0)
    gaze_centered = _clamp01(1.0 - yaw / 90.0)

    return {
        "eyes_open": eyes_open,
        "head_deviation": head_deviation,
        "gaze_centered": gaze_centered,
    }


def video_attention_score(
    sub_signals: dict[str, float], weights: tuple[float, float, float] = DEFAULT_WEIGHTS
) -> float:
    w_eyes, w_head, w_gaze = weights
    raw = (
        w_eyes * sub_signals["eyes_open"]
        + w_head * (1.0 - sub_signals["head_deviation"])
        + w_gaze * sub_signals["gaze_centered"]
    )
    return _clamp01(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/video_engagement && python -m pytest tests/test_scoring.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/video_engagement/scoring.py apps/video_engagement/tests/test_scoring.py
git commit -m "feat(video_engagement): pure scoring formula for local + cloud sub-signals"
```

---

### Task 7: video_engagement — service scaffold, requirements, Dockerfile

**Files:**
- Create: `apps/video_engagement/requirements.txt`
- Create: `apps/video_engagement/Dockerfile`
- Create: `apps/video_engagement/tests/test_main.py` (health check only in this task)
- Create: `apps/video_engagement/main.py` (skeleton: app, health, root — full ingest logic added in Task 9)

**Interfaces:**
- Produces: `app` (FastAPI instance) in `apps/video_engagement/main.py` — Task 9 extends this same file/app object.

- [ ] **Step 1: Write the failing test**

`apps/video_engagement/tests/test_main.py`:

```python
from fastapi.testclient import TestClient
from main import app


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_root():
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.json()["service"] == "video-engagement"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/video_engagement && python -m pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Implement**

`apps/video_engagement/requirements.txt`:

```
fastapi
uvicorn[standard]
httpx
python-dotenv
boto3
mediapipe
opencv-python-headless
numpy
```

`apps/video_engagement/Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
```

`apps/video_engagement/main.py` (skeleton — replaced/extended in Task 9):

```python
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("video-engagement")

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=15.0)
    yield
    await http_client.aclose()


app = FastAPI(title="Video Engagement", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"service": "video-engagement", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/video_engagement && python -m pytest tests/test_main.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/video_engagement/requirements.txt apps/video_engagement/Dockerfile apps/video_engagement/main.py apps/video_engagement/tests/test_main.py
git commit -m "feat(video_engagement): service scaffold with health check"
```

---

### Task 8: video_engagement — local (MediaPipe) and cloud (Rekognition) frame processors

**Files:**
- Modify: `apps/video_engagement/main.py`
- Modify: `apps/video_engagement/tests/test_main.py`

**Interfaces:**
- Consumes: `sub_signals_from_mediapipe_landmarks`, `sub_signals_from_rekognition_face_detail`, `video_attention_score`, `DEFAULT_WEIGHTS` from `scoring.py` (Task 6).
- Produces: `process_frame_local(frame_bytes: bytes) -> dict | None`, `process_frame_aws(frame_bytes: bytes) -> dict | None` (both return `{"eyes_open":..., "head_deviation":..., "gaze_centered":...}` or `None` if no face detected) — Task 9's window-flush logic calls these by these exact names.

- [ ] **Step 1: Write the failing test**

Append to `apps/video_engagement/tests/test_main.py`:

```python
from unittest.mock import MagicMock, patch

import numpy as np
import cv2

import main


def _make_jpeg_bytes(width=100, height=100) -> bytes:
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_process_frame_local_no_face_returns_none():
    # A blank grey image has no face -> MediaPipe finds nothing.
    frame_bytes = _make_jpeg_bytes()
    result = main.process_frame_local(frame_bytes)
    assert result is None


def test_process_frame_aws_no_face_returns_none():
    frame_bytes = _make_jpeg_bytes()
    fake_client = MagicMock()
    fake_client.detect_faces.return_value = {"FaceDetails": []}
    with patch.object(main, "_rekognition_client", return_value=fake_client):
        result = main.process_frame_aws(frame_bytes)
    assert result is None


def test_process_frame_aws_maps_face_detail():
    frame_bytes = _make_jpeg_bytes()
    fake_client = MagicMock()
    fake_client.detect_faces.return_value = {
        "FaceDetails": [{
            "EyesOpen": {"Value": True, "Confidence": 99.0},
            "Pose": {"Yaw": 2.0, "Pitch": 1.0, "Roll": 0.0},
        }]
    }
    with patch.object(main, "_rekognition_client", return_value=fake_client):
        result = main.process_frame_aws(frame_bytes)
    assert result is not None
    assert result["eyes_open"] == 1.0


def test_process_frame_aws_fails_open_on_exception():
    frame_bytes = _make_jpeg_bytes()
    fake_client = MagicMock()
    fake_client.detect_faces.side_effect = Exception("throttled")
    with patch.object(main, "_rekognition_client", return_value=fake_client):
        result = main.process_frame_aws(frame_bytes)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/video_engagement && python -m pytest tests/test_main.py -v`
Expected: FAIL — `AttributeError: module 'main' has no attribute 'process_frame_local'`

- [ ] **Step 3: Implement**

Add to `apps/video_engagement/main.py` (after the CORS middleware, before the endpoints):

```python
import os

import cv2
import mediapipe as mp
import numpy as np

from scoring import (
    DEFAULT_WEIGHTS,
    sub_signals_from_mediapipe_landmarks,
    sub_signals_from_rekognition_face_detail,
    video_attention_score,
)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

WEIGHT_EYES = float(os.getenv("VIDEO_WEIGHT_EYES", str(DEFAULT_WEIGHTS[0])))
WEIGHT_HEAD = float(os.getenv("VIDEO_WEIGHT_HEAD", str(DEFAULT_WEIGHTS[1])))
WEIGHT_GAZE = float(os.getenv("VIDEO_WEIGHT_GAZE", str(DEFAULT_WEIGHTS[2])))
ACTIVE_WEIGHTS = (WEIGHT_EYES, WEIGHT_HEAD, WEIGHT_GAZE)

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def _landmarks_from_mediapipe_result(result, width: int, height: int) -> dict[int, tuple[float, float]]:
    face = result.multi_face_landmarks[0]
    return {i: (lm.x * width, lm.y * height) for i, lm in enumerate(face.landmark)}


def process_frame_local(frame_bytes: bytes) -> dict | None:
    """Decode a JPEG frame, run MediaPipe Face Mesh, return sub-signals or None."""
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    height, width = img.shape[:2]
    landmarks = _landmarks_from_mediapipe_result(result, width, height)
    return sub_signals_from_mediapipe_landmarks(landmarks)


def _rekognition_client():
    import boto3
    return boto3.client(
        "rekognition",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def process_frame_aws(frame_bytes: bytes) -> dict | None:
    """Send one JPEG frame to Rekognition DetectFaces, return sub-signals or None.

    Fails open: any error (including throttling) logs and returns None so
    the window simply skips the cloud score rather than raising.
    """
    try:
        client = _rekognition_client()
        resp = client.detect_faces(Image={"Bytes": frame_bytes}, Attributes=["ALL"])
        face_details = resp.get("FaceDetails", [])
        if not face_details:
            return None
        return sub_signals_from_rekognition_face_detail(face_details[0])
    except Exception as e:
        logger.error("Rekognition detect_faces failed: %s", e)
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/video_engagement && python -m pytest tests/test_main.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/video_engagement/main.py apps/video_engagement/tests/test_main.py
git commit -m "feat(video_engagement): local MediaPipe and cloud Rekognition frame processors"
```

---

### Task 9: video_engagement — WebSocket ingest, window buffering, backend switch, result posting

**Files:**
- Modify: `apps/video_engagement/main.py`
- Modify: `apps/video_engagement/tests/test_main.py`

**Interfaces:**
- Consumes: `process_frame_local`, `process_frame_aws`, `video_attention_score`, `ACTIVE_WEIGHTS` (Task 8).
- Produces: `ws /video/{barter_id}/{user_id}`, `GET/POST /video/config`, `POST /session/{barter_id}/end` — frontend (Task 12) connects to the WS route; `backend`'s `/session/{barter_id}/video-engagement` and `warning_engine`'s `/video-engagement/update` (Tasks 2, 4) are called by this task's `process_buffer`.

- [ ] **Step 1: Write the failing test**

Append to `apps/video_engagement/tests/test_main.py`:

```python
import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest


def test_video_config_get_default():
    with TestClient(app) as client:
        resp = client.get("/video/config")
        assert resp.status_code == 200
        assert resp.json()["backend"] == "local"
        assert set(resp.json()["available"]) == {"local", "aws", "both"}


def test_video_config_post_invalid_backend_rejected():
    with TestClient(app) as client:
        resp = client.post("/video/config", json={"backend": "gcp"})
        assert resp.status_code == 400


def test_video_config_post_switches_backend():
    with TestClient(app) as client:
        resp = client.post("/video/config", json={"backend": "aws"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "aws"
        main.current_video_backend = "local"  # reset for other tests


@pytest.mark.asyncio
async def test_process_buffer_local_posts_result_when_face_found():
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}
    main.http_client = AsyncMock()

    with patch.object(main, "process_frame_local", return_value={
        "eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.85,
    }):
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="local")

    posted_paths = [c.args[0] for c in main.http_client.post.call_args_list]
    assert any("/video-engagement" in p for p in posted_paths)
    assert any("/video-engagement/update" in p for p in posted_paths)


@pytest.mark.asyncio
async def test_process_buffer_skips_window_when_no_face_detected():
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}
    main.http_client = AsyncMock()

    with patch.object(main, "process_frame_local", return_value=None):
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="local")

    assert main.http_client.post.call_count == 0


@pytest.mark.asyncio
async def test_process_buffer_both_mode_posts_twice_with_different_backend_tag():
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}
    main.http_client = AsyncMock()

    with patch.object(main, "process_frame_local", return_value={
        "eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.85,
    }), patch.object(main, "process_frame_aws", return_value={
        "eyes_open": 0.8, "head_deviation": 0.2, "gaze_centered": 0.75,
    }):
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="both")

    backend_store_calls = [
        c.kwargs["json"]["backend_used"]
        for c in main.http_client.post.call_args_list
        if c.args[0].endswith("/video-engagement")
    ]
    assert sorted(backend_store_calls) == ["aws", "local"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/video_engagement && python -m pytest tests/test_main.py -v`
Expected: FAIL — `404` for `/video/config`, `AttributeError` for `process_buffer` (not implemented yet)

- [ ] **Step 3: Implement**

Add to `apps/video_engagement/main.py`:

```python
import time

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
WARNING_ENGINE_URL = os.getenv("WARNING_ENGINE_URL", "http://localhost:8003")
BUFFER_THRESHOLD_SECONDS = 5.0

_VALID_VIDEO_BACKENDS = {"local", "aws", "both"}
current_video_backend: str = os.getenv("VIDEO_BACKEND", "local")

buffers: dict[tuple[int, int], dict] = {}


def _new_buffer() -> dict:
    return {"frames": [], "wall_start": time.time()}


class VideoConfigRequest(BaseModel):
    backend: str


@app.get("/video/config")
async def get_video_config():
    return {"backend": current_video_backend, "available": sorted(_VALID_VIDEO_BACKENDS)}


@app.post("/video/config")
async def set_video_config(body: VideoConfigRequest):
    global current_video_backend
    if body.backend not in _VALID_VIDEO_BACKENDS:
        raise HTTPException(400, f"Unknown backend '{body.backend}'. Choose from: {sorted(_VALID_VIDEO_BACKENDS)}")
    current_video_backend = body.backend
    return {"backend": current_video_backend}


async def _score_and_post(barter_id: int, user_id: int, window_start: float, window_end: float,
                           sub_signals: dict, backend_used: str):
    score = video_attention_score(sub_signals, ACTIVE_WEIGHTS)

    payload = {
        "user_id": user_id,
        "window_start": window_start,
        "window_end": window_end,
        "video_attention_score": score,
        "backend_used": backend_used,
        "raw_signals": sub_signals,
    }
    try:
        await http_client.post(f"{BACKEND_URL}/session/{barter_id}/video-engagement", json=payload)
    except Exception as e:
        logger.error("Failed to POST video-engagement to backend: %s", e)

    try:
        await http_client.post(f"{WARNING_ENGINE_URL}/video-engagement/update", json={
            "barter_id": barter_id, "user_id": user_id, "video_attention_score": score,
        })
    except Exception as e:
        logger.error("Failed to POST video-engagement update to warning_engine: %s", e)


async def process_buffer(barter_id: int, user_id: int, buf: dict, backend: str):
    """Run the configured backend(s) on the buffered frames and post any resulting score(s)."""
    frames = buf["frames"]
    if not frames:
        return

    window_start = buf["wall_start"]
    window_end = time.time()

    if backend in ("local", "both"):
        detected = [s for s in (process_frame_local(f) for f in frames) if s is not None]
        if detected:
            avg = {
                key: sum(s[key] for s in detected) / len(detected)
                for key in ("eyes_open", "head_deviation", "gaze_centered")
            }
            await _score_and_post(barter_id, user_id, window_start, window_end, avg, "local")

    if backend in ("aws", "both"):
        mid_frame = frames[len(frames) // 2]
        sub_signals = process_frame_aws(mid_frame)
        if sub_signals is not None:
            await _score_and_post(barter_id, user_id, window_start, window_end, sub_signals, "aws")


def reset_buffer(buf: dict):
    buf["frames"] = []
    buf["wall_start"] = time.time()


@app.websocket("/video/{barter_id}/{user_id}")
async def video_ws(barter_id: int, user_id: int, ws: WebSocket):
    await ws.accept()
    key = (barter_id, user_id)
    buffers[key] = _new_buffer()
    buf = buffers[key]

    try:
        while True:
            frame = await ws.receive_bytes()
            if not frame:
                continue
            buf["frames"].append(frame)

            if time.time() - buf["wall_start"] >= BUFFER_THRESHOLD_SECONDS:
                await process_buffer(barter_id, user_id, buf, current_video_backend)
                reset_buffer(buf)

    except (WebSocketDisconnect, RuntimeError):
        if buf["frames"]:
            await process_buffer(barter_id, user_id, buf, current_video_backend)
        if key in buffers:
            del buffers[key]


@app.post("/session/{barter_id}/end")
async def end_session(barter_id: int):
    keys_to_delete = [k for k in buffers if k[0] == barter_id]
    for key in keys_to_delete:
        user_id = key[1]
        buf = buffers[key]
        if buf["frames"]:
            await process_buffer(barter_id, user_id, buf, current_video_backend)
        del buffers[key]
    return {"status": "ended", "barter_id": barter_id}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/video_engagement && python -m pytest tests/test_main.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/video_engagement/main.py apps/video_engagement/tests/test_main.py
git commit -m "feat(video_engagement): websocket ingest, window buffering, backend switch"
```

---

### Task 10: Wire into docker-compose and nginx

**Files:**
- Modify: `docker-compose.yml`
- Modify: `apps/frontend/nginx.conf`

**Interfaces:**
- Consumes: `apps/video_engagement/Dockerfile` (Task 7), `ws /video/{barter_id}/{user_id}` and `GET/POST /video/config` (Task 9).
- Produces: routable `video_engagement:8004` service reachable from `backend`, `warning_engine`, and the frontend nginx proxy.

- [ ] **Step 1: Add the service to `docker-compose.yml`**

Add after the `audio_pipeline` service block:

```yaml
  video_engagement:
    build:
      context: ./apps/video_engagement
    volumes:
      - model_cache:/root/.cache
    environment:
      BACKEND_URL: http://backend:8000
      WARNING_ENGINE_URL: http://warning_engine:8003
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:-}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:-}
      AWS_REGION: ${AWS_REGION:-ap-south-1}
      VIDEO_BACKEND: ${VIDEO_BACKEND:-local}
      VIDEO_WEIGHT_EYES: ${VIDEO_WEIGHT_EYES:-0.4}
      VIDEO_WEIGHT_HEAD: ${VIDEO_WEIGHT_HEAD:-0.4}
      VIDEO_WEIGHT_GAZE: ${VIDEO_WEIGHT_GAZE:-0.2}
    restart: on-failure
```

Also add `ENGAGEMENT_FUSION_W_SPEECH` / `ENGAGEMENT_FUSION_W_VIDEO` to the `warning_engine` service's `environment` block:

```yaml
      ENGAGEMENT_FUSION_W_SPEECH: ${ENGAGEMENT_FUSION_W_SPEECH:-0.7}
      ENGAGEMENT_FUSION_W_VIDEO: ${ENGAGEMENT_FUSION_W_VIDEO:-0.3}
```

- [ ] **Step 2: Add the nginx proxy route**

Add to `apps/frontend/nginx.conf`, after the `/audio/` location block:

```nginx
    # WebSocket: video stream → video_engagement
    location /video/ {
        proxy_pass http://video_engagement:8004;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600;
    }
```

- [ ] **Step 3: Verify the compose file parses**

Run: `docker compose config --quiet`
Expected: no output, exit code 0 (valid YAML, all interpolations resolve)

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml apps/frontend/nginx.conf
git commit -m "feat: wire video_engagement service into compose and nginx"
```

---

### Task 11: weight_search.py — offline experiment script + design-choices doc

**Files:**
- Create: `apps/video_engagement/weight_search.py`
- Create: `apps/video_engagement/tests/test_weight_search.py`
- Create: `docs/video_engagement/design-choices.md` (seeded placeholder; overwritten by the script when it has real data)

**Interfaces:**
- Consumes: `video_attention_score` (Task 6), `backend`'s `GET /session/{barter_id}/video-engagement` and `GET /session/{barter_id}/engagement/history` (Task 2).
- Produces: `sweep_subsignal_weights(raw_signal_rows, ground_truth_by_window) -> list[dict]`, `sweep_fusion_weights(video_scores, speech_scores) -> list[dict]`, `pearson_correlation(xs, ys) -> float` — a CLI entry point `if __name__ == "__main__"` that fetches data for a given `--barter-id` and writes the results doc.

- [ ] **Step 1: Write the failing test**

`apps/video_engagement/tests/test_weight_search.py`:

```python
import pytest
from weight_search import pearson_correlation, sweep_subsignal_weights, sweep_fusion_weights


def test_pearson_correlation_perfect_positive():
    assert pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative():
    assert pearson_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_correlation_constant_series_returns_zero():
    # Undefined correlation (zero variance) should degrade to 0.0, not raise.
    assert pearson_correlation([1, 1, 1], [1, 2, 3]) == 0.0


def test_sweep_subsignal_weights_picks_best_correlation():
    raw_signals = [
        {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.9},
        {"eyes_open": 0.2, "head_deviation": 0.8, "gaze_centered": 0.3},
        {"eyes_open": 0.8, "head_deviation": 0.2, "gaze_centered": 0.7},
    ]
    ground_truth = [0.9, 0.2, 0.75]  # matches eyes_open closely

    results = sweep_subsignal_weights(raw_signals, ground_truth, step=0.2)
    assert len(results) > 0
    best = max(results, key=lambda r: r["correlation"])
    assert best["correlation"] > 0.9
    assert best["weights"][0] > 0.3  # eyes_open weight should dominate


def test_sweep_fusion_weights_returns_sorted_by_correlation():
    video_scores = [0.9, 0.2, 0.8]
    speech_scores = [0.85, 0.25, 0.75]
    results = sweep_fusion_weights(video_scores, speech_scores, step=0.1)
    assert results == sorted(results, key=lambda r: -r["correlation"])
    assert all(abs(r["w_video"] + r["w_speech"] - 1.0) < 1e-9 for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/video_engagement && python -m pytest tests/test_weight_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'weight_search'`

- [ ] **Step 3: Implement**

`apps/video_engagement/weight_search.py`:

```python
"""Offline experiment: pick sub-signal and fusion weights by correlating
video_attention_score against the existing speech-based engagement_score
(the only per-learner ground truth this repo has — see spec section
"Weight search"). Not run automatically; invoke manually against a pilot
session's data:

    python weight_search.py --barter-id 1 --backend-url http://localhost:8000
"""

import argparse
import itertools
import math
import statistics

import httpx

from scoring import video_attention_score


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or statistics.pvariance(xs) == 0 or statistics.pvariance(ys) == 0:
        return 0.0
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def _weight_grid(step: float) -> list[tuple[float, float, float]]:
    """All (w1, w2, w3) triples on the step grid that sum to 1.0."""
    steps = [round(i * step, 4) for i in range(int(round(1.0 / step)) + 1)]
    grid = []
    for w1, w2 in itertools.product(steps, steps):
        w3 = round(1.0 - w1 - w2, 4)
        if 0.0 <= w3 <= 1.0:
            grid.append((w1, w2, round(w3, 4)))
    return grid


def sweep_subsignal_weights(
    raw_signals: list[dict], ground_truth: list[float], step: float = 0.1
) -> list[dict]:
    """Try every (w_eyes, w_head, w_gaze) on the grid, correlate resulting
    video_attention_score series against ground_truth, return all results."""
    results = []
    for weights in _weight_grid(step):
        scores = [video_attention_score(s, weights) for s in raw_signals]
        corr = pearson_correlation(scores, ground_truth)
        results.append({"weights": weights, "correlation": corr})
    return sorted(results, key=lambda r: -r["correlation"])


def sweep_fusion_weights(
    video_scores: list[float], speech_scores: list[float], step: float = 0.1
) -> list[dict]:
    """Try every (w_video, w_speech) pair summing to 1.0, correlate the fused
    series against the speech series alone (sanity: fused should track speech
    reasonably while still being influenced by video)."""
    results = []
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        w_video = round(i * step, 4)
        w_speech = round(1.0 - w_video, 4)
        fused = [w_speech * s + w_video * v for s, v in zip(speech_scores, video_scores)]
        corr = pearson_correlation(fused, speech_scores)
        results.append({"w_video": w_video, "w_speech": w_speech, "correlation": corr})
    return sorted(results, key=lambda r: -r["correlation"])


def _fetch_session_data(backend_url: str, barter_id: int) -> tuple[list[dict], list[float]]:
    video_rows = httpx.get(f"{backend_url}/session/{barter_id}/video-engagement").json()
    history_rows = httpx.get(f"{backend_url}/session/{barter_id}/engagement/history").json()

    raw_signals = [r["raw_signals"] for r in video_rows]
    speech_scores = [
        r["speech_engagement_score"] for r in history_rows
        if r["speech_engagement_score"] is not None
    ]
    return raw_signals, speech_scores


def _write_design_choices(path: str, subsignal_results: list[dict], fusion_results: list[dict] | None):
    lines = ["# Video Engagement — Weight Design Choices\n"]
    lines.append("Every weight below was picked by grid search + correlation against the\n"
                  "existing speech-based `engagement_score`, not hand-picked. See\n"
                  "`weight_search.py` and the design spec for the full method.\n")

    lines.append("\n## Sub-signal weights (eyes_open, head_deviation, gaze_centered)\n")
    lines.append("| w_eyes | w_head | w_gaze | correlation |\n|---|---|---|---|\n")
    for r in subsignal_results[:10]:
        w1, w2, w3 = r["weights"]
        lines.append(f"| {w1} | {w2} | {w3} | {r['correlation']:.4f} |\n")
    best = subsignal_results[0]
    lines.append(f"\n**Chosen:** `{best['weights']}` (correlation {best['correlation']:.4f})\n")

    if fusion_results:
        lines.append("\n## Fusion weights (w_speech, w_video)\n")
        lines.append("| w_speech | w_video | correlation |\n|---|---|---|\n")
        for r in fusion_results[:10]:
            lines.append(f"| {r['w_speech']} | {r['w_video']} | {r['correlation']:.4f} |\n")
        best_fusion = fusion_results[0]
        lines.append(f"\n**Chosen:** w_speech={best_fusion['w_speech']}, w_video={best_fusion['w_video']}\n")

    with open(path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--barter-id", type=int, required=True)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--out", default="../../docs/video_engagement/design-choices.md")
    args = parser.parse_args()

    raw_signals, speech_scores = _fetch_session_data(args.backend_url, args.barter_id)
    if len(raw_signals) < 2:
        raise SystemExit("Not enough data yet — need a pilot session with real windows first.")

    ground_truth = speech_scores[: len(raw_signals)]
    subsignal_results = sweep_subsignal_weights(raw_signals, ground_truth, step=args.step)

    video_scores = [video_attention_score(s, subsignal_results[0]["weights"]) for s in raw_signals]
    fusion_results = None
    if len(speech_scores) >= 2:
        fusion_results = sweep_fusion_weights(
            video_scores[: len(speech_scores)], speech_scores[: len(video_scores)], step=args.step
        )

    _write_design_choices(args.out, subsignal_results, fusion_results)
    print(f"Wrote {args.out}")
```

Seed `docs/video_engagement/design-choices.md` (checked in now; overwritten once real pilot data exists):

```markdown
# Video Engagement — Weight Design Choices

Status: **placeholder — not yet run against real data.**

`apps/video_engagement/scoring.py`'s `DEFAULT_WEIGHTS = (0.4, 0.4, 0.2)` and
`warning_engine`'s `ENGAGEMENT_FUSION_W_SPEECH=0.7` / `ENGAGEMENT_FUSION_W_VIDEO=0.3`
are engineering guesses, not experimentally chosen values.

Run `python apps/video_engagement/weight_search.py --barter-id <id>` against a
pilot session's real data to replace this file with actual grid-search
results and a chosen weight set, per the method in
`docs/superpowers/specs/2026-09-17-video-engagement-design.md`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/video_engagement && python -m pytest tests/test_weight_search.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/video_engagement/weight_search.py apps/video_engagement/tests/test_weight_search.py docs/video_engagement/design-choices.md
git commit -m "feat(video_engagement): offline weight_search script + placeholder design-choices doc"
```

---

### Task 12: Frontend — capture and stream video frames

**Files:**
- Modify: `apps/frontend/src/screens/LiveSession.jsx`

**Interfaces:**
- Consumes: `ws /video/{barterId}/{userId}` (Task 9, proxied per Task 10).
- Produces: none consumed by later tasks — this is the last task.

- [ ] **Step 1: Manual verification plan (no automated frontend test in this repo's conventions — none exist for `.jsx` files)**

After Step 2's change, verify manually: run `docker compose up --build`, open two browser tabs for the two session participants, start a session, open each tab's DevTools → Network → WS, confirm a connection to `/video/{barterId}/{userId}` with binary frames flowing, and confirm (via `docker compose logs -f video_engagement`) that windows are being processed every ~5s.

- [ ] **Step 2: Add video capture + WS send**

In `apps/frontend/src/screens/LiveSession.jsx`, add near the top with the other WS URL constants:

```javascript
const VIDEO_WS = `${WS}://${location.host}`
```

Add a ref alongside the existing ones (near `audioWsRef`, `mrRef`, `frameIntervalRef`):

```javascript
const videoWsRef = useRef(null)
const videoFrameIntervalRef = useRef(null)
```

In the session-start handler, right after the existing `frameIntervalRef.current = setInterval(...)` block (the NSFW check interval), add a second interval that reuses the same `localVideoElRef` canvas capture but streams to the new video_engagement WS at ~1fps:

```javascript
      const videoWs = new WebSocket(`${VIDEO_WS}/video/${barterId}/${userId}`)
      videoWsRef.current = videoWs
      await new Promise((resolve, reject) => { videoWs.onopen = resolve; videoWs.onerror = reject })

      videoFrameIntervalRef.current = setInterval(() => {
        if (!localVideoElRef.current || localVideoElRef.current.videoWidth === 0) return
        if (videoWs.readyState !== WebSocket.OPEN) return
        const canvas = document.createElement('canvas')
        canvas.width  = localVideoElRef.current.videoWidth
        canvas.height = localVideoElRef.current.videoHeight
        canvas.getContext('2d').drawImage(localVideoElRef.current, 0, 0)
        canvas.toBlob((blob) => {
          if (blob && videoWs.readyState === WebSocket.OPEN) videoWs.send(blob)
        }, 'image/jpeg', 0.7)
      }, 1_000)
```

Find the existing cleanup/stop logic that clears `frameIntervalRef.current` and closes `audioWsRef.current` (session end / unmount) and add matching cleanup:

```javascript
      if (videoFrameIntervalRef.current) clearInterval(videoFrameIntervalRef.current)
      if (videoWsRef.current) videoWsRef.current.close()
```

- [ ] **Step 3: Manual verification**

Run: `docker compose up --build`
Then follow the Step 1 verification plan. Confirm no console errors from the new WS connection and that `apps/video_engagement` logs show windows being processed.

- [ ] **Step 4: Commit**

```bash
git add apps/frontend/src/screens/LiveSession.jsx
git commit -m "feat(frontend): stream webcam frames to video_engagement service"
```

---

## Post-plan note

Real weight tuning (Task 11's `weight_search.py` actually run against real data, replacing the placeholder `docs/video_engagement/design-choices.md`) requires a pilot session with the full pipeline live — that's a follow-up action after this plan ships, not a task here, since it needs real recorded engagement + video data to run against.
