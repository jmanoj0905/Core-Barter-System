# Per-User Gaze/Head-Pose Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Personalize `video_engagement`'s `head_deviation`/`gaze_centered` sub-signals against a short live per-user calibration sample, closing the documented domain-gap in the fixed-0.5-neutral-point heuristic, with a cooldown-limited mid-session recalibration path and a full audit trail.

**Architecture:** A new `POST /video/{barter_id}/{user_id}/calibrate` endpoint on `video_engagement` runs the existing MediaPipe landmark pipeline on a short burst of frames, averages two raw ratios (head, gaze) instead of the final 0-1 scores, and stores the result in the same in-memory per-session state that already holds frame buffers. `scoring.py`'s sub-signal formulas gain an optional `baseline` parameter that recenters "neutral" on that stored value instead of a hardcoded `0.5`, with an algebraic identity guaranteeing bit-identical output when no calibration exists. Every calibration attempt (success or failure) is logged to a new `backend` table for audit/evaluation. A new frontend calibration step and a "Recalibrate" control wire this into the live session flow.

**Tech Stack:** Python/FastAPI (`video_engagement`, `backend`), SQLAlchemy async ORM, MediaPipe Face Mesh, React (frontend), pytest/pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-29-gaze-calibration-design.md`

## Global Constraints

- Calibration is fails-open: it never blocks session start or scoring — every failure path falls back to today's fixed `{head_ratio: 0.5, gaze_ratio: 0.5}` behavior.
- Backward compatibility: `baseline={"head_ratio": 0.5, "gaze_ratio": 0.5}` (or no baseline) must reproduce today's `head_deviation`/`gaze_centered` output exactly — existing `test_scoring.py`/`test_weight_search.py` tests must keep passing unmodified.
- Recalibration cooldown: exactly `120.0` seconds since the last successful calibration for that `(barter_id, user_id)`. The first calibration in a session is never subject to the cooldown.
- Minimum viable calibration: at least `3` frames with a detected face, out of whatever the frontend sends.
- No cross-session persistence — calibration state lives only in `video_engagement`'s in-memory dict for the session's lifetime; `backend`'s copy is an audit log only, never read back for live scoring.
- Every calibration/recalibration attempt (success or failure) is POSTed to `backend` as an audit record; a failure to log never fails the calibration call itself (fire-and-forget, matching `video_engagement`'s existing `_score_and_post` pattern).

## Review Focus

- **No face detected in any calibration frame** (camera off, bad lighting, face out of frame): must return `{"calibrated": false, "reason": "insufficient_face_detections"}` and leave any existing baseline untouched, never a 500 or an exception.
- **Recalibration attempted exactly at/just under the 120s cooldown boundary**: a call at `elapsed < 120.0` must be rejected with `retry_after_seconds > 0`; a call at `elapsed >= 120.0` must succeed if the frames qualify — an off-by-one here silently breaks the mitigation the design relies on.
- **Two users in the same barter session calibrate independently**: `calibration_baselines` is keyed by `(barter_id, user_id)`; one user's calibration or cooldown must never affect the other's.
- **A window is scored for a user who never calibrated** (or whose entry was already cleaned up): `process_buffer`'s baseline lookup must fall back to the default `0.5`/`0.5` baseline, never `KeyError`.
- **The backend audit POST fails** (network error, backend down): the `/calibrate` endpoint must still return its normal success/failure response to the caller — the log POST is best-effort and wrapped so it can never raise into the endpoint's response path.

---

## Task 1: Baseline-relative scoring math (`video_engagement/scoring.py`)

**Files:**
- Modify: `apps/video_engagement/scoring.py`
- Test: `apps/video_engagement/tests/test_scoring.py`

**Interfaces:**
- Produces: `_deviation_from_baseline(value: float, baseline: float) -> float`
- Produces: `raw_ratios_from_landmarks(landmarks: dict[int, tuple[float, float]]) -> dict[str, float | None]` — returns `{"head_ratio": float | None, "gaze_ratio": float}`. `head_ratio` is `None` only in the degenerate case where the cheek span is zero (mirrors today's `span == 0` guard, which previously forced `head_deviation = 1.0` unconditionally).
- Modifies: `sub_signals_from_mediapipe_landmarks(landmarks: dict[int, tuple[float, float]], baseline: dict[str, float] | None = None) -> dict[str, float]` — `baseline` defaults to `{"head_ratio": 0.5, "gaze_ratio": 0.5}` when omitted.

- [ ] **Step 1: Write the failing tests**

```python
# test_scoring.py additions
from scoring import _deviation_from_baseline, raw_ratios_from_landmarks, sub_signals_from_mediapipe_landmarks

def test_deviation_from_baseline_at_baseline_is_zero():
    assert _deviation_from_baseline(0.5, 0.5) == 0.0

def test_deviation_from_baseline_matches_old_formula_when_baseline_is_half():
    # old formula: abs(x - 0.5) * 2.0
    for x in (0.0, 0.2, 0.5, 0.8, 1.0):
        assert _deviation_from_baseline(x, 0.5) == pytest.approx(min(1.0, abs(x - 0.5) * 2.0))

def test_deviation_from_baseline_asymmetric_off_center_baseline():
    # baseline 0.2: max_dev = max(0.2, 0.8) = 0.8
    assert _deviation_from_baseline(1.0, 0.2) == pytest.approx(0.8 / 0.8)
    assert _deviation_from_baseline(0.2, 0.2) == 0.0

def test_sub_signals_no_baseline_matches_default_baseline():
    landmarks = _make_frontal_landmarks()  # existing test fixture/helper in this file
    no_baseline = sub_signals_from_mediapipe_landmarks(landmarks)
    explicit_default = sub_signals_from_mediapipe_landmarks(
        landmarks, baseline={"head_ratio": 0.5, "gaze_ratio": 0.5}
    )
    assert no_baseline == explicit_default

def test_sub_signals_default_baseline_matches_pre_calibration_formula():
    # Regression guard: with the default baseline, head_deviation/gaze_centered
    # must equal today's fixed-0.5 formula bit-for-bit for a known landmark set.
    landmarks = _make_frontal_landmarks()
    result = sub_signals_from_mediapipe_landmarks(landmarks)
    ratios = raw_ratios_from_landmarks(landmarks)
    expected_head_dev = min(1.0, abs(ratios["head_ratio"] - 0.5) * 2.0)
    expected_gaze_centered = 1.0 - min(1.0, abs(ratios["gaze_ratio"] - 0.5) * 2.0)
    assert result["head_deviation"] == pytest.approx(expected_head_dev)
    assert result["gaze_centered"] == pytest.approx(expected_gaze_centered)

def test_sub_signals_personalized_baseline_shifts_neutral_point():
    landmarks = _make_frontal_landmarks()
    ratios = raw_ratios_from_landmarks(landmarks)
    # Calibrating baseline to this exact frame's ratios must yield zero deviation.
    result = sub_signals_from_mediapipe_landmarks(
        landmarks, baseline={"head_ratio": ratios["head_ratio"], "gaze_ratio": ratios["gaze_ratio"]}
    )
    assert result["head_deviation"] == pytest.approx(0.0)
    assert result["gaze_centered"] == pytest.approx(1.0)

def test_raw_ratios_degenerate_span_returns_none_head_ratio():
    landmarks = _make_frontal_landmarks()
    landmarks[_LEFT_CHEEK] = landmarks[_RIGHT_CHEEK]  # force span == 0
    ratios = raw_ratios_from_landmarks(landmarks)
    assert ratios["head_ratio"] is None

def test_sub_signals_degenerate_span_forces_max_head_deviation_regardless_of_baseline():
    landmarks = _make_frontal_landmarks()
    landmarks[_LEFT_CHEEK] = landmarks[_RIGHT_CHEEK]
    result = sub_signals_from_mediapipe_landmarks(landmarks, baseline={"head_ratio": 0.9, "gaze_ratio": 0.5})
    assert result["head_deviation"] == 1.0
```

If `test_scoring.py` has no existing landmark-construction helper (`_make_frontal_landmarks` or similar), write one first: a `dict[int, tuple[float, float]]` covering every index `scoring.py` reads (`_RIGHT_EYE`, `_LEFT_EYE`, `_NOSE_TIP`, `_LEFT_CHEEK`, `_RIGHT_CHEEK`, `_LEFT_IRIS_CENTER`, `_RIGHT_IRIS_CENTER`), values chosen so the face reads as roughly frontal/eyes-open (check the existing test file for `sub_signals_from_mediapipe_landmarks` fixtures already in use and reuse/extend them rather than inventing a second one).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/video_engagement && venv/bin/pytest tests/test_scoring.py -v`
Expected: FAIL — `_deviation_from_baseline`/`raw_ratios_from_landmarks` not defined, or `sub_signals_from_mediapipe_landmarks` rejects the `baseline` kwarg.

- [ ] **Step 3: Implement in `apps/video_engagement/scoring.py`**

Add `_deviation_from_baseline` exactly as specified:

```python
def _deviation_from_baseline(value: float, baseline: float) -> float:
    max_dev = max(baseline, 1.0 - baseline)
    if max_dev == 0:
        return 0.0
    return _clamp01(abs(value - baseline) / max_dev)
```

Extract the existing inline `ratio`/`avg_gaze_ratio` computation out of `sub_signals_from_mediapipe_landmarks` into `raw_ratios_from_landmarks`, preserving the `span == 0` guard as a `None` head_ratio (previously this guard set `head_deviation = 1.0` directly; keep that behavior by having the caller check for `None`). Then rewrite `sub_signals_from_mediapipe_landmarks` to call `raw_ratios_from_landmarks`, default `baseline`, and use `_deviation_from_baseline` for both `head_deviation` and `gaze_centered`. Do not change the `eyes_open`/EAR computation — it is untouched by this task.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/video_engagement && venv/bin/pytest tests/test_scoring.py -v`
Expected: PASS, all tests including pre-existing ones in the file.

- [ ] **Step 5: Run the full existing video_engagement test suite to confirm no regression**

Run: `cd apps/video_engagement && venv/bin/pytest -v`
Expected: PASS — `test_weight_search.py` and `test_main.py` must be unaffected (Task 2 wires `main.py` into the new signature next).

- [ ] **Step 6: Commit**

```bash
git add apps/video_engagement/scoring.py apps/video_engagement/tests/test_scoring.py
git commit -m "feat: personalize video engagement sub-signals against a per-user baseline"
```

---

## Task 2: Calibration audit log (`backend`)

**Files:**
- Modify: `apps/backend/app/models.py`
- Modify: `apps/backend/app/schemas.py`
- Modify: `apps/backend/app/routes.py`
- Test: `apps/backend/tests/test_calibration_log_routes.py`

**Interfaces:**
- Produces: model `CalibrationLog` (table `calibration_logs`), columns: `id`, `barter_session_id` (FK `barter_sessions.id`), `user_id` (FK `users.id`), `outcome` (`str`, one of `"calibrated"`, `"insufficient_face_detections"`, `"recalibration_limit_reached"`), `old_baseline_head_ratio: float | None`, `old_baseline_gaze_ratio: float | None`, `new_baseline_head_ratio: float | None`, `new_baseline_gaze_ratio: float | None`, `recalibration_count: int`, `created_at`.
- Produces: schema `CalibrationLogRequest(user_id: int, outcome: str, old_baseline_head_ratio: float | None = None, old_baseline_gaze_ratio: float | None = None, new_baseline_head_ratio: float | None = None, new_baseline_gaze_ratio: float | None = None, recalibration_count: int)`.
- Produces: `POST /session/{barter_id}/video-engagement/calibration-log` → `{"status": "stored", "id": int}`. `GET /session/{barter_id}/video-engagement/calibration-log` → list of records ordered by `created_at`, same field names as the request plus `created_at` (ISO string).

- [ ] **Step 1: Write the failing tests**

```python
# apps/backend/tests/test_calibration_log_routes.py
import pytest
from app.models import CalibrationLog


@pytest.mark.asyncio
async def test_calibration_log_model_roundtrip(backend_client, db_session):
    row = CalibrationLog(
        barter_session_id=1, user_id=2, outcome="calibrated",
        old_baseline_head_ratio=None, old_baseline_gaze_ratio=None,
        new_baseline_head_ratio=0.51, new_baseline_gaze_ratio=0.48,
        recalibration_count=0,
    )
    db_session.add(row)
    await db_session.commit()
    assert row.id is not None


@pytest.mark.asyncio
async def test_post_and_list_calibration_log(backend_client):
    resp = await backend_client.post("/session/1/video-engagement/calibration-log", json={
        "user_id": 2, "outcome": "calibrated",
        "old_baseline_head_ratio": None, "old_baseline_gaze_ratio": None,
        "new_baseline_head_ratio": 0.51, "new_baseline_gaze_ratio": 0.48,
        "recalibration_count": 0,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "stored"

    resp = await backend_client.post("/session/1/video-engagement/calibration-log", json={
        "user_id": 2, "outcome": "recalibration_limit_reached",
        "old_baseline_head_ratio": 0.51, "old_baseline_gaze_ratio": 0.48,
        "new_baseline_head_ratio": None, "new_baseline_gaze_ratio": None,
        "recalibration_count": 0,
    })
    assert resp.status_code == 200

    resp = await backend_client.get("/session/1/video-engagement/calibration-log")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert rows[0]["outcome"] == "calibrated"
    assert rows[1]["outcome"] == "recalibration_limit_reached"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/backend && venv/bin/pytest tests/test_calibration_log_routes.py -v`
Expected: FAIL — `CalibrationLog` not importable / route 404.

- [ ] **Step 3: Add `CalibrationLog` to `apps/backend/app/models.py`**

Place it directly after `EngagementScoreLog` (same section as the other video-engagement-adjacent tables), matching that class's column style exactly.

- [ ] **Step 4: Add `CalibrationLogRequest` to `apps/backend/app/schemas.py`**

Place it directly after `VideoEngagementRequest`.

- [ ] **Step 5: Add routes to `apps/backend/app/routes.py`**

Add `POST`/`GET /session/{barter_id}/video-engagement/calibration-log` in the "Video Engagement Results" section, immediately after the existing `list_video_engagement` function, following that function pair's exact shape (`db.add`/`db.commit`/return `{"status": "stored", "id": row.id}` for POST; `select().where().order_by(CalibrationLog.created_at)` for GET, serializing `created_at` with `.isoformat()`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd apps/backend && venv/bin/pytest tests/test_calibration_log_routes.py -v`
Expected: PASS

- [ ] **Step 7: Run the full backend test suite to confirm no regression**

Run: `cd apps/backend && venv/bin/pytest -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add apps/backend/app/models.py apps/backend/app/schemas.py apps/backend/app/routes.py apps/backend/tests/test_calibration_log_routes.py
git commit -m "feat: add calibration audit log table and routes to backend"
```

---

## Task 3: Calibration endpoint (`video_engagement/main.py`)

**Files:**
- Modify: `apps/video_engagement/main.py`
- Test: `apps/video_engagement/tests/test_main.py`

**Interfaces:**
- Consumes: `raw_ratios_from_landmarks`, `sub_signals_from_mediapipe_landmarks(landmarks, baseline=None)` from Task 1 (`scoring.py`).
- Consumes: `POST /session/{barter_id}/video-engagement/calibration-log` from Task 2 (`backend`).
- Produces: `POST /video/{barter_id}/{user_id}/calibrate` (multipart `frames: list[UploadFile]`) → `{"calibrated": true, "baseline_head_ratio": float, "baseline_gaze_ratio": float, "frames_used": int, "recalibration_count": int}` on success, or `{"calibrated": false, "reason": "insufficient_face_detections"}` / `{"calibrated": false, "reason": "recalibration_limit_reached", "retry_after_seconds": float}` on failure.
- Produces: module-level `calibration_baselines: dict[tuple[int, int], dict]` (keys `(barter_id, user_id)`, values `{"head_ratio": float, "gaze_ratio": float, "recalibration_count": int, "last_calibrated_at": float}`) — read by `process_buffer`/`process_frame_local` for scoring, written only by the calibrate endpoint.
- Produces constants: `CALIBRATION_COOLDOWN_SECONDS = 120.0`, `CALIBRATION_MIN_FACE_FRAMES = 3`.

- [ ] **Step 1: Write the failing tests**

```python
# test_main.py additions
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import main


def _frontal_jpeg_bytes():
    # Reuse this file's _make_jpeg_bytes() helper as the base; calibration
    # tests patch main._landmarks_from_frame_bytes directly rather than
    # relying on a JPEG that MediaPipe can actually detect a face in, since
    # this test suite does not ship real face imagery (see existing
    # test_process_frame_local_no_face_returns_none for the same pattern).
    return _make_jpeg_bytes()


def test_calibrate_success_stores_baseline_and_logs_audit():
    main.calibration_baselines.clear()
    main.http_client = AsyncMock()
    with patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch("scoring.raw_ratios_from_landmarks", return_value={"head_ratio": 0.51, "gaze_ratio": 0.48}):
        with TestClient(main.app) as client:
            resp = client.post(
                "/video/1/2/calibrate",
                files=[("frames", ("f.jpg", _frontal_jpeg_bytes(), "image/jpeg"))] * 3,
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "calibrated": True, "baseline_head_ratio": 0.51, "baseline_gaze_ratio": 0.48,
        "frames_used": 3, "recalibration_count": 0,
    }
    assert main.calibration_baselines[(1, 2)]["head_ratio"] == 0.51
    posted_paths = [c.args[0] for c in main.http_client.post.call_args_list]
    assert any("/calibration-log" in p for p in posted_paths)


def test_calibrate_insufficient_faces_falls_back():
    main.calibration_baselines.clear()
    main.http_client = AsyncMock()
    with patch.object(main, "_landmarks_from_frame_bytes", return_value=None):
        with TestClient(main.app) as client:
            resp = client.post(
                "/video/1/2/calibrate",
                files=[("frames", ("f.jpg", _frontal_jpeg_bytes(), "image/jpeg"))] * 2,
            )
    assert resp.json() == {"calibrated": False, "reason": "insufficient_face_detections"}
    assert (1, 2) not in main.calibration_baselines


def test_calibrate_within_cooldown_is_rejected():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = {
        "head_ratio": 0.5, "gaze_ratio": 0.5, "recalibration_count": 0,
        "last_calibrated_at": main.time.time(),
    }
    main.http_client = AsyncMock()
    with TestClient(main.app) as client:
        resp = client.post(
            "/video/1/2/calibrate",
            files=[("frames", ("f.jpg", _frontal_jpeg_bytes(), "image/jpeg"))] * 3,
        )
    body = resp.json()
    assert body["calibrated"] is False
    assert body["reason"] == "recalibration_limit_reached"
    assert body["retry_after_seconds"] > 0


def test_calibrate_after_cooldown_elapsed_succeeds():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = {
        "head_ratio": 0.5, "gaze_ratio": 0.5, "recalibration_count": 0,
        "last_calibrated_at": main.time.time() - main.CALIBRATION_COOLDOWN_SECONDS - 1,
    }
    main.http_client = AsyncMock()
    with patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch("scoring.raw_ratios_from_landmarks", return_value={"head_ratio": 0.6, "gaze_ratio": 0.4}):
        with TestClient(main.app) as client:
            resp = client.post(
                "/video/1/2/calibrate",
                files=[("frames", ("f.jpg", _frontal_jpeg_bytes(), "image/jpeg"))] * 3,
            )
    body = resp.json()
    assert body["calibrated"] is True
    assert body["recalibration_count"] == 1


def test_calibrate_two_users_independent():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = {
        "head_ratio": 0.5, "gaze_ratio": 0.5, "recalibration_count": 0,
        "last_calibrated_at": main.time.time(),
    }
    main.http_client = AsyncMock()
    with patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch("scoring.raw_ratios_from_landmarks", return_value={"head_ratio": 0.55, "gaze_ratio": 0.45}):
        with TestClient(main.app) as client:
            resp = client.post(
                "/video/1/3/calibrate",
                files=[("frames", ("f.jpg", _frontal_jpeg_bytes(), "image/jpeg"))] * 3,
            )
    assert resp.json()["calibrated"] is True  # user 3's calibration unaffected by user 2's cooldown


def test_calibrate_backend_log_post_failure_does_not_break_response():
    main.calibration_baselines.clear()
    main.http_client = AsyncMock()
    main.http_client.post.side_effect = Exception("backend down")
    with patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch("scoring.raw_ratios_from_landmarks", return_value={"head_ratio": 0.5, "gaze_ratio": 0.5}):
        with TestClient(main.app) as client:
            resp = client.post(
                "/video/1/2/calibrate",
                files=[("frames", ("f.jpg", _frontal_jpeg_bytes(), "image/jpeg"))] * 3,
            )
    assert resp.status_code == 200
    assert resp.json()["calibrated"] is True


@pytest.mark.asyncio
async def test_process_buffer_uses_stored_baseline():
    import time as time_mod
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = {
        "head_ratio": 0.6, "gaze_ratio": 0.4, "recalibration_count": 0,
        "last_calibrated_at": time_mod.time(),
    }
    main.http_client = AsyncMock()
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time_mod.time()}

    captured = {}
    def fake_process_frame_local(frame_bytes, baseline=None):
        captured["baseline"] = baseline
        return {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.85}

    with patch.object(main, "process_frame_local", side_effect=fake_process_frame_local):
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="local")

    assert captured["baseline"] == {"head_ratio": 0.6, "gaze_ratio": 0.4}


@pytest.mark.asyncio
async def test_process_buffer_no_baseline_defaults_to_none():
    main.calibration_baselines.clear()
    main.http_client = AsyncMock()
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": __import__("time").time()}

    captured = {}
    def fake_process_frame_local(frame_bytes, baseline=None):
        captured["baseline"] = baseline
        return {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.85}

    with patch.object(main, "process_frame_local", side_effect=fake_process_frame_local):
        await main.process_buffer(barter_id=1, user_id=99, buf=buf, backend="local")

    assert captured["baseline"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/video_engagement && venv/bin/pytest tests/test_main.py -v -k calibrat`
Expected: FAIL — endpoint/attributes don't exist yet.

- [ ] **Step 3: Implement in `apps/video_engagement/main.py`**

Factor the existing decode-and-MediaPipe steps out of `process_frame_local` into `_landmarks_from_frame_bytes(frame_bytes: bytes) -> dict[int, tuple[float, float]] | None`, then have `process_frame_local(frame_bytes: bytes, baseline: dict | None = None) -> dict | None` call it and pass `baseline` through to `sub_signals_from_mediapipe_landmarks`.

Add module-level `CALIBRATION_COOLDOWN_SECONDS = 120.0`, `CALIBRATION_MIN_FACE_FRAMES = 3`, `calibration_baselines: dict[tuple[int, int], dict] = {}`.

Add `_compute_calibration_baseline(frames: list[bytes]) -> dict | None` (returns `{"head_ratio": float, "gaze_ratio": float, "frames_used": int}` or `None` if fewer than `CALIBRATION_MIN_FACE_FRAMES` frames yield both a detected face and a non-`None` `head_ratio`): for each frame, call `_landmarks_from_frame_bytes`, skip `None`; call `scoring.raw_ratios_from_landmarks`, skip entries with `head_ratio is None`; average the survivors.

Add `async def _post_calibration_log(barter_id, user_id, outcome, old_baseline, new_baseline, recalibration_count)`: builds the payload per Task 2's `CalibrationLogRequest` shape and POSTs to `f"{BACKEND_URL}/session/{barter_id}/video-engagement/calibration-log"`, wrapped in `try/except Exception as e: logger.error(...)` exactly like `_score_and_post`'s existing backend POST.

Add the endpoint:

```python
@app.post("/video/{barter_id}/{user_id}/calibrate")
async def calibrate(barter_id: int, user_id: int, frames: list[UploadFile] = File(...)):
    key = (barter_id, user_id)
    existing = calibration_baselines.get(key)
    now = time.time()
    recalibration_count = existing["recalibration_count"] if existing else 0

    if existing is not None:
        elapsed = now - existing["last_calibrated_at"]
        if elapsed < CALIBRATION_COOLDOWN_SECONDS:
            retry_after = round(CALIBRATION_COOLDOWN_SECONDS - elapsed, 1)
            await _post_calibration_log(barter_id, user_id, "recalibration_limit_reached", existing, None, recalibration_count)
            return {"calibrated": False, "reason": "recalibration_limit_reached", "retry_after_seconds": retry_after}

    frame_bytes = [await f.read() for f in frames]
    baseline = _compute_calibration_baseline(frame_bytes)
    if baseline is None:
        await _post_calibration_log(barter_id, user_id, "insufficient_face_detections", existing, None, recalibration_count)
        return {"calibrated": False, "reason": "insufficient_face_detections"}

    new_count = recalibration_count + 1 if existing else 0
    calibration_baselines[key] = {
        "head_ratio": baseline["head_ratio"], "gaze_ratio": baseline["gaze_ratio"],
        "recalibration_count": new_count, "last_calibrated_at": now,
    }
    await _post_calibration_log(barter_id, user_id, "calibrated", existing, calibration_baselines[key], new_count)
    return {
        "calibrated": True, "baseline_head_ratio": baseline["head_ratio"],
        "baseline_gaze_ratio": baseline["gaze_ratio"], "frames_used": baseline["frames_used"],
        "recalibration_count": new_count,
    }
```

Add the `File`, `UploadFile` imports from `fastapi` alongside the existing `HTTPException, WebSocket, WebSocketDisconnect` import line.

Wire baseline lookup into `process_buffer`: where it currently calls `process_frame_local(f)` inside the list comprehension, fetch `baseline = calibration_baselines.get((barter_id, user_id))` once at the top of `process_buffer` (translate the stored dict's `head_ratio`/`gaze_ratio` keys into the `{"head_ratio": ..., "gaze_ratio": ...}` shape `sub_signals_from_mediapipe_landmarks` expects, or `None` if absent) and pass it through: `process_frame_local(f, baseline=baseline)`.

Wire cleanup: in `video_ws`'s `except (WebSocketDisconnect, RuntimeError)` block and in `end_session`, add `calibration_baselines.pop(key, None)` alongside the existing `buffers` cleanup (`key = (barter_id, user_id)` already exists in scope in both places).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/video_engagement && venv/bin/pytest tests/test_main.py -v`
Expected: PASS, all tests including pre-existing ones.

- [ ] **Step 5: Run the full video_engagement test suite**

Run: `cd apps/video_engagement && venv/bin/pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add apps/video_engagement/main.py apps/video_engagement/tests/test_main.py
git commit -m "feat: add calibration endpoint with cooldown-limited recalibration"
```

---

## Task 4: Frontend calibration UI

**Files:**
- Create: `apps/frontend/src/screens/GazeCalibration.jsx`
- Modify: `apps/frontend/src/screens/LiveSession.jsx`

**Interfaces:**
- Consumes: `POST /video/{barterId}/{userId}/calibrate` from Task 3, via relative `fetch` (same-origin, proxied by nginx — no new base-URL constant, matching the existing `API = ''` pattern in `LiveSession.jsx`).
- Produces: `GazeCalibration` component, props `{ videoEl: HTMLVideoElement, barterId: number, userId: number, onDone: (result: {calibrated: boolean, reason?: string}) => void }`. Captures frames from `videoEl` via canvas (same `drawImage`/`toBlob` pattern already used at `LiveSession.jsx:289-295`) for ~5 seconds at 1 frame/sec, submits them as a single multipart `POST`, then calls `onDone`.

- [ ] **Step 1: Build `GazeCalibration.jsx`**

```jsx
import { useEffect, useRef, useState } from 'react'

export default function GazeCalibration({ videoEl, barterId, userId, onDone }) {
  const [status, setStatus] = useState('capturing') // capturing | submitting | done
  const framesRef = useRef([])

  useEffect(() => {
    let cancelled = false
    let count = 0
    const interval = setInterval(() => {
      if (cancelled || !videoEl || videoEl.videoWidth === 0) return
      const canvas = document.createElement('canvas')
      canvas.width = videoEl.videoWidth
      canvas.height = videoEl.videoHeight
      canvas.getContext('2d').drawImage(videoEl, 0, 0)
      canvas.toBlob((blob) => { if (blob) framesRef.current.push(blob) }, 'image/jpeg', 0.7)
      count += 1
      if (count >= 5) {
        clearInterval(interval)
        submit()
      }
    }, 1000)

    async function submit() {
      if (cancelled) return
      setStatus('submitting')
      const form = new FormData()
      framesRef.current.forEach((blob, i) => form.append('frames', blob, `frame${i}.jpg`))
      try {
        const res = await fetch(`/video/${barterId}/${userId}/calibrate`, { method: 'POST', body: form })
        const body = await res.json()
        setStatus('done')
        onDone(body)
      } catch {
        setStatus('done')
        onDone({ calibrated: false, reason: 'request_failed' })
      }
    }

    const timeout = setTimeout(() => { clearInterval(interval); submit() }, 6000)
    return () => { cancelled = true; clearInterval(interval); clearTimeout(timeout) }
  }, [videoEl, barterId, userId, onDone])

  return (
    <div className="calibration-overlay">
      {status === 'capturing' && <p>Look at your screen naturally for a few seconds…</p>}
      {status === 'submitting' && <p>Calibrating…</p>}
    </div>
  )
}
```

- [ ] **Step 2: Wire into `LiveSession.jsx`**

Add `const [calibrating, setCalibrating] = useState(false)` and `const [calibrationCooldownUntil, setCalibrationCooldownUntil] = useState(null)` near the other `useState` declarations (around line 36-49).

In `handleStart`, after `setupWebRTC(stream)` and before the `videoWs` block (i.e., once `localVideoElRef.current.srcObject` is set), set `setCalibrating(true)`; render `<GazeCalibration videoEl={localVideoElRef.current} barterId={barterId} userId={userId} onDone={handleCalibrationDone} />` conditionally on `calibrating`. `handleCalibrationDone(result)` sets `setCalibrating(false)` and, if `result.reason === 'recalibration_limit_reached'`, sets `calibrationCooldownUntil = Date.now() + result.retry_after_seconds * 1000`.

Add a "Recalibrate" button in the session controls JSX (near the existing mic/video toggle buttons, e.g. by `toggleLocalVideo`/`toggleRemoteMute`), disabled while `calibrationCooldownUntil && Date.now() < calibrationCooldownUntil`, `onClick` sets `calibrating` true (re-rendering `GazeCalibration` triggers a fresh capture-and-submit cycle).

This step does not include a Jest/RTL test file — the project has no existing frontend test suite to extend (confirm via `find apps/frontend -iname "*.test.*"` before writing code; if one exists, add a test for `GazeCalibration`'s capture-and-submit cycle using its established patterns instead of skipping this step).

- [ ] **Step 3: Manual verification**

Run: `docker compose up --build frontend video_engagement backend` (or the project's existing dev-run flow), start a session, confirm the calibration prompt appears, completes, and the Recalibrate button respects its cooldown.

- [ ] **Step 4: Commit**

```bash
git add apps/frontend/src/screens/GazeCalibration.jsx apps/frontend/src/screens/LiveSession.jsx
git commit -m "feat: add per-user gaze calibration step and recalibrate control to live session"
```

---

## Task 5: Head-pose personalization evaluation script

**Files:**
- Create: `apps/video_engagement/evaluate_calibration.py`
- Test: `apps/video_engagement/tests/test_evaluate_calibration.py`

**Interfaces:**
- Consumes: `parse_head_pose_filename`, `head_pose_label` from `apps/video_engagement/public_datasets.py`; `raw_ratios_from_landmarks`, `_deviation_from_baseline` from `scoring.py`; `process_frame_local`'s underlying `_landmarks_from_frame_bytes` pattern (this script reimplements a minimal local-only landmark call since it runs standalone over a dataset directory, not over the live service).
- Produces: `group_frames_by_subject(image_paths: list[Path]) -> dict[str, list[Path]]` (keyed by the `subject` field `parse_head_pose_filename` extracts).
- Produces: `evaluate_subject(subject_frames: list[Path]) -> dict | None` — returns `{"subject": str, "n_frames": int, "balanced_accuracy_uncalibrated": float, "balanced_accuracy_calibrated": float}`, or `None` if the subject has no frontal (`pan == 0 and tilt == 0`) frame or fewer than 2 other usable frames.
- Produces: CLI `main(argv=None) -> int`, invoked as `python evaluate_calibration.py --corpus datasets/head_pose_image_database`, printing a per-subject table and the pooled mean balanced-accuracy for calibrated vs. uncalibrated, and writing `evaluation-results.json` next to the corpus (mirrors `weight_search.py`'s `--corpus`/output-file convention).

- [ ] **Step 1: Write the failing tests**

```python
# test_evaluate_calibration.py
from unittest.mock import patch
from pathlib import Path

import evaluate_calibration as ec


def test_group_frames_by_subject_groups_correctly(tmp_path):
    files = [tmp_path / "Person01100-10+20.jpg", tmp_path / "Person01101+00+00.jpg",
             tmp_path / "Person02100-10+20.jpg"]
    for f in files:
        f.touch()
    grouped = ec.group_frames_by_subject(files)
    assert set(grouped.keys()) == {"Person01", "Person02"}
    assert len(grouped["Person01"]) == 2


def test_evaluate_subject_uses_frontal_frame_as_baseline(tmp_path):
    frontal = tmp_path / "Person01101+00+00.jpg"
    other1 = tmp_path / "Person01102-30+00.jpg"
    other2 = tmp_path / "Person01103+30+00.jpg"
    for f in (frontal, other1, other2):
        f.touch()

    ratios_by_file = {
        frontal: {"head_ratio": 0.5, "gaze_ratio": 0.5},
        other1: {"head_ratio": 0.65, "gaze_ratio": 0.5},
        other2: {"head_ratio": 0.35, "gaze_ratio": 0.5},
    }
    with patch.object(ec, "_raw_ratios_for_file", side_effect=lambda p: ratios_by_file[p]):
        result = ec.evaluate_subject([frontal, other1, other2])

    assert result["subject"] == "Person01"
    assert result["n_frames"] == 2
    assert 0.0 <= result["balanced_accuracy_calibrated"] <= 1.0
    assert 0.0 <= result["balanced_accuracy_uncalibrated"] <= 1.0


def test_evaluate_subject_returns_none_without_frontal_frame(tmp_path):
    other1 = tmp_path / "Person01102-30+00.jpg"
    other2 = tmp_path / "Person01103+30+00.jpg"
    for f in (other1, other2):
        f.touch()
    assert ec.evaluate_subject([other1, other2]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/video_engagement && venv/bin/pytest tests/test_evaluate_calibration.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `apps/video_engagement/evaluate_calibration.py`**

`group_frames_by_subject`: call `parse_head_pose_filename(path.name)` per file (catch/skip files that don't match, mirroring `convert_head_pose`'s existing tolerance in `public_datasets.py`), group by the returned `subject`.

`_raw_ratios_for_file(path: Path) -> dict | None`: reads the JPEG bytes, runs MediaPipe Face Mesh (same construction as `main._face_mesh`, but instantiate locally in this script rather than importing `main` — this script must run standalone without the FastAPI app/event loop), returns `raw_ratios_from_landmarks(...)` or `None` if no face detected. This is the only function that touches MediaPipe directly; the tests above patch it out entirely, so its body has no test coverage obligation beyond "does not crash on a real image," verified by the manual run in Step 5.

`evaluate_subject`: find the frame where `parse_head_pose_filename` gives `pan == 0 and tilt == 0` (the calibration frame); if none, return `None`. Compute that frame's baseline via `_raw_ratios_for_file`. For every other frame in the subject's list: get `head_pose_label(tilt, pan)` (skip `None` — the excluded ambiguous band, per `docs/threshold-calibration.md`); get its `_raw_ratios_for_file`; compute `head_deviation` two ways — `_deviation_from_baseline(ratio, 0.5)` (uncalibrated) and `_deviation_from_baseline(ratio, baseline["head_ratio"])` (calibrated); classify attentive if `head_deviation < 0.5` (midpoint decision rule — this script is evaluating the personalization transform in isolation, not the fitted `ATTENTIVE_SCORE_THRESHOLD`, which applies to the full fused score); compute balanced accuracy against `head_pose_label`'s ground truth (`"attentive"` vs. not) for both. Return `None` if fewer than 2 usable other frames.

`main`: `argparse` with `--corpus` (required) and `--output` (default `evaluation-results.json` in the corpus directory), glob `*.jpg` under the corpus dir, call `group_frames_by_subject`, run `evaluate_subject` per subject (skip `None`s), print a per-subject table plus pooled mean of `balanced_accuracy_calibrated` vs. `balanced_accuracy_uncalibrated`, write the full per-subject results list to `--output` as JSON.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/video_engagement && venv/bin/pytest tests/test_evaluate_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Manual run against the real corpus**

Run: `cd apps/video_engagement && venv/bin/python evaluate_calibration.py --corpus datasets/head_pose_image_database --output evaluation-results.json` (requires the dataset already fetched per `docs/video_engagement/design-choices.md`'s existing instructions). Confirm it completes without error and produces a plausible table (calibrated balanced accuracy should not be dramatically worse than uncalibrated — if it is, treat that as a real finding to write up, not a bug to hide).

- [ ] **Step 6: Commit**

```bash
git add apps/video_engagement/evaluate_calibration.py apps/video_engagement/tests/test_evaluate_calibration.py
git commit -m "feat: add head-pose personalization evaluation script"
```

---

## Task 6: Documentation update

**Files:**
- Modify: `docs/video_engagement/design-choices.md`
- Modify: `docs/threshold-calibration.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Add a section to `docs/video_engagement/design-choices.md`** noting that per-user calibration (this plan) sits on top of, not instead of, the fitted global weights (`0.55/0.40/0.05`) and threshold (`0.6123`) — personalization changes what "centered" means as an input to the same fitted formula, and does not require re-fitting. Reference the new evaluation script and where its results live.
- [ ] **Step 2: Update `docs/threshold-calibration.md`'s "What is not calibrated" section** — item 3 (domain gap) and item 4 (gaze/head-pose confound) are now partially addressed by per-session personalization; update their wording to reflect the mitigation without claiming the underlying public-dataset domain gap in the *global* fit is resolved (it isn't — only the per-session neutral point is now personalized).
- [ ] **Step 3: Commit**

```bash
git add docs/video_engagement/design-choices.md docs/threshold-calibration.md
git commit -m "docs: note per-user calibration layer on top of fitted video engagement weights"
```
