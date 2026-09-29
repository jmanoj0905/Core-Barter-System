# Per-User Gaze/Head-Pose Calibration — Design

Status: draft, pending user review
Date: 2026-09-29

## Purpose

This is sub-project 1 of 4 research-driven modifications to the
during-session QA/safety pipeline, targeting eventual publication (SMARTCOM
Goa 2027). Source research: see
`3_Capstone/during-session-qa-safety-novelty-research.md` (Obsidian vault),
Cluster 4.

`video_engagement`'s `head_deviation` and `gaze_centered` sub-signals
(`apps/video_engagement/scoring.py`) both assume a fixed geometric neutral
point — "centered" is hardcoded as ratio `0.5` — derived from datasets fit
offline on public, off-domain imagery (studio-lit head-pose photos,
synthetic eye-state images; see `docs/threshold-calibration.md`). This is a
documented domain-gap: what counts as "looking at the screen" varies by
person (facial geometry, camera angle, seating position) and by rig
(webcam position, lighting), and no calibration against the actual live
user happens today.

This design personalizes both sub-signals against a short per-user
calibration sample taken live, at session start and optionally
mid-session, sidestepping the need for any new public gaze dataset —
citing WebEyeTrack (arXiv:2508.19544) and MAGE (arXiv:2505.16384)'s
few-shot on-device personalization approach, adapted to this project's
existing landmark-ratio heuristics rather than a full gaze-regression
model (scoped-down variant, see research note Cluster 4 options A vs B —
this design implements option A).

## Non-goals

- No full 6-DoF gaze/point-of-regard regression (WebEyeTrack/MAGE's own
  approach) — out of scope for this sub-project.
- No cross-session persistence of calibration profiles — each session
  (re)calibrates fresh; nothing is reused across sessions.
- No change to the fusion weights (`0.55/0.40/0.05`) or attention
  threshold (`0.6123`) — personalization changes what "centered" means
  as an input, not the fitted combination downstream.

## Architecture & data flow

1. Frontend camera stream starts (as today) → shows a short calibration
   prompt ("look at your screen naturally") with a ~5s countdown →
   captures frames from the same local stream already in use.
2. Frontend `POST`s the captured JPEG frames to a new endpoint on
   `video_engagement`: `POST /video/{barter_id}/{user_id}/calibrate`.
3. `video_engagement` runs the *existing* MediaPipe landmark extraction
   (`process_frame_local`'s landmark step) on each frame, but instead of
   computing the final 0-1 sub-signal score, keeps the raw ratios
   (`head_ratio`, `avg_gaze_ratio`) and averages them across frames with
   a detected face.
4. The resulting baseline is stored in-memory, keyed by
   `(barter_id, user_id)`, in the same dict-of-session-state structure
   that already holds `buffers` — same lifecycle, same cleanup points.
5. Frontend proceeds to normal engagement streaming
   (`ws /video/{barter_id}/{user_id}`) as today. Every window's scoring
   call now looks up that user's stored baseline (defaulting to `0.5`/`0.5`
   — today's behavior — if no baseline is stored).
6. A "Recalibrate" control stays visible during the session, calling the
   same endpoint again; subject to a cooldown (see below).
7. Every calibration and recalibration event (success or failure) is
   POSTed to `backend` as an audit log record — new data, not used for
   live scoring, kept for post-session review and for the evaluation
   study.

No new persistent store backs live scoring; `backend`'s copy is
audit/evaluation-only.

## Scoring math

`sub_signals_from_mediapipe_landmarks` gains an optional `baseline`
parameter: `{"head_ratio": float, "gaze_ratio": float}`, defaulting to
`{"head_ratio": 0.5, "gaze_ratio": 0.5}` (today's fixed assumption,
producing bit-identical output when no calibration exists).

Both formulas change from "distance from the fixed midpoint 0.5" to
"distance from this user's measured baseline, normalized by the actual
maximum possible deviation from that baseline" (asymmetric, since a
baseline off-center has a shorter distance to one edge than the other):

```python
def _deviation_from_baseline(value: float, baseline: float) -> float:
    max_dev = max(baseline, 1.0 - baseline)
    if max_dev == 0:
        return 0.0
    return _clamp01(abs(value - baseline) / max_dev)

head_deviation = _deviation_from_baseline(ratio, baseline["head_ratio"])
gaze_centered = 1.0 - _deviation_from_baseline(avg_gaze_ratio, baseline["gaze_ratio"])
```

When `baseline == {0.5, 0.5}`, `_deviation_from_baseline` reduces
algebraically to the current `abs(x - 0.5) * 2.0` formula — this change is
strictly backward-compatible; the fitted fusion weights and attention
threshold in `docs/video_engagement/design-choices.md` remain valid
without re-fitting.

The AWS Rekognition path (`sub_signals_from_rekognition_face_detail`) is
**not** personalized in this design — Rekognition already returns
yaw/pitch in absolute degrees with no comparable raw-ratio primitive to
calibrate against a short sample cheaply; flagged as future work, not a
gap this sub-project closes.

## API surface

### `POST /video/{barter_id}/{user_id}/calibrate`

Request: multipart/batch of JPEG frames (same decode path as
`process_frame_local`).

Success response:
```json
{"calibrated": true, "baseline_head_ratio": 0.51, "baseline_gaze_ratio": 0.48,
 "frames_used": 4, "recalibration_count": 0}
```

Failure responses (fails open — session is never blocked):
```json
{"calibrated": false, "reason": "insufficient_face_detections"}
{"calibrated": false, "reason": "recalibration_limit_reached", "retry_after_seconds": 73}
```

Rules:
- Minimum 3 frames with a detected face out of whatever the frontend
  sends, or the call fails with `insufficient_face_detections` and the
  existing baseline (or the `0.5`/`0.5` default) is left untouched.
- First call per `(barter_id, user_id)` in a session always succeeds if
  it clears the minimum-detections bar — no cooldown on the *first*
  calibration.
- Every subsequent call (recalibration) is subject to a **cooldown of
  120 seconds** since the last successful calibration for that user in
  that session. A call inside the cooldown fails with
  `recalibration_limit_reached` and does not touch the stored baseline.
- On success, the endpoint overwrites the stored baseline and increments
  `recalibration_count`.
- Every call (success or failure) is POSTed to `backend` as an audit
  record: `{barter_id, user_id, timestamp, outcome, old_baseline,
  new_baseline, recalibration_count}`. This is fire-and-forget logging
  (best-effort POST, matching the existing `post_to_backend` retry
  pattern in `warning_engine`) — a logging failure never fails the
  calibration call itself.

### Frontend

- New calibration screen/step in `apps/frontend/src/screens`, shown once
  per user right after their camera stream starts, before/alongside
  session start. ~5s auto countdown, auto-submits, shows a brief
  "calibrated" / "skipped — using defaults" status, then proceeds.
- A small always-visible "Recalibrate" button during the session;
  disabled with a live countdown when within the 120s cooldown (driven
  by the `retry_after_seconds` the endpoint returns on the last
  cooldown-rejected or successful call).

### State lifecycle

Baseline entries are cleaned up at exactly the same points the existing
`buffers` dict entries are: the `WebSocketDisconnect`/`RuntimeError`
handler in `video_ws`, and `POST /session/{barter_id}/end`.

## Failure handling

- No face detected during calibration (camera off, bad lighting,
  face out of frame): fails open to the `0.5`/`0.5` default, session
  proceeds uncalibrated — identical to today's behavior. Never blocks
  session start.
- Frontend-side timeout: cap the calibration capture window at 6s
  wall-clock; if the countdown elapses without enough usable frames,
  submit what was captured (may still fail server-side per the 3-frame
  minimum, same fallback).
- Recalibration abuse (repeated calls to reset a drifting baseline
  favorably right before a warning): mitigated by the 120s cooldown and
  the full audit trail to `backend`, which makes recalibration timing
  visible in post-session review even though it isn't blocked outright.

## Evaluation plan

No new dataset collection is needed for the head-pose half of the claim;
a small honest pilot covers the gaze half.

1. **Head-pose personalization** (reuses already-licensed data — no
   synthesis): the Gourier/Crowley Head Pose Image Database
   (already vetted and used for `docs/video_engagement/design-choices.md`)
   has, per subject, a frontal (0°/0°) image plus many off-angle images.
   Treat each subject's frontal image as their calibration frame, compute
   their personal `baseline_head_ratio`, then compare `head_deviation`-
   based attentive/not classification accuracy on that subject's
   off-angle images **with** vs. **without** personalization. Held out
   by subject (never by row), matching the project's existing calibration
   methodology in `docs/threshold-calibration.md`.
2. **Gaze personalization**: no public dataset has per-subject neutral +
   off-gaze image pairs (the documented, still-unresolved gap). Plan: a
   small self-collected pilot — a handful of webcam clips across a few
   seating/lighting variations, manually labeled "looking at screen" vs.
   "looking away," comparing calibrated vs. uncalibrated `gaze_centered`
   accuracy. Explicitly reported in the paper as a small pilot, not a
   large-scale study.
3. **Metric**: balanced accuracy / AUROC, pre- vs. post-calibration, on
   both halves — same metric family `docs/threshold-calibration.md`
   already standardizes on.

## Testing

- Unit tests on `_deviation_from_baseline` and the updated
  `sub_signals_from_mediapipe_landmarks`: baseline `{0.5,0.5}` reproduces
  today's exact output (regression guard); off-center baselines produce
  the expected asymmetric normalization; edge cases (`baseline` at `0.0`
  or `1.0`).
- `video_engagement/tests/test_main.py`: new tests for
  `/video/{barter_id}/{user_id}/calibrate` — success path, insufficient-
  detections fallback, cooldown rejection and its `retry_after_seconds`,
  state cleanup on disconnect/end.
- Existing `test_scoring.py`, `test_weight_search.py` must continue to
  pass unmodified (backward-compatibility guarantee from the algebraic
  reduction above).

## What this touches

- `apps/video_engagement/scoring.py` — sub-signal formulas gain the
  `baseline` parameter.
- `apps/video_engagement/main.py` — new calibrate endpoint, baseline
  storage alongside `buffers`, cleanup wiring, audit POST to backend.
- `apps/backend/` — new endpoint `POST
  /session/{barter_id}/video-engagement/calibration-log` to receive and
  store the calibration audit log, alongside the existing
  `/session/{barter_id}/video-engagement` route family, plus a new table
  or column set to persist the audit records for post-session review and
  the evaluation study.
- `apps/frontend/src/screens` — new calibration step + recalibrate
  control.
- `apps/video_engagement/tests/`, plus whatever `apps/backend/tests/`
  covers the new audit-log route.
- `docs/video_engagement/design-choices.md` — should note the
  personalization layer sits on top of, not instead of, the fitted
  global weights/threshold.
