# Video Engagement Score — Design

Status: draft, pending user review
Date: 2026-09-17

## Purpose

Add a video-based engagement signal to barter sessions, computed two ways
(local on-device vs cloud API), so the two can be compared for accuracy.
Multi-modal: video signal fuses with the existing audio/topic-adherence
signal in `warning_engine`.

## Architecture

New service `apps/video_engagement/` (port 8004), same shape as
`apps/audio_pipeline/`:

- FastAPI + WebSocket ingest: `ws /video/{barter_id}/{user_id}`, client
  streams JPEG frames.
- In-memory per-`(barter_id, user_id)` frame buffer, ~5s windows, aligned
  with audio_pipeline's `BUFFER_THRESHOLD_SECONDS`.
- Runtime-switchable backend via `VIDEO_BACKEND` env (`local` | `aws` |
  `both`) + `GET/POST /video/config` (mirrors audio_pipeline's
  `/stt/config`).
- `POST /session/{barter_id}/end` flushes all buffers for that session
  (mirrors audio_pipeline's `end_session`).

### Backends

- **local**: MediaPipe Face Mesh. ~5 sampled frames/window (1fps). Per
  frame: eye-aspect-ratio (openness/blink), gaze offset (iris vs eye
  corner), head pose (yaw/pitch from landmarks). Averaged across the
  window's detected-face frames.
- **aws**: boto3 Rekognition `detect_faces(Attributes=['ALL'])`, 1
  mid-window frame (cost control). Maps `EyesOpen`, `Pose`,
  `Smile`/`Emotions` into the same sub-signal shape as local.
- **both**: runs local on all sampled frames AND Rekognition on 1
  frame/window, in the same session, so scores are paired for direct
  comparison rather than compared across separate runs.

Video service is video-only and single-purpose — no cross-service
fusion happens inside it, matching the existing services' one-job
pattern (audio_pipeline does STT only, semantic_analysis does topic
scoring only).

## Score formula — experimental, weights not fixed

```
video_attention_score = w1*eyes_open + w2*(1 - head_deviation) + w3*gaze_centered
```

All sub-signals normalized 0-1. Same formula shape for both backends —
only sub-signal extraction differs. `w1 + w2 + w3 = 1`.

**Weights are not hardcoded up front.** Every window's raw sub-signals
(`eyes_open`, `head_deviation`, `gaze_centered`) are stored alongside
the final score, not just the final score — this lets weight search run
offline without re-running any video model.

### Weight search

`apps/video_engagement/weight_search.py` (offline script):

1. Grid-sweep weight triples summing to 1 (e.g. step 0.1 → ~66 combos).
2. For each combo, recompute `video_attention_score` per stored window
   from the raw sub-signals.
3. Correlate (Pearson) against that window's existing topic-adherence
   label from `WindowResults` (correct=1, weakly_correct=0.5,
   incorrect=0) — used as a proxy ground truth for engagement, since no
   manual labeling pass exists yet.
4. Pick the argmax-correlation weight set.
5. Write every combo tried + its correlation + the chosen combo and
   rationale to `docs/video_engagement/design-choices.md`.

Chosen weights become the default in `video_engagement/main.py`,
overridable via env (`VIDEO_WEIGHT_EYES`, `VIDEO_WEIGHT_HEAD`,
`VIDEO_WEIGHT_GAZE`) for further experiments.

**Bootstrap problem**: weight search needs real windows (raw signals +
topic labels) to run against. Until a pilot session produces that data,
ship with a placeholder set (0.4 / 0.4 / 0.2) explicitly marked in
`design-choices.md` as "placeholder, pending weight_search run" — not
presented as a considered choice.

## Data flow / storage

- `video_engagement` → `POST backend:8000/session/{id}/video-engagement`
  → new `VideoEngagementResults` table: `barter_id`, `user_id`,
  `window_start`, `window_end`, `video_attention_score`, `backend_used`,
  `raw_signals` (JSON: eyes_open, head_deviation, gaze_centered).
- `video_engagement` → `POST warning_engine:8003/engagement/update` with
  the same payload. `warning_engine` looks up that window's existing
  topic-adherence classification (it already has `WindowResults`
  access) and fuses:

  ```
  engagement_score = 0.6*video_attention_score + 0.4*topic_adherence_score
  ```

  Stored, exposed via `GET /session/{id}/engagement`. **Does not affect
  escalation logic** (1/2/3+ off-topic warning levels) — additive new
  signal only. Wiring it into escalation thresholds is an explicit
  future decision, out of scope here.

## Error handling

- No face detected in a frame: skip that frame, don't zero the score —
  average only over frames that did detect a face. If zero faces
  detected across the whole window, omit the window (no score posted),
  log a warning (mirrors audio_pipeline's empty-transcript skip).
- Rekognition call fails or is throttled: fail-open like
  `check_toxicity` — log, skip cloud score for that window; in `both`
  mode, local score still posts.
- WebSocket disconnects: flush partial buffer on disconnect (mirrors
  audio_pipeline's `WebSocketDisconnect` handling).

## Testing

- Unit: score-formula math — given synthetic landmark/Rekognition JSON,
  assert expected sub-scores and final score. Pure functions, no I/O.
- Integration: feed a short sample clip with known frames (eyes closed,
  eyes open, looking away) through both backends, assert relative score
  ordering (sanity check, not exact match — Rekognition is a live API).
- No dedicated eval-dataset build in this pass; reuse an existing/quick
  sample clip. A proper eval harness (like the existing audio
  eval-dataset tooling) is a possible follow-up, not built here.

## Out of scope (explicitly)

- Changing warning_engine escalation levels based on engagement score.
- Manual human-labeled ground truth dataset for engagement (proxy via
  topic-adherence used instead).
- Post-session/recorded-video analysis (live webcam only).
