# Video Engagement Score — Design

Status: draft, pending user review
Date: 2026-09-17

## Purpose

Add a video-based engagement signal to barter sessions, computed two ways
(local on-device vs cloud API), so the two can be compared for accuracy.
Multi-modal: the video signal and the existing audio/speech-based
engagement signal stay two independent, single-purpose services; they
are combined only in `warning_engine`.

**Existing context this design must respect** (discovered while writing
the implementation plan, not present when the architecture section
above was first drafted): `semantic_analysis` already computes a
speech-based `engagement_score` for the session's *learner* only
(speaking-ratio + question-rate + acknowledgment heuristics, see
`calculate_engagement_score()`), and already POSTs low-score alerts to
`warning_engine` via `POST /engagement/alert`. `WindowResult` (topic
adherence) has no `user_id` column — it's session-level, not per-user —
so it cannot serve as a per-user fusion partner. The design below fuses
with the existing per-learner speech `engagement_score` instead, and
does so in `warning_engine`, not inside `semantic_analysis` — each
scoring service stays untouched and single-purpose; only the combine
step is new.

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
3. Correlate (Pearson) against the *existing* speech-based
   `engagement_score` for that session's learner (the value
   `semantic_analysis` already computes via `calculate_engagement_score`)
   as the ground-truth proxy — checking whether video-derived attention
   agrees with the independently-computed speech signal. Topic
   adherence (`WindowResult`) is not used: it has no `user_id`, so it
   cannot stand in for a per-learner signal.
4. Pick the argmax-correlation weight set.
5. Write every combo tried + its correlation + the chosen combo and
   rationale to `docs/video_engagement/design-choices.md`.

Chosen weights become the default in `video_engagement/main.py`,
overridable via env (`VIDEO_WEIGHT_EYES`, `VIDEO_WEIGHT_HEAD`,
`VIDEO_WEIGHT_GAZE`) for further experiments.

**Bootstrap problem**: weight search needs real windows (raw signals +
the corresponding speech `engagement_score`) to run against. Until a
pilot session produces that data, ship with a placeholder set
(0.4 / 0.4 / 0.2) explicitly marked in `design-choices.md` as
"placeholder, pending weight_search run" — not presented as a
considered choice.

### Fusion weight (video vs speech)

A second, separate weight controls how much the video score influences
the final combined engagement number:

```
fused_engagement_score = w_speech * speech_engagement_score + w_video * video_attention_score
```

Picked the same experimental way — swept and compared, not hardcoded —
default placeholder `w_speech=0.7, w_video=0.3` (speech signal is
already tuned/trusted; video starts as a minority signal until proven).
Documented in the same `design-choices.md`.

## Data flow / storage

Two independent producers, one combiner:

- **`semantic_analysis`** (unchanged internals): its existing
  `calculate_engagement_score()` logic is not touched. The only change
  is additive — every time it recomputes the learner's score, it now
  also `POST`s the current value to `warning_engine:8003/engagement/update`
  (a new endpoint, distinct from the existing alert-only
  `/engagement/alert`), not just when the score is low. This gives
  `warning_engine` the live speech score, not only alerts.
- **`video_engagement`** (new service): POSTs each window's result to
  two places:
  - `backend:8000/session/{id}/video-engagement` → new
    `VideoEngagementResults` table: `barter_id`, `user_id`,
    `window_start`, `window_end`, `video_attention_score`,
    `backend_used`, `raw_signals` (JSON: eyes_open, head_deviation,
    gaze_centered). Storage/exposition only — no fusion here.
  - `warning_engine:8003/video-engagement/update` (new endpoint) with
    the same score, keyed by `(barter_id, user_id)`.
- **`warning_engine`** (new combine step, everything else in it
  unchanged): holds the latest `speech_engagement_score` and
  `video_attention_score` per `(barter_id, learner_user_id)` — the
  video score is only used for fusion when its `user_id` matches that
  session's `learner_user_id` (fetched once via the same
  `SessionInitRequest`/contract data warning_engine already has for
  topic-adherence decisions; a video score for the teacher is still
  stored for later exposition but not fused). Computes
  `fused_engagement_score` per the formula above and runs the
  *existing* low-engagement-alert logic off the fused number instead of
  the raw speech-only one. If no video score has arrived yet for that
  session, fused = speech-only — behavior is identical to today until
  video data exists (backward compatible, no regression).
  Exposed via new `GET /session/{id}/engagement`. **Does not change
  the topic-adherence escalation levels** (1/2/3+ off-topic warnings) —
  those stay driven by `WindowResult` only, as today. This is a
  parallel, additive signal.

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

- Changing warning_engine escalation levels (topic-adherence warnings)
  based on engagement score.
- Manual human-labeled ground truth dataset for engagement (proxy via
  the existing speech-based `engagement_score` used instead).
- Post-session/recorded-video analysis (live webcam only).
- Any change to `semantic_analysis`'s internal engagement calculation —
  it stays exactly as-is; only a new outbound POST is added.
