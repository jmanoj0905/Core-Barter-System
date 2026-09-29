# Fitting the Speech/Video Engagement Fusion Weights — Design

Status: draft, pending user review
Date: 2026-09-29

## Purpose

This is sub-project 3 of 4 research-driven modifications to the
during-session QA/safety pipeline, targeting eventual publication (SMARTCOM
Goa 2027). Source research: see
`3_Capstone/during-session-qa-safety-novelty-research.md` (Obsidian vault),
Cluster 3. Sub-projects 1 (gaze calibration) and 2 (meaning-reversal check)
are spec'd/planned separately and out of scope here.

`apps/warning_engine/main.py`'s `_recompute_and_log_fusion` blends a
speech-engagement score (from `semantic_analysis`) and the video-attention
score (from `video_engagement`) with `ENGAGEMENT_FUSION_W_SPEECH=0.7` /
`ENGAGEMENT_FUSION_W_VIDEO=0.3` — documented in this project's own
`docs/threshold-calibration.md` as an **unfitted engineering guess**, with
no paired speech+video ground truth to fit against. This design replaces
the fixed linear blend with a fitted logistic regression, following the
same "fit offline, hardcode the fitted constants" house style already used
for `video_engagement`'s sub-signal weights (`weight_search.py` →
`DEFAULT_WEIGHTS` in `scoring.py`).

## Honest scope statement (read this before the rest)

**No real paired session data exists.** This design does not manufacture
that data or pretend to. It delivers two distinct, separately-true things:

1. **A real methodological upgrade, fit-on-synthetic-data only:** a
   logistic regression replacing the fixed scalar blend, fit on a
   synthesized labeled corpus. This is a genuine improvement over a
   hand-set guess (it is now *a fitted model*, with reported precision/
   recall/AUC), but its validity is bounded by the synthetic generator —
   the evaluation section states this plainly and does not claim
   real-world predictive validity.
2. **A semi-supervised extension pipeline that cannot yet run for real,**
   inspired by Chang et al. (arXiv:2506.13971), wired to pull from
   `backend`'s already-existing `engagement_score_log` /
   `video_engagement_results` tables. It is built and tested against
   synthetic "unlabeled" data now, and is ready to run against real
   session logs the moment they accumulate. The paper reports this as a
   validated pipeline awaiting real data, not as a completed result.

Chang et al.'s own method operates over raw audio/facial/text streams with
modality-specific encoders — a much higher-dimensional setting than this
project's two scalar features. This design does not reproduce their
architecture; it adapts their core idea (use unlabeled data to reduce
labeled-data need) via **self-training/pseudo-labeling**, a standard,
much simpler semi-supervised technique suited to a 2-feature, 3-parameter
model. This adaptation is stated explicitly, not glossed over.

## Non-goals

- No GateFusion-style (arXiv:2512.15707) multi-layer gated architecture —
  explicitly rejected in the research note as needing intermediate
  features from both modality encoders that neither `semantic_analysis`
  nor `video_engagement` currently expose.
- No new runtime model-serving dependency — the fitted model is 3 floats
  (`a`, `b`, `bias`) hardcoded into `warning_engine`, exactly like
  `video_engagement`'s fitted weights, not a loaded model file.
- No change to the fallback behavior when only one of the two scores is
  present (today: use the raw single score directly) — the fitted model
  only replaces the two-score fusion branch.
- No new backend endpoints — the semi-supervised script pulls per-session
  data via the existing `GET /session/{barter_id}/engagement/history`
  route, looped over known session IDs.

## Architecture

### Model

Replace:
```python
fused = ENGAGEMENT_FUSION_W_SPEECH * speech + ENGAGEMENT_FUSION_W_VIDEO * video
```
with:
```python
fused = _sigmoid(FUSION_WEIGHT_SPEECH * speech + FUSION_WEIGHT_VIDEO * video + FUSION_BIAS)
```
`_sigmoid(x) = 1 / (1 + math.exp(-x))`. The sigmoid's output is already in
`[0, 1]`, so `fused_engagement_score`'s existing contract (a continuous
score in that range, logged and consumed exactly as today) is unchanged —
this is a drop-in replacement, not a new field or new type.

`FUSION_WEIGHT_SPEECH`, `FUSION_WEIGHT_VIDEO`, `FUSION_BIAS` are module
constants in `warning_engine/main.py`, overridable by env vars (mirroring
`video_engagement`'s `VIDEO_WEIGHT_*` pattern), with defaults set to the
values fit in the Evaluation section below. `docker-compose.yml`'s
`ENGAGEMENT_FUSION_W_SPEECH`/`ENGAGEMENT_FUSION_W_VIDEO` env vars are
retired and replaced with `FUSION_WEIGHT_SPEECH`/`FUSION_WEIGHT_VIDEO`/
`FUSION_BIAS` — this is a capstone project with no external consumers of
the old env var names, so a rename carries no compatibility burden.

### Fitting (offline, standard library only)

New directory `apps/warning_engine/ground_truth/` (mirrors
`semantic_analysis/ground_truth/`'s existing convention; `warning_engine`
has no such directory yet):

- `synthesize_fusion_corpus.py` — generates the labeled synthetic corpus
  (see Synthetic data below), writing `fusion_corpus.csv` with columns
  `scenario_id, speech_score, video_score, engaged_label`.
- `fit_fusion_weights.py` — loads `fusion_corpus.csv`, fits `a`, `b`,
  `bias` by gradient descent on binary cross-entropy loss (pure Python,
  no numpy — `warning_engine/requirements.txt` has none today and this
  does not add one, matching `semantic_analysis/ground_truth/
  threshold_experiment.py`'s existing "standard library only" precedent),
  holds out by `scenario_id` (never by row — same discipline as
  `docs/threshold-calibration.md`'s shared method), and prints/writes the
  fitted constants plus held-out AUC/accuracy compared against the
  current fixed `0.7`/`0.3` blend (thresholded at 0.5) as a baseline.

### Synthetic data

`synthesize_fusion_corpus.py`'s generative rule, stated precisely so a
codex agent (or a human) can produce it deterministically from a fixed
seed:

- Each `scenario_id` has a true latent engagement level drawn from
  `{high: 0.85, medium: 0.5, low: 0.15}`.
- `speech_score` and `video_score` are each the true level plus
  independent Gaussian noise (`sigma = 0.1`, clamped to `[0, 1]`) —
  modeling two independently-noisy sensors of the same underlying state.
- `engaged_label = 1` if the true latent level `>= 0.5`, else `0`.
- **Deliberate disagreement cases** (at least 20% of scenarios): the two
  scores are generated from *different* latent levels (e.g. speech from
  `high`, video from `low` — "talking a lot while visibly distracted") to
  give the fitted model, and the evaluation, something to say about how
  it handles modality disagreement, which the fixed linear blend cannot
  express any differently than agreement cases at the same weighted sum.
- Target at least 200 scenarios (a few hundred rows), generated by a
  small codex agent per the project's established approach for missing
  datasets, following this exact rule so the corpus is reproducible and
  its generative assumptions are auditable in the paper.

### Semi-supervised extension (built, not yet run for real)

`apps/warning_engine/ground_truth/semi_supervised_fusion_fit.py`:

1. Train the initial logistic model on the labeled synthetic corpus
   (reuses `fit_fusion_weights.py`'s fitting function).
2. Fetch unlabeled `(speech_score, video_score)` pairs by looping over a
   provided list of `barter_id`s and calling the existing
   `GET /session/{barter_id}/engagement/history` backend route (script
   accepts `--backend-url` and `--barter-ids`, or `--offline-corpus` to
   run against a locally synthesized "unlabeled" stand-in corpus for
   testing, since no real session IDs exist yet).
3. Self-training loop: predict pseudo-labels for the unlabeled pairs,
   keep only predictions with confidence above a stated threshold (e.g.
   `|p - 0.5| >= 0.3`), add them to the training set, refit, repeat for a
   fixed small number of rounds (e.g. 3) or until no new confident
   pseudo-labels are added.
4. Report the same held-out AUC/accuracy metric as `fit_fusion_weights.py`,
   on the same held-out synthetic split, so the two approaches are
   directly comparable in the paper.

This script is tested against `--offline-corpus` (synthetic data standing
in for "real but unlabeled") in this sub-project. Running it against
`--backend-url` with real `barter_id`s is future work, gated on real
sessions existing — the script's `--backend-url` path is exercised in
tests only via a mocked HTTP client, not against a live backend with real
data.

## Testing

- Unit tests on `_sigmoid`, the gradient-descent fitting function (fits a
  trivially-separable toy dataset correctly), and the held-out-by-
  `scenario_id` split logic (never leaks a scenario's rows across the
  split).
- `apps/warning_engine` test: `_recompute_and_log_fusion` uses the sigmoid
  form with the new constants and produces a value in `[0, 1]`; the
  single-score fallback path (only speech or only video present) is
  unchanged from today's behavior (regression guard).
- `synthesize_fusion_corpus.py`: a fixed seed produces a deterministic,
  reproducible corpus (regression guard so results are re-runnable).
- `semi_supervised_fusion_fit.py`: run end-to-end against
  `--offline-corpus`, confirm it terminates within the fixed round limit
  and reports a metric; the `--backend-url` fetch path is tested with a
  mocked HTTP client returning a small fixture history.

## What this touches

- `apps/warning_engine/main.py` — `_recompute_and_log_fusion`'s formula,
  new constants `FUSION_WEIGHT_SPEECH`/`FUSION_WEIGHT_VIDEO`/
  `FUSION_BIAS` replacing `ENGAGEMENT_FUSION_W_SPEECH`/
  `ENGAGEMENT_FUSION_W_VIDEO`.
- `apps/warning_engine/ground_truth/` — new directory: corpus generator,
  fitting script, semi-supervised script, and their tests.
- `docker-compose.yml` — env var rename.
- `docs/threshold-calibration.md` — add this fusion fit as a third row in
  the calibrated-thresholds table, explicitly marked "fit on synthetic
  data only" in the same honest-caveat style as the existing rows'
  stated limits.
- Tests in `apps/warning_engine/tests/`.
