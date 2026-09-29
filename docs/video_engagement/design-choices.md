# Video Engagement — Weight Design Choices

Status: **sub-signal weights and the attention threshold are fitted values,
measured on public data.** Fitted 2026-09-26 on 2,295 labeled frames from two
openly-licensed public datasets and checked on 977 held-out frames from
subjects and shards the fit never saw.

The fusion weights that blend this video score with the speech engagement
score — `FUSION_WEIGHT_SPEECH`, `FUSION_WEIGHT_VIDEO`, and `FUSION_BIAS` in
`apps/warning_engine/main.py` — were **also** fitted, via logistic regression
on a synthetic corpus (`apps/warning_engine/ground_truth/`), rather than being
engineering guesses; see the "Fusion weight fitting" section of
[threshold-calibration.md](../threshold-calibration.md) for that fit's method
and numbers. That corpus is synthetic, not drawn from the production capture
distribution discussed above, so it is a separate open question from — and
does not close — the domain-gap caveats this document raises for the video
sub-signal weights themselves.

| Quantity | Value | Where it lives |
|---|---|---|
| `w_eyes`, `w_head`, `w_gaze` | **0.55, 0.40, 0.05** | `apps/video_engagement/scoring.py:DEFAULT_WEIGHTS`, overridable by `VIDEO_WEIGHT_EYES` / `VIDEO_WEIGHT_HEAD` / `VIDEO_WEIGHT_GAZE` (defaults set in `docker-compose.yml`) |
| Attentive decision threshold | **0.6123** on the fused score | `apps/video_engagement/scoring.py:ATTENTIVE_SCORE_THRESHOLD` |
| Full grid-search output | 231 weight triples | `docs/video_engagement/weight-search-results.json` |

`main.py` reads the env vars at import and falls back to `DEFAULT_WEIGHTS`, so
re-fitting means editing those two constants and the three
`docker-compose.yml` defaults — nothing else consumes the numbers.

## The two commands

Frames are never committed (`.gitignore` excludes
`apps/video_engagement/datasets/` and `apps/video_engagement/corpus/`), so
reproducing the fit means fetching the data first. From `apps/video_engagement`:

```bash
# 1. Fetch + label the public corpora (~30 MB + ~23 MB, a few minutes)
mkdir -p datasets/head_pose_image_database datasets/closed_open_eyes
curl -o datasets/head_pose_image_database/HeadPoseImageDatabase.tar.gz \
  http://crowley-coutaz.fr/HeadPoseDataSet/HeadPoseImageDatabase.tar.gz
tar xzf datasets/head_pose_image_database/HeadPoseImageDatabase.tar.gz \
  -C datasets/head_pose_image_database
for i in 001 002 003 004 005 900 995 996 997 998 999; do
  curl -L -o datasets/closed_open_eyes/dataset_$i.parquet \
    https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes/resolve/main/data/dataset_$i.parquet
done
venv/bin/pip install pyarrow          # converter-only dependency
venv/bin/python public_datasets.py head-pose datasets/head_pose_image_database
venv/bin/python public_datasets.py closed-open-eyes datasets/closed_open_eyes

# 2. Fit the weights and the threshold (~60s, MediaPipe over 3,440 frames)
venv/bin/python weight_search.py \
  --corpus datasets/head_pose_image_database datasets/closed_open_eyes \
  --step 0.05 --top 12
```

Step 2 prints the report below and rewrites
`docs/video_engagement/weight-search-results.json`.

## Data

Both datasets were verified fetchable and licence-compatible on **2026-09-26**
before use. Each is converted into the same labeled-frame corpus format
(JPEGs + a `labels.jsonl` manifest) by `public_datasets.py`, so the fitter has
one input format regardless of source.

### Split 1 — Head Pose Image Database (`head_pose_image_database`)

- **URL**: <http://crowley-coutaz.fr/Head%20Pose%20Image%20Database.html>,
  tarball `http://crowley-coutaz.fr/HeadPoseDataSet/HeadPoseImageDatabase.tar.gz`
  (28,512,828 bytes, sha256
  `5629cb428f38c65ba0220f480ea53ba859f56fbfc5a34e7396a981f0485ed209`).
- **Licence**: the dataset page states verbatim: *"This database can be used
  for any purpose, provided that the following article is cited: N. Gourier,
  D. Hall, J. L. Crowley, Estimating Face Orientation from Robust Detection of
  Salient Facial Features, Proceedings of Pointing 2004, ICPR, International
  Workshop on Visual Observation of Deictic Gestures, Cambridge, UK."* "Any
  purpose" covers this calibration; the citation obligation is discharged here
  and in `public_datasets.py`.
- **Downloaded**: 2026-09-26. **2,820 JPEGs**, 15 subjects, 2 series each,
  real photographs, pan/tilt encoded per filename.
- **After the rubric**: 2,340 labeled, 480 excluded (ambiguous band or
  unparseable name). 2,172 survived face detection and entered the experiment.
- **Speaks to**: `head_deviation`.
- **Cannot speak to**: `eyes_open` — every subject has their eyes open. Nor
  `gaze_centered` — the subjects rotate head *and* eyes together toward the
  target marker, so the iris stays centred in the socket even at pan 90°.

### Split 2 — MichalMlodawski/closed-open-eyes (`closed_open_eyes`)

- **URL**: <https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes>
- **Licence**: ODC-BY (Open Data Commons Attribution) as declared on the
  dataset card — open use with attribution.
- **Downloaded**: 2026-09-26. **11 of 1,000 parquet shards** (`001`–`005`,
  `900`, `995`–`999`) = **1,100 images**: 500 `closed_eyes`, 600 `open_eyes`.
  The full set is 126,560 images / 4.2 GB; a balanced sample was taken because
  the shards are class-homogeneous and the fit is not data-starved.
- **Speaks to**: `eyes_open`.
- **Cannot speak to**: `head_deviation` or `gaze_centered` — head pose is
  uncontrolled and unlabelled here.
- **Known weakness**: the dataset card tags this imagery `ai-generated`. It is
  photoreal but not photographic, which widens the domain gap discussed under
  *Known limitations*.

### What was rejected, and why

- **Columbia Gaze** — the only candidate with true head-forward/eyes-aside
  annotations. Its download is gated behind an e-mail form
  (`downloader.php` returns an HTML form, not the archive), so it could not be
  fetched non-interactively. This is why `gaze_centered` has no ground truth.
- **CEW / Closed Eyes In The Wild** — the NUAA host returns HTTP 403.
- **Head Pose Image Database's original INRIA/FGnet URLs** — HTTP 404; the
  author's own site (above) still serves it.
- **notgoodkeeper/cnn-based-drowsiness-detection-data** (MIT, fetchable) —
  rejected on substance, not licence: its labels were *generated by a
  MediaPipe EAR heuristic*. Fitting our MediaPipe EAR-based signal against it
  would measure agreement with another heuristic, not with reality.
- **Driver-drowsiness sets with no declared licence** — rejected: the brief
  requires a verified licence.

## Labels and targets

One label per frame. The target is what the fused score is asked to predict.

| Label | Target | Meaning | Present in |
|---|---|---|---|
| `attentive` | 1.0 | Face to camera, eyes open | both splits |
| `eyes_closed` | 0.0 | Face forward, eyes shut — breaks `eyes_open` alone | eye split |
| `looking_away` | 0.0 | Head turned — breaks `head_deviation` alone | head-pose split |
| `gaze_off` | 0.0 | Head forward, eyes to the side — breaks `gaze_centered` alone | **neither** (no fetchable data) |
| `absent` | *excluded* | No face in frame | webcam capture only |

`absent` is deliberately **not** scored 0.0. `video_attention_score` needs
landmarks and a faceless frame yields none, and production already skips
no-detection frames rather than treating them as zero attention
(`process_buffer` filters `None`). Scoring them 0.0 would fit a case the
service never evaluates. Those frames instead measure detector coverage: they
should produce *no* detection.

The inattentive labels are designed to be **one per sub-signal**, each
breaking exactly one signal while leaving the others attentive. Without that
structure the three weights are not identifiable — a corpus of plain
"attentive vs not" fits every weighting equally well.

### Head-pose rubric

Applied to the pan/tilt in each filename, using the larger of |tilt| and |pan|:

- **≤ 15° → `attentive`.** Someone genuinely watching a screen does not hold
  their head perfectly still.
- **≥ 45° → `looking_away`.**
- **15°–45° → excluded (480 frames).** A 30° turn could be a glance at a
  second monitor or a shift in the chair. No rubric calls that consistently,
  so those frames stay out of the fit rather than being pushed into whichever
  class is convenient.

### Eye-state mapping

`open_eyes → attentive` (1.0), `closed_eyes → eyes_closed` (0.0). Label noise
to be aware of: an `open_eyes` image may have an arbitrary head pose, so a low
fused score there can be a true head turn rather than an error.

## Objective function

Ranking is by, in order:

1. **Mean per-split ROC-AUC** — threshold-free ranking quality, averaged over
   splits so the larger dataset cannot outvote the smaller one on a sub-signal
   it cannot speak to. Pooling would have let 2,172 head-pose frames decide
   `w_eyes`.
2. **Minimum per-split balanced accuracy** at each split's own best threshold
   — rejects a triple that wins on average by abandoning one split.
3. **Class mean separation** — the margin, as a proxy for surviving a change
   of camera or lighting.

Pearson r against the 0/1 target is computed and reported but is deliberately
**not** the objective: with binary targets it rewards a spread-out score
distribution rather than a separable one. (The backend-correlation mode in
`weight_search.py` still uses Pearson r, because there the ground-truth proxy
is continuous.)

AUC, threshold selection and balanced accuracy are implemented in pure python
in `weight_search.py` — no sklearn dependency in this service.

### Plateau tie-break, and why `w_gaze` is 0.05

29 of the 231 triples land within 0.005 mean-AUC of the best. Taking the
sort's first row would award that plateau to floating-point noise, so the
choice is explicit (`choose_weights`): inside the plateau, keep
`w_gaze ≥ 0.05` (one grid step), then take the highest minimum per-split
balanced accuracy, then the lowest `w_gaze`, then the widest separation.

The floor and the "lowest `w_gaze`" rule pull against each other on purpose.
`gaze_centered` is the sub-signal **no fetched dataset can rank**, so it gets
the smallest weight the evidence permits. But a weight of exactly zero would
*delete* the signal from production — a stronger claim than "unmeasured",
asserting it is worthless including for the `gaze_off` failure mode the corpus
happens not to contain. Several plateau members do score `w_gaze = 0.00`
(0.50/0.50/0.00 tops the raw sort at mean AUC 0.9898); the floor is what keeps
the signal alive, and it costs 0.0004 mean AUC. Pass `--gaze-floor 0` to allow zero.

There is also evidence that this corpus *cannot* isolate gaze: a pure-gaze
weighting (0, 0, 1) scores AUC 0.9207 on the head-pose split. The
`gaze_centered` sub-signal is measurably contaminated by head rotation (the
eye-corner landmarks compress as the head turns), so head-pose data cannot
separate the two.

### Threshold definition

Predict **attentive when `fused score >= threshold`**. Candidates are the
midpoints between adjacent distinct scores plus one below the minimum and one
above the maximum, ranked by Youden's J (`tpr - fpr`, equivalently balanced
accuracy). Plain accuracy is not used: on an unbalanced corpus it can be won
by always predicting the larger class. Ties are broken by taking the median
tied candidate, so the threshold sits in the middle of the winning plateau
rather than hugging an observed score.

### Held-out evaluation

The fit is checked on groups it never saw — 4 of 15 head-pose **subjects**
(`person12`–`person15`) and 4 eye-state **shards**. Grouping by subject rather
than by frame is essential: consecutive frames of one person are
near-duplicates, and a random frame split would report memorisation. The
holdout is stratified by class, because the eye-state shards are
single-class and an unstratified split produced a positives-only holdout where
false-positive rate is undefined.

## Results (2026-09-26)

Corpus: 3,440 manifest frames → 3,272 with a detected face → 2,295 fit /
977 held out. Grid step 0.05 (231 triples).

Detector coverage: `attentive` 900/900, `eyes_closed` 500/500,
`looking_away` 1,872/2,040. **All 168 dropped frames are `looking_away`** —
MediaPipe loses the face at extreme pan/tilt. Those are the *easiest*
inattentive cases, so dropping them makes the reported task harder, not
easier. In production such a window yields no video sample at all.

### Top 12 of 231 triples (fit set)

| w_eyes | w_head | w_gaze | mean split AUC | min split bal.acc | pooled AUC | separation | threshold |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.50 | 0.00 | 0.9898 | 0.9604 | 0.9819 | 0.3995 | 0.5697 |
| 0.50 | 0.45 | 0.05 | 0.9898 | 0.9592 | 0.9814 | 0.3966 | 0.5631 |
| 0.60 | 0.35 | 0.05 | 0.9897 | 0.9575 | 0.9801 | 0.3694 | 0.6479 |
| 0.50 | 0.40 | 0.10 | 0.9897 | 0.9574 | 0.9809 | 0.3936 | 0.5976 |
| 0.60 | 0.40 | 0.00 | 0.9896 | 0.9612 | 0.9808 | 0.3723 | 0.6539 |
| 0.65 | 0.35 | 0.00 | 0.9895 | 0.9588 | 0.9764 | 0.3587 | 0.6988 |
| 0.60 | 0.30 | 0.10 | 0.9895 | 0.9587 | 0.9792 | 0.3664 | 0.6520 |
| 0.55 | 0.35 | 0.10 | 0.9895 | 0.9592 | 0.9799 | 0.3800 | 0.6346 |
| 0.50 | 0.35 | 0.15 | 0.9894 | 0.9550 | 0.9800 | 0.3907 | 0.6250 |
| **0.55** | **0.40** | **0.05** | **0.9894** | **0.9599** | **0.9805** | **0.3830** | **0.6123** |
| 0.65 | 0.30 | 0.05 | 0.9893 | 0.9582 | 0.9762 | 0.3558 | 0.6981 |
| 0.55 | 0.30 | 0.15 | 0.9892 | 0.9580 | 0.9789 | 0.3771 | 0.6503 |

All 231 rows, with per-split AUC for each, are in
`weight-search-results.json`.

### Chosen: (0.55, 0.40, 0.05), threshold 0.6123

Fit set (n=2,295): mean per-split AUC **0.9894**, pooled AUC 0.9805, pooled
balanced accuracy 0.9569 (tpr 0.9758, fpr 0.0621), Pearson r 0.7769.

Mean fused score per label (fit set):

| Label | Mean score |
|---|---|
| `attentive` | 0.8703 |
| `looking_away` | 0.5206 |
| `eyes_closed` | 0.3348 |

Per split, at the single pooled threshold (what production would apply):

| Split | n | AUC | tpr | fpr | balanced accuracy |
|---|---|---|---|---|---|
| `closed_open_eyes` | 700 | 0.9999 | 0.9675 | 0.0000 | 0.9838 |
| `head_pose_image_database` | 1,595 | 0.9789 | 0.9909 | 0.0756 | 0.9576 |

### Held out — the only numbers not optimised for

n=977 (`person12`–`person15`, shards `004`, `005`, `998`, `999`):

| Set | n | AUC | tpr | fpr | balanced accuracy | accuracy |
|---|---|---|---|---|---|---|
| **All held out** | 977 | **0.9886** | 0.9750 | 0.0516 | **0.9617** | 0.9560 |
| `closed_open_eyes` | 400 | 0.9967 | 0.9650 | 0.0100 | 0.9775 | — |
| `head_pose_image_database` | 577 | 0.9853 | 1.0000 | 0.0684 | 0.9658 | — |

Held-out mean scores: `attentive` 0.8758, `looking_away` 0.5298,
`eyes_closed` 0.3185. The holdout matches the fit set closely, which is the
evidence that the triple is not tuned to particular faces.

### For comparison

| Weighting | mean split AUC | Note |
|---|---|---|
| (0.55, 0.40, 0.05) | 0.9894 | chosen (rank 10; the top 29 are one plateau) |
| (0.40, 0.40, 0.20) | 0.9685 | **the previous placeholder** — rank 65 of 231 |
| (0.00, 1.00, 0.00) | 0.7286 | head only: AUC 0.5274 on the eye split (rank 231) |
| (0.00, 0.00, 1.00) | 0.7343 | gaze only: AUC 0.5480 on the eye split |
| (1.00, 0.00, 0.00) | 0.7367 | eyes only: AUC 0.4773 — below chance — on the head split |

The degenerate corners confirm the grid is measuring something real: each
single-signal weighting collapses on the split it cannot see.

## Where the threshold should be consumed

Nothing thresholds the score today. `video_engagement` posts the continuous
`video_attention_score` to the backend and to
`warning_engine/video-engagement/update`, which fuses it with the speech score.
`ATTENTIVE_SCORE_THRESHOLD = 0.6123` is the number to use wherever a boolean
"was this user attentive in this window" is needed — a per-window attentive
flag alongside the stored score, an attentive-fraction figure in the
post-session verdict, or a video-side escalation rule mirroring the
consecutive-off-topic logic in `warning_engine`. Any such consumer should read
the constant, not re-type 0.6123.

## Per-user calibration layer

Per-user calibration sits **on top of** the fitted global weights and threshold,
not instead of them. The fitted formula (`0.55 / 0.40 / 0.05`, threshold
`0.6123`) is unchanged; what changes is the neutral point two of its inputs are
measured against.

- **What is personalized.** `head_deviation` and `gaze_centered` were measured
  from a fixed neutral point of 0.5. Each user now has a baseline captured at
  the start of a session, and both sub-signals are measured from that instead.
  `eyes_open` is not personalized.
- **Normalization.** `_deviation_from_baseline` divides the deviation by
  `max(b, 1 - b)` for baseline `b`, so the result still spans 0..1 wherever the
  baseline sits.
- **No re-fit needed.** The default baseline (0.5) reproduces the old output
  exactly, so the fitted weights and `0.6123` remain valid as fitted.
  Personalization changes what "centered" means as an input to the same
  formula; it does not move the formula.
- **MediaPipe path only.** The Rekognition path is not personalized (see the
  local-vs-aws section below).
- **Lifecycle.** Baselines are in-memory per `(barter_id, user_id)`, cleared on
  WebSocket disconnect and on session end. Calibration needs at least 3 face
  frames, and recalibration has a 120 s cooldown. It fails open: if calibration
  cannot run, scoring continues with the default baseline.
- **Endpoint.** `POST /video/{barter_id}/{user_id}/calibrate` on
  `video_engagement`.
- **Audit log.** Every attempt, successful or not, is logged to the backend
  `calibration_logs` table via `POST` / `GET`
  `/session/{barter_id}/video-engagement/calibration-log`.

**Evaluation.** `apps/video_engagement/evaluate_calibration.py` compares
scoring with and without a per-user baseline:

```bash
venv/bin/python evaluate_calibration.py --corpus datasets/head_pose_image_database
```

It writes `evaluation-results.json` into the corpus directory by default.
**Results have not been produced yet**, so no numbers are reported here; this
section will cite them once the script has been run.

**What this does not fix.** The global public-dataset domain gap remains: the
weights and threshold were still fitted on studio-lit public imagery. Only the
per-session neutral point is personalized. A gaze pilot study (a user looking
away with the head forward) is future work; `w_gaze` remains a floor, not a
measurement.

## Known limitations

**Domain gap — the headline caveat.** Public dataset imagery does not match
this rig's webcam, lighting, framing, or compression. The Head Pose Image
Database is studio-lit against a neutral background; the eye-state split is
AI-generated. The chosen threshold is therefore a **starting point validated
on public data, not on the production capture distribution**. Treat 0.6123 as
a default to be re-checked, not a constant discovered about this product.

**Validating it on real frames** is what
`apps/video_engagement/collect_attention_frames.py` exists for — an *optional*
webcam capture tool, not the primary ground truth. It walks a user through the
five labeled states (including `gaze_off` and `absent`, which no public
dataset provided) and writes the same corpus format, so the same command fits
or checks it:

```bash
venv/bin/python collect_attention_frames.py          # writes corpus/
venv/bin/python weight_search.py --corpus corpus/    # or add it as a third split
```

Captured frames are personal data: `corpus/` is gitignored and must never be
committed.

**`gaze_centered` is unmeasured.** No fetchable, licence-clear dataset has
head-forward/eyes-aside annotations (Columbia Gaze is behind an e-mail form).
`w_gaze = 0.05` is a floor, not a measurement, and the corpus contains no
`gaze_off` frames at all. The webcam path above is the cheapest way to close
this gap. Per-user calibration (above) personalizes its neutral point but does
not give it ground truth.

**No `absent` frames in the public corpus**, so the no-detection path is
exercised only by the webcam tool and by unit tests.

**The two splits are not independent evidence for one threshold.** Each
speaks to one sub-signal; the pooled threshold is a compromise between two
distributions that would never co-occur in a single session.

## Known bias: local vs aws sample differently

`local` scores are the mean of sub-signals across every frame captured in
the ~5s window (`apps/video_engagement/main.py`, `process_buffer`'s
`local`/`both` branch). `aws` scores come from a single mid-window frame
only (`frames[len(frames) // 2]`), to limit Rekognition API call volume —
this is a deliberate cost tradeoff, not a bug.

Consequence: in `both` mode, `local` and `aws` are not directly comparable
per-window — a blink or head-turn that happens to land on the sampled
middle frame can swing the `aws` score for a window where the `local`
average stays stable, and vice versa. Any local-vs-cloud accuracy
comparison drawn from pilot data should account for this before treating a
per-window discrepancy as a backend-quality signal rather than a sampling
artifact.

Note also that the weights above were fitted on MediaPipe sub-signals only.
The Rekognition path derives `gaze_centered` from yaw alone (it has no iris
data), so the same weights are applied to a different, coarser signal there.

## Backend-correlation mode (unchanged, still unrun)

`weight_search.py --barter-id <id> --learner-user-id <id>` still implements
the original method from
`docs/superpowers/specs/2026-09-17-video-engagement-design.md`: correlate the
video score against the speech-based `engagement_score` for a pilot session.
It remains unrun — `video_engagement_results` and `engagement_score_log` are
both empty — and it is the weaker experiment anyway, since the speech score is
a proxy rather than a label of visual attention. It now writes to
`backend-correlation-results.md` rather than overwriting this file.
