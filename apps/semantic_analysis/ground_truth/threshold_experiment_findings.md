# Frozen SBERT threshold experiment

Date: 2026-09-19

Follow-up, 2026-09-26: the fixture was regenerated after two cleaning bug fixes
in `main.py` (punctuation-adjacent and multi-word filler matching). Its SHA-256
is now `00ffe940932ae4d154a6a7845ea13884c607f86b79c6cdb50231eaf4857a5540`, not
the `a4a8f8a9…` recorded below. **Every result in this document reproduced
exactly** — 28/60 current, 45/60 global, 42/60 per-contract, macro-F1 0.744 —
because only 1 of 60 rows changed similarity at all (`0.1586 → 0.1642`, a
trailing `well.` the old matcher missed). The conclusion here, that per-topic
thresholds are not supported, is unchanged and was acted on: they were not
adopted. The calibrated global pair *was* adopted, with `LOWER` set to 0.14
rather than the accuracy optimum 0.17 — see
[design_decisions.md](./design_decisions.md) (D1–D3) for that tradeoff.

## Result

The available fixture diagnostic does **not** support topic-specific thresholds
over global calibration. With each sentence held out, global thresholds correctly
classify 45/60 examples; per-contract upper thresholds classify 42/60. The latter
lose three previously correct predictions and gain none. This is a descriptive
result on six synthetic topics, not a demonstrated population-level difference.

No model was trained and no runtime classification threshold was changed.
The intended evaluation on independently human-labeled, production-like windows
could not run: the local `apps/backend/barter.db` contains zero windows and lacks
the `human_label` column; the evaluation transcript and annotation directories
contain no data. Family calibration is implemented and tested with controlled
inputs, but was not measured on the fixtures because they have no family mapping.

## Data and method

- Input: `synthetic_dataset.csv`, 60 single sentences across six topics;
  24 `correct`, 18 `weakly_correct`, and 18 `incorrect` labels.
- Input SHA-256: `a4a8f8a906b5397b75f4051588d9ca0e1c890a575e7e3bb4e5607b31695532b5`.
- Use the existing saved cosine scores from the stock SBERT fixture pipeline.
  No encoder recomputation, label edits, or relabeling from model predictions.
- Fit global upper/lower thresholds on calibration examples only, maximizing
  accuracy on a 0.01 grid across the cosine range. An upper cutoff of 1.01 is
  allowed to represent no predictions in the top class.
- For each `(topic, scope)` group, fit only its upper cutoff; hold the global
  lower cutoff fixed. Require at least two calibration examples in each of
  `correct` and `weakly_correct`; otherwise use the global thresholds. This
  support rule is not a claim of sufficient statistical power.
- Resolve tied calibration accuracies toward the current `(0.55, 0.35)` globally,
  and toward the fitted global pair within groups; remaining ties prefer lower
  cutoffs. These rules were fixed before running the diagnostic.
- **Sentence holdout:** 60 folds, each with 59 calibration examples and one
  evaluation example. This shares authors and topics, not independent sessions.
- **Topic holdout:** six folds, each with 50 calibration examples and the ten
  examples of an unseen topic for evaluation. Every group prediction uses the
  global fallback in this setting.

Full predictions, thresholds for each held-out example, confusion matrices,
per-topic scores, and configuration are in
[threshold_fixture_results.json](./threshold_fixture_results.json).

## Sentence-holdout diagnostic

| Method | Correct | Accuracy | Macro-F1 | Weakly-correct F1 | False-incorrect rate |
|---|---:|---:|---:|---:|---:|
| Current fixed `0.55 / 0.35` | 28/60 | 46.7% | 0.427 | 0.278 | 17/42 = 40.5% |
| Global calibration | 45/60 | 75.0% | 0.744 | 0.588 | 3/42 = 7.1% |
| Per-contract upper cutoff | 42/60 | 70.0% | 0.691 | 0.485 | 3/42 = 7.1% |

False-incorrect rate means predicting `incorrect` among examples labeled
`correct` or `weakly_correct`. It is a window-label metric, not a measured
rate of escalated warnings. Both calibrated variants recover 17/18 genuinely
`incorrect` examples, versus 18/18 for the current thresholds, so the reduction
in false off-topic labels comes with one missed off-topic example here.

Per-contract calibration changes three labels from right to wrong relative to
global calibration: one guitar example and two tax examples. All lower cutoffs
are identical within each global/grouped comparison. Consequently these three
changes affect only the `correct`/`weakly_correct` boundary, not off-topic labels.

## Topic-holdout diagnostic

Global calibration again gets 45/60, macro-F1 0.744 and weakly-correct F1 0.588.
The grouped method is identical because all 60 predictions fall back to global
thresholds. This verifies unseen-topic behavior; it is not evidence that a
topic-specific model generalized to an unseen topic.

## Interpretation

Global recalibration is worth evaluating on real windows. This small diagnostic
gives no reason to add per-topic upper thresholds to the live service. It also
does not establish that fine-tuning is necessary: ambiguous fixture labels,
only six topics, and the sentence-to-window mismatch remain unresolved.

The folds share calibration data, so do not treat them as independent repetitions
or interpret these 60 rows as 60 independent topics. There is no power claim,
significance claim, or confidence interval attached to these exploratory scores.

The next required input is reviewed session-window data. Preserve the source
session for every window and derivative; reserve separate sessions for calibration
and evaluation, plus unseen families for a separate generalization evaluation.
Do not manufacture independent sessions by concatenating fixture sentences or
use an LLM's suggested labels as the gold standard.

## Reproduce the diagnostic

From the repository root, using Python's standard library:

```bash
python3 apps/semantic_analysis/ground_truth/threshold_experiment.py \
  --fixture-diagnostic apps/semantic_analysis/ground_truth/synthetic_dataset.csv \
  --output apps/semantic_analysis/ground_truth/threshold_fixture_results.json
```

The CSV is ignored by git. If it is unavailable, regenerate it with
`generate_synthetic.py` in an environment with the semantic service dependencies.
Regenerated cosine scores may differ; the report records the input hash.

## Run on independently labeled windows

Export feedback from a database with labeled windows:

```bash
python3 apps/semantic_analysis/ground_truth/export_real_labels.py path/to/barter.db real_dataset.csv
```

The export retains `session_id` and `window_id`, selects human rather than model
labels, and excludes `out_of_scope`, which is not part of this three-class task.
It reads the database without modifying it. Verify that windows reflect the
25-second accumulated-teacher-speech pipeline and that their scores come from the
same encoder/preprocessing version; the current database does not record enough
provenance to establish those facts automatically.

Assign source sessions to calibration or evaluation **before** inspecting model
results. Keep variants of a source session in its original split. Save CSVs with
`topic,scope,text,expected_label,similarity,session_id`; export's extra columns
are accepted. Then run:

```bash
python3 apps/semantic_analysis/ground_truth/threshold_experiment.py \
  --calibration calibration.csv \
  --evaluation evaluation.csv \
  --output /private/tmp/threshold-real-results.json
```

The runner rejects shared session IDs, repeated contract/text examples, missing
labels or session IDs, and non-finite/out-of-range scores. It cannot detect every
semantic paraphrase or establish that a label was independently reviewed.

For family thresholds, add a `family` column assigned from contract metadata
before looking at results and pass `--group-by family`. Known-family sessions
and entirely unseen families should be evaluated in separate invocations. The
global fallback applies to unknown groups and groups lacking calibration support.

Reports contain per-row predictions and contract metadata. Keep real-data reports
local; the checked-in-style diagnostic artifact contains only synthetic data.

## Verification

```bash
python3 -m pytest apps/semantic_analysis/ground_truth/test_threshold_experiment.py \
  apps/semantic_analysis/ground_truth/test_export_real_labels.py -q
```

Tests cover evaluation-label isolation, shared-session/duplicate rejection,
unseen/sparse-group fallback, family transfer, fixed lower cutoffs, class metrics,
negative cosine scores, fixture-only diagnostics, and feedback export provenance.
