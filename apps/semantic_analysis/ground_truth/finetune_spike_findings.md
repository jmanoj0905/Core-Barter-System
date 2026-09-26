# Spike: fine-tune SBERT vs stock all-MiniLM-L6-v2

Date: 2026-09-18

## Question

Does fine-tuning `all-MiniLM-L6-v2` on labeled (topic, text, score) pairs
improve correct/weakly_correct/incorrect separation vs the stock model
used in `main.py`?

## Method

- Expanded `topics.py`'s 6 hand-authored topics with 5 more (SQL joins,
  photosynthesis, compound interest, first aid for burns, negotiating a
  car price) — 11 topics total, ~10 labeled examples each.
- Leave-topic-out split: 7 topics for training, last 4 held out entirely
  for eval (generalization to unseen topics, not memorization).
- Fine-tuned with `CosineSimilarityLoss` (topic+scope vs example text,
  target score correct=1.0/weakly_correct=0.5/incorrect=0.0), 4 epochs.
- Evaluated both models on held-out topics using the same grid-search
  threshold calibration as `calibrate_thresholds.py`.

## Results

| Model | Best (upper, lower) | Accuracy |
|---|---|---|
| Stock all-MiniLM-L6-v2 | (0.48, 0.21) | 0.825 |
| Fine-tuned (4 epochs, 7 train topics) | (0.58, 0.28) | 0.800 |

Reconstructing the reported exact accuracies gives **40 evaluation examples**:
stock got 33/40 correct and fine-tuned got 32/40 correct. This is a net difference
of one correct prediction (2.5 percentage points), not evidence of a reliable
generalization regression. It does not establish that only one row's prediction
changed; individual predictions were not retained. The reported confusion shape
was the same: `incorrect` perfectly separated, with `weakly_correct` versus
`correct` the persistent failure mode (4-5 of 12 misclassified either way).

These scores also used thresholds selected on the evaluation rows themselves,
so they are calibration-set results rather than independent test performance.

## Why

- The original "28 held-out examples" was a sample-count typo: 40 is the only
  sample size from 20 through 60 compatible with both exact accuracies, matching
  four topics with ten examples each. A net one-correct-prediction difference on
  this calibration set cannot establish a generalization gain or regression.
- ~70 training pairs is far too little for a 22M-param model to learn
  general topic-adherence rather than overfit train-topic phrasing.
- The weakly_correct/correct boundary is inherently the fuzziest human
  judgment call in the label scheme — more data alone doesn't fix that
  without also tightening what counts as "weakly_correct".

## Recommendation

Don't fine-tune on hand-authored synthetic data at this scale — not worth
the complexity over the stock model + calibrated thresholds already in
place. Higher-leverage next step: `export_real_labels.py` once enough
real session transcripts accumulate — real phrasing diversity synthetic
data can't fake. A public tutoring-dialogue corpus (e.g. NCTE, TSCC)
could serve as a bigger pretrain-style dataset first, but that's a bigger
lift (licensing, cleaning, label mapping) — only worth it if pursuing the
fine-tune route seriously rather than probing it.

## Artifacts

Scripts used for this spike (`topics_expanded.py`, `finetune_probe.py`)
were throwaway, run from a scratch directory, not committed. Not
reproduced here — re-derivable from this doc if needed.
