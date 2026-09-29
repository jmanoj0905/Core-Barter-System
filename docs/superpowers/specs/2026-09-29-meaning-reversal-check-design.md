# Meaning-Reversal Check Alongside the Cosine Gate — Design

Status: draft, pending user review
Date: 2026-09-29

## Purpose

This is sub-project 2 of 4 research-driven modifications to the
during-session QA/safety pipeline, targeting eventual publication (SMARTCOM
Goa 2027). Source research: see
`3_Capstone/during-session-qa-safety-novelty-research.md` (Obsidian vault),
Cluster 2. Sub-project 1 (per-user gaze calibration) is spec'd separately at
`docs/superpowers/specs/2026-09-29-gaze-calibration-design.md` and is
unrelated to this one.

`apps/semantic_analysis`'s topic-adherence gate (`classify()` in
`windowing.py`, thresholds `UPPER=0.36`/`LOWER=0.14`) measures only cosine
similarity between a window's embedding and the topic embedding. This is
architecturally the same failure mode Frias (arXiv:2608.10216) audits: a
fixed cosine-threshold gate approving semantically reversed content because
cosine similarity is topically "loose" — it does not encode polarity or
negation (You, arXiv:2504.16318, explains the mechanism: normalization and
anisotropy discard exactly this kind of signal). Concretely: a barter party
could say "I will not pay the agreed amount" and it would score `correct`
or `weakly_correct` against a topic like "payment terms," because the
sentence is highly topical — it is just also a reversal of the commitment
being negotiated.

## Explicit non-goals / relationship to other in-flight work

- This is **not** the analogy-vs-drift / escrow-adjudication work described
  in `docs/superpowers/specs/2026-09-27-analogy-vs-drift-handoff.md` and
  `2026-09-28-analogy-adjudication-design.md`. Those are unapproved,
  unimplemented, and address a different failure mode (sustained
  off-topic-looking pedagogical vehicles vs. genuine drift). This design
  does not modify `consecutive_incorrect`, the warning ladder, or any
  escrow/verdict logic, and must remain valid regardless of how that
  separate work resolves.
- No NLI/entailment model in this scope (Option B from the earlier
  discussion) — documented as a future extension, not built here.
- No change to `UPPER`/`LOWER` thresholds or the cosine-similarity
  computation itself.

## Architecture & detection rule

New module `apps/semantic_analysis/reversal_detection.py`, matching
`windowing.py`'s existing lexicon-constant style:

- `COMMITMENT_PHRASES`: barter-domain commitment verbs/phrases — `"pay"`,
  `"give"`, `"provide"`, `"deliver"`, `"send"`, `"transfer"`, `"return"`,
  `"refund"`, `"owe"`, `"accept"`, `"honor"`, `"complete"`, `"fulfill"`,
  `"commit to"`, `"promise"`, `"agree to"`, `"guarantee"`.
- `NEGATION_CUES`: `"not"`, `"never"`, `"no longer"`, `"won't"`, `"can't"`,
  `"cannot"`, `"wouldn't"`, `"shouldn't"`, `"couldn't"`, `"refuse to"`,
  `"refused to"`, `"unable to"`.
- `NEGATION_PROXIMITY_TOKENS = 4`: a negation cue and a commitment phrase
  are considered linked if they occur within this many tokens of each
  other in the same sentence. This is a judgment-call constant, stated
  here and tuned/reported during evaluation — same spirit as
  `windowing.py`'s `MIN_CONTENT_TOKENS = 3`.
- `detect_meaning_reversal(cleaned_text: str) -> bool`: split `cleaned_text`
  into sentences (on `.`, `!`, `?`), tokenize each sentence, and return
  `True` if any sentence contains both a negation cue token/phrase and a
  commitment phrase within `NEGATION_PROXIMITY_TOKENS` tokens of each
  other. Multi-word phrases in both lexicons are matched the same way
  `windowing.py`'s `FILLER_PHRASES`/`_FILLER_PHRASE_RE` already handles
  multi-word filler entries — reuse that pattern rather than inventing a
  second one.

**Known, stated limitation:** naive proximity-based negation detection
cannot handle double negation ("I will definitely not renege on my promise
to pay") — a negation cue near a commitment-adjacent word can misfire in
either direction. This is documented, not hidden, matching this codebase's
existing practice of stating what a calibration does not cover (see
`docs/threshold-calibration.md`'s "What is not calibrated" section). The
evaluation corpus (below) includes such cases specifically to demonstrate
and quantify this limitation, not to pass it.

## Integration point

In `apps/semantic_analysis/main.py`'s `process_window` (currently lines
323-359): immediately after `classification = classify(similarity)`
(line 348), call:

```python
reversal_detected = detect_meaning_reversal(cleaned)
if reversal_detected:
    classification = "incorrect"
```

before `_window(...)` and `post_window_result(...)` are called. This
reuses the existing `consecutive_incorrect` escalation path in
`warning_engine` exactly as-is — **no changes to `warning_engine`'s
escalation logic are needed or made.**

## Visibility (audit trail, not just override)

A reversal-forced `incorrect` must be distinguishable from a genuine
low-cosine `incorrect`, both for post-session review and for the paper's
evaluation. Add `meaning_reversal_detected: bool` to:

1. `post_window_result`'s payload in `semantic_analysis/main.py`.
2. `WindowResultRequest` in `warning_engine/main.py` (accepted but not
   acted on by escalation logic — purely passed through to `backend`).
3. A new nullable column `meaning_reversal_detected` on `backend`'s
   `window_results` table, added via the existing `_add_missing_columns`
   migration pattern in `apps/backend/app/database.py` (the same
   mechanism that added `human_label` and `warnings.advisory`), and
   surfaced in `GET` responses for that table.

## Evaluation plan

Mirrors the existing `apps/semantic_analysis/ground_truth/generate_synthetic.py`
+ `topics.py` pattern. A small codex agent generates a labeled corpus with
four categories, across a handful of the existing `topics.py` topics:

1. **Genuine commitment sentences** (on-topic, no negation) — expected
   `meaning_reversal_detected = False`.
2. **Meaning-reversed counterparts** of (1) — expected
   `meaning_reversal_detected = True`.
3. **Distractor sentences** with unrelated negations (negation present,
   but not near any commitment phrase, or negating something unrelated
   to the barter commitment) — expected `False` (false-positive check).
4. **Double-negation edge cases** — expected to demonstrate the stated
   limitation, not silently pass; results here are reported as a known
   failure mode, not tuned away.

**Metric:** precision/recall of `meaning_reversal_detected` against these
labels, reported the same way `docs/threshold-calibration.md` reports the
existing cosine thresholds (a results table + explicit statement of what
is and isn't covered).

## Testing

- Unit tests on `detect_meaning_reversal`: each lexicon category from the
  evaluation corpus gets at least one direct unit test (reversal detected,
  distractor not flagged, commitment-only not flagged, double-negation
  documented as a known-failing case with a skip/xfail-style comment
  rather than a silently-passing assertion).
- `apps/semantic_analysis` integration test: a window whose cosine
  similarity would classify `correct` gets forced to `incorrect` by
  `process_window` when `detect_meaning_reversal` returns `True`.
- `apps/warning_engine` test: `WindowResultRequest` accepts and passes
  through `meaning_reversal_detected` without altering escalation
  behavior (a reversal-forced `incorrect` escalates exactly like any
  other `incorrect` window — no special-casing).
- `apps/backend` test: the new column round-trips through the existing
  `window_results` POST/GET routes, and a database created before this
  column existed still works (regression guard on the migration
  pattern, mirroring the existing `human_label` column's test coverage).

## What this touches

- `apps/semantic_analysis/reversal_detection.py` — new module.
- `apps/semantic_analysis/main.py` — `process_window` integration.
- `apps/semantic_analysis/ground_truth/` — new synthetic reversal-eval
  corpus + generator script, following `generate_synthetic.py`'s pattern.
- `apps/warning_engine/main.py` — `WindowResultRequest` schema gains one
  optional passthrough field; no escalation-logic changes.
- `apps/backend/app/models.py`, `apps/backend/app/database.py`,
  `apps/backend/app/routes.py` — new column + migration entry + response
  field on `window_results`.
- Tests across all three services, plus a new eval doc following
  `docs/threshold-calibration.md`'s reporting style.
