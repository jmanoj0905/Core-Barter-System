# Context-Gating the Safety/Toxicity Alert Path — Design

Status: draft, pending user review
Date: 2026-09-29

## Purpose

This is sub-project 4 of 4 research-driven modifications to the
during-session QA/safety pipeline, targeting eventual publication (SMARTCOM
Goa 2027). Source research: see
`3_Capstone/during-session-qa-safety-novelty-research.md` (Obsidian vault),
Clusters 5 and 7. Sub-projects 1-3 are spec'd/planned separately and out
of scope here.

**Correcting the research note's assumption:** the note described "the
safety alert path" generically as an undocumented classifier. Investigation
in this session found there are actually **two independent safety paths**:

1. `apps/backend/app/safety.py` — image-based NSFW detection (NudeNet) on
   video frames, POSTed from the frontend to `/safety/check-frame`. Not
   touched by this design; it has no topical content to be "context-blind"
   about — a nude frame is a nude frame regardless of what's being
   discussed.
2. `apps/audio_pipeline/main.py`'s `check_toxicity()` — text-based
   moderation via the Mistral Moderation API, run on every transcribed
   segment, POSTed to `warning_engine`'s `/safety/alert` →
   `receive_safety_alert`, which sets `severity = "severe" if hard_block
   else "strong"` with **no awareness of topic-relevance at all.**

This design targets **only path 2** — the text-toxicity path — which is
the one Kim et al. (arXiv:2605.25454) and ToxiAlert-Bench
(arXiv:2605.15984) actually speak to (context-blind moderation over-
flagging legitimate sensitive discussion; text-only detection missing
delivery/tone). Path 1 is out of scope: there is no "context" dimension
for nudity detection to be gated against.

## Non-goals

- No ToxiAlert-Bench-style dual-head audio+text/paralinguistic model —
  the research note already scoped this out for a capstone timeframe
  (needs a new dataset, training infra, and a new speech-encoder
  inference path in `audio_pipeline`). ToxiAlert-Bench is cited as
  motivating evidence for *why* text-only detection misses things, not
  implemented here.
- No change to the Mistral Moderation API call itself, its thresholds
  (`TOXICITY_FLAG_THRESHOLD`, `TOXICITY_BLOCK_THRESHOLD`), or the NSFW
  path in `backend/app/safety.py`.
- No per-utterance topic-relevance check — `semantic_analysis` classifies
  ~25-second windows, not individual 5-second transcribed segments, and
  there is no existing mechanism to align a specific flagged segment to
  a specific window. This design uses the session's **most recent window
  classification** as the available context signal, not a same-utterance
  match — stated as a limitation, not hidden.

## Architecture

### Context signal

`warning_engine`'s `sessions[barter_id]` state dict already updates on
every window result (`run_warning_decision`, currently tracking
`consecutive_incorrect` etc.) but does not retain the *last* raw
classification. Add `state["last_classification"]: str | None` (one of
`"correct"`, `"weakly_correct"`, `"incorrect"`, or `None` before any
window has been classified), set in `run_warning_decision` from
`request.classification`, alongside the existing counter updates.

### Gating rule in `receive_safety_alert`

Currently:
```python
severity = "severe" if hard_block else "strong"
```
Becomes:
```python
on_topic_confirmed = state.get("last_classification") in ("correct", "weakly_correct")
severity = "strong" if hard_block and on_topic_confirmed else ("severe" if hard_block else "strong")
```
(`state` must be looked up via `sessions.get(barter_id)`, defaulting to
`{}` if the session hasn't been `init`'d yet — matching this file's
existing `if barter_id in sessions:` guard pattern used a few lines
below for `warning_history`.)

Exact rule, stated plainly:

| `hard_block` | `last_classification` | Severity |
|---|---|---|
| `False` | any | `strong` (unchanged from today) |
| `True` | `"correct"` or `"weakly_correct"` | `strong` (**downgraded** from today's `severe`) |
| `True` | `"incorrect"` | `severe` (unchanged) |
| `True` | `None` (no window classified yet) | `severe` (unchanged — **no evidence, no downgrade**) |

The `None` case is a deliberate design choice, not an oversight: it
mirrors this codebase's existing principle (see
`docs/superpowers/specs/2026-09-27-analogy-vs-drift-handoff.md`'s framing
of "missing evidence proves nothing," there applied to denying a teacher
an undeserved `DISPUTE` escape) — here applied to denying a flagged
utterance an undeserved downgrade. A session with no window evidence yet
gets today's existing (safer, more punitive) default, not the new,
more lenient behavior.

### What is NOT claimed

This does not verify that the flagged segment and the "most recent
window" cover the same span of speech — `audio_pipeline` transcribes in
5-second segments and posts toxicity alerts immediately per segment,
while `semantic_analysis` classifies ~25-second accumulated windows on a
separate cadence. The context signal is "what was this session's general
topical state around this time," not "was this exact flagged sentence
on-topic." This approximation is stated explicitly in the evaluation
write-up, not glossed over — a tighter same-span alignment would require
segment-to-window correlation that does not exist anywhere in the
pipeline today and is out of scope.

## Evaluation plan

No real toxic-and-on-topic session transcripts exist (and are not
something to manufacture for real, for obvious reasons). A small codex
agent synthesizes a labeled corpus of **plausible session-state
sequences**, not real transcripts:

- Each synthetic case is a tuple `(last_classification, hard_block,
  expected_severity)` covering all four rows of the table above, plus a
  handful of realistic session narratives in prose for the paper's
  qualitative discussion (e.g., "a barter session discussing an overdue
  debt — repeated mentions of money owed trip the moderation API's
  financial-threat category, but the session's last three windows were
  all `correct` against the agreed topic 'settling an outstanding debt'
  — today's system escalates to `severe`; the gated system downgrades to
  `strong`").
- Metric: this is a deterministic rule over three discrete inputs, so
  "evaluation" is exhaustive-case coverage (all 2×4 combinations of
  `hard_block` and `last_classification`), not a statistical
  precision/recall report — stated as such, not dressed up as a larger
  empirical result than it is.

## Testing

- Unit/integration tests in `apps/warning_engine/tests/` covering all
  four table rows directly against `receive_safety_alert`, plus the
  `barter_id not in sessions` case (no session state at all, not just
  no classification yet — must behave identically to the `None`
  classification case, i.e. `severe`, not raise).
- A regression test confirming non-`hard_block` alerts are completely
  unaffected by `last_classification` (the gating logic must not
  accidentally touch the `strong` branch that already existed).

## What this touches

- `apps/warning_engine/main.py` — `run_warning_decision` (set
  `last_classification`), `receive_safety_alert` (gating rule),
  `_new_state` (add the new key with its `None` default).
- `apps/warning_engine/tests/` — new test file.
- `docs/threshold-calibration.md` or a new short doc — record the
  gating rule and its "most recent window, not same-span" limitation,
  matching this codebase's existing practice of stating what a
  mechanism does not cover.
