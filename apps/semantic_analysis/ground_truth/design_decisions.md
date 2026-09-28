# Design decisions: semantic threshold calibration

Date: 2026-09-26
Branch: `feature/semantic-ground-truth`

Why these eight pieces of work, in this order, and what evidence settled each
choice. Companion docs: [finetune_spike_findings.md](./finetune_spike_findings.md)
(the fine-tune probe), [finetune_alternatives_research.md](./finetune_alternatives_research.md)
(option survey), [threshold_experiment_findings.md](./threshold_experiment_findings.md)
(threshold diagnostic), [rho_calibration_findings.md](./rho_calibration_findings.md)
(the `RHO` relative-threshold verdict behind D8),
[labeling_rubric.md](./labeling_rubric.md) (label definitions).

## The framing decision: not fine-tuning, yet

Fine-tuning `all-MiniLM-L6-v2` on 70 hand-authored pairs changed held-out
accuracy by one prediction out of 40 — noise in both directions. Two facts make
that the wrong thing to have been optimizing:

- **The thresholds were the larger error.** The production constants `0.55 /
  0.35` scored 28/60 on the fixture, worst of every option tested. Refitting two
  numbers on the same frozen encoder scores 45/60. No encoder change was
  involved.
- **The error being chased does not reach the product.** Both models separated
  `incorrect` nearly perfectly and failed on `correct` vs `weakly_correct`. But
  `warning_engine/main.py:189-194` escalates only on *consecutive* `incorrect`
  windows and treats `correct` and `weakly_correct` identically. Perfecting that
  boundary would change zero warnings.

So: fix the boundary, unblock real labels, and revisit training when there is
something to train on. The ranked alternatives (CrossEncoder first, ordinal
triplets as the SBERT-compatible fallback) stand recorded in the research doc;
nothing here forecloses them.

---

## D1 — Recalibrate the global thresholds

**Decision.** `UPPER = 0.55 → 0.36`, `LOWER = 0.35 → 0.14` in
`apps/semantic_analysis/main.py`.

**Why 0.36.** It is the grid-search optimum, and it is stable: identical in
60/60 leave-one-example-out folds and 6/6 leave-one-topic-out folds. Not a
hand-picked round number.

**Why the old values were so bad.** They were never fitted. On this fixture the
`incorrect` class tops out at cosine `0.164` and on-topic content bottoms out at
`0.148` — the entire decision region sits below `0.25`. A `LOWER` of `0.35`
therefore classified 40% of genuinely on-topic windows as off-topic.

**Held-out result** (leave-one-example-out; thresholds fitted on 59 rows,
scored on the 1 held out):

| Thresholds | Correct | Macro-F1 | Weak-F1 | False off-topic |
|---|---:|---:|---:|---:|
| `0.55 / 0.35` (old) | 28/60 | 0.427 | 0.278 | 40.5% |
| Calibrated global | 45/60 | 0.744 | 0.588 | 7.1% |
| Per-contract `UPPER` | 42/60 | 0.691 | 0.485 | 7.1% |

**Limits, stated plainly.** 60 synthetic single sentences, six topics, one
author. Production classifies ~25-second multi-sentence windows. These numbers
justify *replacing* two unfitted constants; they do not establish production
accuracy, and they should be refitted on real windows as soon as there are any.

## D2 — `LOWER = 0.14`, below its own accuracy optimum

**Decision.** Take `0.14`, not the accuracy-optimal `0.17`.

**Why.** Accuracy ties across the whole plateau, but the error *mix* does not:

| `LOWER` | Accuracy | Off-topic caught | False accusations |
|---|---:|---:|---:|
| 0.11 – 0.14 | 47/60 | 17/18 | **0/42** |
| 0.15 | 46/60 | 17/18 | 1/42 |
| 0.16 | 46/60 | 17/18 | 2/42 |
| **0.17** (accuracy optimum) | 47/60 | **18/18** | 2/42 |
| 0.18 – 0.21 | 46/60 | 18/18 | 4/42 |

`0.17` buys the eighteenth off-topic window at the price of two false
accusations. That trade is bad here, and the asymmetry is structural, not a
preference:

- A false `incorrect` is the only classification that can escalate a warning,
  and warnings feed trust-score and escrow outcomes. It penalizes someone who
  was doing the work.
- A missed `incorrect` costs one window of evidence, and escalation requires
  *consecutive* off-topic windows anyway — real drift lasts longer than 25
  seconds and will be caught on the next window.

`0.14` is the most sensitive point that still produces zero false accusations on
the fixture. Anyone who prefers maximum off-topic recall should set `0.17` and
record that choice here; it is a product call, not a modeling one.

## D3 — No per-topic or per-family thresholds

**Decision.** Rejected. Not implemented.

**Why.** Tested and lost: 42/60 against 45/60 for global calibration. It changed
three predictions from right to wrong (one guitar, two tax) and gained nothing.
Under leave-one-topic-out it is mathematically identical to global calibration,
because every unseen topic falls back to the global pair.

The family-calibration code already in `threshold_experiment.py` is kept and
tested, but stays unused: the fixture carries no family mapping, and with ~10
examples per topic there is not enough support per group to fit a boundary
anyone should trust.

## D4 — Unblock real labels: the columns never existed

**Decision.** Add the missing columns, make the write failure visible.

**Root cause.** `app/models.py` has declared `human_label`,
`labeled_by_user_id` and `labeled_at` since the feedback endpoint was written,
and `POST /session/{id}/window/{n}/feedback` writes all three — but no migration
ever created them, and `Base.metadata.create_all()` creates missing *tables*
only, never missing columns on a table that already exists. Every database built
before those attributes were declared was missing them, so the endpoint failed
and `export_real_labels.py` had nothing to export.

**Why nobody noticed.** `submitWindowFeedback` in `LiveSession.jsx` was
fire-and-forget, and `fetch` does not reject on HTTP 500. The row optimistically
re-rendered as "you said: correct" whether or not the write landed. Every rating
anyone gave was discarded silently.

**Changes.**
- `app/database.py`: `_add_missing_columns()` runs after `create_all()` on
  SQLite, checks `PRAGMA table_info` and adds only declared-but-absent columns.
  Idempotent — verified by running `init_db()` twice. Deliberately limited to
  additive nullable columns; it is not a migration framework and must never
  drop, rename or retype.
- `migrations/008_window_human_labels.sql`: the same three columns plus a
  partial index on `human_label`, for parity with the migration lineage.
  `run_migrations.py` is not wired into startup, and SQLite has no
  `ADD COLUMN IF NOT EXISTS`, so the startup path above is authoritative for
  existing databases.
- `LiveSession.jsx`: check `res.ok`, roll the optimistic update back on failure,
  and surface the error. A rating that looks saved and isn't is worse than no
  rating — nobody gives the same rating twice.

**Verified end to end** on a throwaway copy of the database: five seeded
windows, three exported. The unrated (`NULL`) row and the `out_of_scope` row
were both correctly excluded, and the output carries `session_id` / `window_id`
so future splits can be made per session rather than per row.

**Still open, and now the critical path:** zero real labeled windows exist. The
pipeline works; it has no data in it. Everything downstream — refitting on real
windows, any training comparison — waits on this.

## D5 — Cleaning: two real bugs, one wrong hypothesis

The starting suspicion was that filler-stripping was too aggressive and
distorted short windows. **An ablation showed the opposite.** Held-out accuracy
on the fixture, varying one thing at a time:

| Cleaning variant | Held-out | Fitted (U, L) |
|---|---:|---|
| No cleaning at all | 40/60 | (0.26, 0.16) |
| Original list | 45/60 | (0.36, 0.17) |
| + punctuation-aware matching | 45/60 | (0.36, 0.17) |
| + working multi-word removal | 45/60 | (0.36, 0.17) |
| + narrowed to pure vocalizations | 40/60 | (0.26, 0.16) |

Narrowing the list — leaving "like", "so", "well", "basically", "literally",
"actually", "okay" in the text — cost 5 of 60 predictions and landed exactly on
the no-cleaning score. Those markers dilute a window toward generic
conversational language; stripping them concentrates the topical content the
cosine has to work with.

**Decision.** Keep the aggressive list. Ship only the two changes the ablation
priced at zero:

- **Punctuation-aware matching.** The old code compared raw whitespace tokens,
  so `"um,"` never matched `"um"`. This changed exactly 1 of 60 fixture rows —
  a trailing `well.` that the old code kept — and moved it in the right
  direction (`0.1586 → 0.1642`).
- **Working multi-word removal.** `"you know"` and `"i mean"` sat in the set but
  `split()` can never produce them; they were dead weight. Now stripped from the
  raw string first, word-bounded so `"dopamine"` survives, with dangling
  punctuation tidied afterwards.

Casing is no longer forced to lowercase: the encoder's tokenizer is uncased, so
it changed no embedding (verified — 0 of 60 rows differ) and lowercasing only
made logged text harder to read.

**Also checked, no change needed:** the contract side already embeds
`f"{topic}. {scope}"` — full scope text was never missing.

## D6 — Skip low-content windows instead of scoring them

**Decision.** Replace `if not cleaned` with
`if len(cleaned.split()) < MIN_CONTENT_TOKENS` (3) and skip.

**Why not a fallback to the raw text.** The first draft fell back to the
uncleaned text when cleaning left too little. That is actively harmful: a
filler-only window ("so basically, like, well") embeds to near-noise, lands
under `LOWER`, and registers as `incorrect` — and consecutive `incorrect` is
exactly what escalates a warning. The system would manufacture drift out of a
teacher saying nothing much.

No evidence must mean no verdict, not an accusation. Skipping is what the
existing empty-window branch already did; this only widens it from "zero tokens"
to "too few tokens to judge". Untested on the fixture, which has no such rows —
it guards a production failure mode the fixture cannot express.

## D7 — Scope counterfactuals as a probe, never as calibration data

**Decision.** `topics.SCOPE_COUNTERFACTUALS` + `scope_sensitivity.py`, kept
strictly out of the calibration set.

**Why they were needed.** The `incorrect` examples are unrelated small talk,
which any embedding separates trivially. They cannot answer whether the model
reads the *scope* clause or merely recognizes the topic — the question the
research doc flagged as unresolved and as the main argument for a CrossEncoder.

**Design.** Twelve utterances, each genuine teaching content on one topic and
labeled twice, once under each of two different scopes for that topic. Scoring
is paired and threshold-free: does similarity move toward the scope the labels
prefer? Chance is 50%, and no calibration can flatter the result.

**Result: 11/12 (92%), mean margin `+0.10`.** The bi-encoder does read the
scope clause. That is real evidence against the "one cosine cannot model scope"
motivation for a CrossEncoder, and it raises the bar such a model has to clear.
The single failure is a guitar pair — "move between G and C slowly" scored
higher against *tuning and changing strings* than against *chord shapes and
transitions*. Guitar was also one of the two topics per-contract calibration got
wrong in D3; worth a closer look when real data exists.

**Why they stay out of calibration.** They are adversarial by construction.
Fitting thresholds on them would pull the boundary toward hard cases that are
not representative of ordinary session traffic. Twelve comparisons over six
topics is a probe, not a powered test.

## D8 — `RHO` and the relative threshold `thr = RHO*R`

**Decision.** Rejected. Not implemented, not shipped. `LOWER = 0.14` stays flat.

**Why.** The handoff proposed `RHO = 0.45`. At that value the mechanism loses to
the flat threshold already shipped, on a 25-session / 364-window corpus:

| | Off-topic caught | False accusations | Informedness |
|---|---:|---:|---:|
| flat `LOWER = 0.14` | 50/66 | 69/289 | 0.5188 |
| **`RHO = 0.45`** (proposed) | 56/66 | **96/289** | **0.5163** |
| `RHO = 0.26` (best fit) | 52/66 | 56/289 | 0.5941 |

0.45 buys six more caught digressions for **27 more false accusations** — the
exact trade D2 refused at one-fourteenth the price. Spec §7.2: if
`thr = RHO·R` does not beat a flat threshold there is nothing to ship.

**Why not simply adopt 0.26 instead.** Because it fails D1's bar and does not
hold up held out. D1 accepted `UPPER = 0.36` because it was *identical* in 60/60
and 6/6 folds. `RHO`'s argmax is identical in no fold family: 23/25 session
folds, 23/25 topic folds, and **1 of 4 author folds** — 0.25 / 0.26 / 0.36 /
0.36 as each author is held out.

**Held-out result** (each fold's held-out sessions scored at that fold's own
training-fitted argmax, then pooled):

| | Off-topic caught | False accusations | Informedness |
|---|---:|---:|---:|
| in-sample @0.26 | 52/66 | 56/289 | 0.5941 |
| pooled leave-one-session-out | 50/66 | 61/289 | 0.5465 |
| pooled leave-one-**author**-out | 50/66 | **71/289** | **0.5119** |
| flat `LOWER` control | 50/66 | 69/289 | 0.5188 |

The caught-rate advantage is entirely in-sample — 50/66 in every held-out arm,
identical to flat. Held out it is at best a quieter threshold, and under the
author fold it is louder than flat. Splitting by authorship says the same thing:
the whole -14 false-accusation gain comes from the 10 real AnnoMI sessions, while
the 15 agent-authored sessions show +1.

**Limits, stated plainly.** 25 sessions, 4 author strata, structural labels with
no kappa, synthetic prosody throughout, three of five categories LLM-authored,
and the WER robustness leg (spec §7.5 criterion 4) **never run** — the S3 bucket
`transcribe.py` stages audio in belongs to another AWS account. This rejects
`RHO` on this corpus; it does not prove no relative threshold can work. Two
further caveats belong here: §7.1's "only above-threshold windows update `R`"
safeguard is a provable no-op for any `RHO < 1`, so `R` is just a running max;
and the argmax is objective-dependent (0.20 under the committed secondary and
under accuracy, 0.26 under the primary).

Anyone who wants to revisit this needs more than four author strata and real
audio durations, should run the WER leg first, and should record the result here.
Full detail: [rho_calibration_findings.md](./rho_calibration_findings.md).

---

## What this does not do

- **No model was trained**, and no encoder or inference path changed.
- **No production accuracy was measured.** Every number here comes from 60
  synthetic single sentences; production sees ~25-second multi-sentence windows.
  The sentence-to-window distribution shift remains the largest untested
  uncertainty.
- **Session replay now exists, but not as a warning-sequence metric.**
  `eval_dataset/tools/replay.py` reproduces production windowing (`main.py`'s
  buffering/flush rules) offline and has been run over all 25 calibration
  sessions to build the window corpus D8 scores against. It does windowing
  only — no cosine, no embedding, no `warning_engine` — so the false-off-topic
  *rate* above remains a window-label statistic, not a measured rate of
  escalated warnings. Replaying complete sessions through the warning ladder
  to measure false warning sequences is still open (see "Next, in order",
  item 3).

## Next, in order

1. **Collect real labels.** The pipeline is fixed and verified; it needs
   sessions with humans rating windows. Everything else is blocked on this.
2. **Measure reviewer agreement** on the `weakly_correct` / `incorrect` boundary
   before scaling collection — two reviewers, overlapping sample. Disagreement
   there caps any model's achievable score.
3. **Refit `UPPER` / `LOWER` on real windows**, splitting by session, and replay
   complete sessions to measure false *warning sequences* rather than window
   accuracy.
4. **Then, and only then**, revisit training — and only if the
   `weakly_correct` / `incorrect` boundary is still wrong after recalibration.
   The `correct` / `weakly_correct` boundary is not worth training for while the
   warning engine ignores it.

## Note on the harness's loader

`apps/semantic_analysis` is a service directory, not an importable package, so
the offline harness reaches production `windowing.py` through
`importlib.util.spec_from_file_location`. That shim was copy-pasted at four
call sites — `eval_dataset/tools/replay.py`, `embed_windows.py`,
`tests/test_embed_windows.py` and `rho_calibration.py` — which loaded the file
four independent times under four different module names. Harmless in practice,
but squarely against the premise the work opens with: production and the
harness share the logic *so they cannot diverge*, and four live copies is the
shape in which they can. It is now loaded exactly once, in
`eval_dataset/tools/windowing_bridge.py`, which every call site imports from;
`tests/test_windowing_bridge.py` fails if a new shim appears.

## Note on version control

As of commit `f5424d0`, `threshold_experiment.py`, all findings/design
documents, `rho_calibration_results.json` and the test files
(`test_threshold_experiment.py`, `test_rho_calibration.py`) are **tracked**,
not untracked as an earlier draft of this note claimed. `.gitignore:34` still
excludes `ground_truth/*.csv` deliberately (synthetic is regenerable; real
transcripts should not be committed), but the scripts and findings themselves
are committed and are the record of why these thresholds have the values they
do.
