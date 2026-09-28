# Design — `severe` becomes advisory; drift is flagged in-session, priced at session end

**Status: approved for implementation. Nothing here is implemented yet.**

This spec covers **Half A** of the analogy-vs-drift work: the parameter-free
half, which contains no fitted constant and makes no model call.

Half B — telling a pedagogical analogy from deliberate drift, and paying credit
for the former — is **deliberately out of scope**. §7 records why: the corpus
cannot currently calibrate it, and the reason is measured, not assumed.

Read alongside:

- [2026-09-28-analogy-adjudication-design.md](./2026-09-28-analogy-adjudication-design.md) — the
  unapproved synthesis this spec carves Half A out of. One of its claims is
  corrected in §4.1.
- [2026-09-27-analogy-vs-drift-handoff.md](./2026-09-27-analogy-vs-drift-handoff.md)
- [analogy_handling_findings.md](../../../apps/semantic_analysis/ground_truth/analogy_handling_findings.md)

---

## 1. The intent

Teachers digress. Some digressions are analogies that teach the topic; some are
deliberate drift away from the agreed contract. **The system cannot tell them
apart while the session is running**, and attempting to do so in-session has
been abandoned.

The intent is therefore to split the two jobs across time:

- **During the session** — flag drift accurately and warn the teacher. Warnings
  stay honest and immediate, because the teacher needs them in time to act.
- **At session end** — price the session. No warning issued mid-session may, by
  itself, destroy the teacher's escrow, because at the moment it was issued the
  system did not yet know whether it was watching an analogy or a drift.

Half A delivers the second half of that split. It does not yet adjudicate
analogy-vs-drift; it removes the mechanism that currently prices that
distinction wrongly and irreversibly.

## 2. The failure being fixed

`_decide_verdict_type` (`apps/backend/app/routes.py:169-171`) currently reads:

```python
topic_failed = topic["has_evidence"] and (
    topic["on_topic_percentage"] < 40 or topic["has_severe_warning"]
)
```

A single `severe` warning anywhere in a session forces `DISPUTE`, hence
`qa_score 0.0`, hence total loss of the teacher's escrowed credits — regardless
of how the other fifty-nine minutes went. `severe` fires at three consecutive
`incorrect` windows, roughly fifteen seconds of sustained off-topic speech.

Replaying the ladder over the 25-session calibration corpus:

| | reaches `severe` → `DISPUTE` → `qa_score 0.0` |
|---|---|
| honest sessions (zero declared digressions) | **5 of 10** |
| drift-bearing sessions | 9 of 15 |
| `interleaved_abuse` specifically | **0 of 5** |

Half of the honest teachers in the corpus lose their escrow. `sess_CAL23` runs
eight consecutive `incorrect` windows with no declared digression at all.
Meanwhile the deliberate abuser who alternates one on-topic window with one
off-topic window escapes the ladder entirely, because any non-`incorrect`
window resets the counter.

**This is not primarily an analogy problem.** Any session that simply embeds low
costs its teacher the escrow, and the one shape the ladder reliably misses is
the deliberate one. The analogy case is a special case of a general
false-accusation problem, and Half A fixes the general problem without needing
to recognise an analogy at all.

Supporting evidence, from the same corpus at the shipped flat `LOWER = 0.14`:
**66 of 69 false digressions arise in `abstract_on_topic` and
`depressed_baseline`** — the two categories containing no drift whatsoever. The
window classifier is not confused about drift. It is confused about sessions
that sit low throughout. Moving thresholds cannot fix that; not pricing it
fatally can.

## 3. Design

### 3.1 Behaviour

In-session behaviour is unchanged. The same windows are classified by the same
`classify()` with the same `UPPER = 0.36` / `LOWER = 0.14`; the same
silent → strong → severe ladder fires on the same schedule; the teacher sees the
same warnings at the same moments.

What changes is that `severe` no longer reaches into settlement. It becomes an
**advisory record**: stored, displayed, broadcast, and available to Half B's
adjudicator — but not a veto on payout. The money decision is made from the
on-topic percentage alone.

### 3.2 Code changes

Two files change logic, plus one column. No new module, no new service, no
model call.

`has_severe_warning` occurs exactly three times in production code
(`routes.py:151`, `:156`, `:170`) — all three inside the two functions below.
Nothing else reads it: no other backend caller, no test, and no frontend code.
The frontend renders warning severity from the `warnings` rows themselves
(`LiveSession.jsx:19`, `PostSession.jsx:22`), so severe warnings keep
displaying unchanged, which is what F5 protects.

| File | Change |
|---|---|
| `apps/backend/app/routes.py` `_evaluate_topic_quality` (:132-157) | stop computing `has_severe_warning`; return `severe_warning_count: int` in its place — retained for display and for Half B |
| `apps/backend/app/routes.py` `_decide_verdict_type` (:160-182) | delete `or topic["has_severe_warning"]` from the `topic_failed` clause. Signature unchanged. The `< 40` and `>= 70` bands are **not** touched |
| `apps/warning_engine/main.py` `run_warning_decision` (:178-260) | include `advisory: True` in the `/warnings/log` payload for `severe` rows. Ladder copy and thresholds unchanged |
| `apps/warning_engine/main.py` lifespan banner (:133) | fix the stale startup line, which advertises `1→silent 2→mild 3–4→strong 5+→severe`. The actual ladder is `1→silent 2→strong 3+→severe`. No `mild` tier has ever existed |
| `apps/backend/app/escrow.py` | **none.** `apply_settlement` already keys off `qa_score` |
| `apps/semantic_analysis/main.py` | **none.** Deliberate — this is why the false-digression count cannot regress |

### 3.3 Data model

One column:

```sql
ALTER TABLE warnings ADD COLUMN advisory BOOLEAN DEFAULT 0;
```

shipped as `apps/backend/migrations/009_advisory_warnings.sql` **and** added
idempotently in the `app/database.py` startup path. Migration `008` established
that convention and documents why: `run_migrations.py` uses `executescript`
with no applied-migrations table, SQLite has no `ADD COLUMN IF NOT EXISTS`, and
so the startup path is the route that works on an existing database.

`Warning.advisory` is added to `apps/backend/app/models.py`.

Nothing else. `SpanAdjudication`, `WindowResult.credited_by_span_id`,
`WindowResult.teacher_seconds` and `Warning.retracted_at` all belong to Half B.

### 3.4 The verdict policy consequence

Until Half B lands, **the on-topic percentage is the only penalty mechanism.**
Sustained drift still disputes, because it drives the percentage below 40.
Moderate drift softens.

Measured on the 25-session corpus, removing the veto moves **12 of 25 sessions
from `DISPUTE` to `PARTIAL`**, including all five `gradual_drift` sessions:

- **5 of the 12** are the false accusations of §2 — the fix working as intended.
- **7 of the 12** are genuine drifters who will receive `qa_score 0.5` instead
  of `0.0` in the interim.

That is a real, temporary loosening of enforcement, accepted deliberately. It is
the price of refusing to ship a fitted constant, and it is reversed when Half B
restores a calibrated penalty path. **A spec that reported only the five would
be hiding the seven.**

## 4. Corrections to the prior design document

### 4.1 The warning counter is already span-local

`2026-09-28-analogy-adjudication-design.md` §3 lists "counter becomes span-local
(reset on any window ≥ `LOWER`)" as a production change. **It is not a change.**
`apps/warning_engine/main.py:194` already resets `consecutive_incorrect` to zero
on any non-`incorrect` window. That document's own §1.1 depends on this fact —
it is exactly why `interleaved_abuse` escapes the ladder 5 of 5.

Half A is correspondingly smaller than that table implies.

## 5. Testing

`_decide_verdict_type` is a pure function over a dict and needs no database.
Table-driven cases, in `apps/backend/tests/`:

| Case | Expected | Guards |
|---|---|---|
| severe present, 85% on-topic, duration+confirmation pass | `SUCCESSFUL` | the bug in §2 |
| severe present, 35% on-topic | `DISPUTE` | sustained drift still disputes |
| no severe, 35% on-topic | `DISPUTE` | the percentage band is untouched |
| no severe, 55% on-topic | `PARTIAL` | band boundaries unmoved |
| `has_evidence=False` (zero windows), duration+confirmation pass | `SUCCESSFUL` — unchanged | pins current behaviour; see note below |
| `has_evidence=False`, neither pass | `DISPUTE` | missing evidence never *manufactures* a pass |
| `terminated=True` | `DISPUTE` | termination still dominates |

Plus: `_evaluate_topic_quality` returns `severe_warning_count` as an integer
count and no longer returns `has_severe_warning`; and the `advisory` column is
created idempotently by the startup path when run twice against one database.

**A note on ISSUE-019, corrected during planning.** An earlier draft of this
section claimed a zero-window session can never reach `SUCCESSFUL`. It can:
`topic_ok_for_success` is `not has_evidence or pct >= 70`, so a session with no
windows at all but a passing duration and both confirmations returns
`SUCCESSFUL` today. Verified directly against `_decide_verdict_type`.

What ISSUE-019 actually guarantees is the other direction — `topic_failed`
requires `has_evidence`, so missing evidence can never manufacture a `DISPUTE`,
and the zero-window `on_topic_percentage` of 0.0 is never read as a failure.
**Half A changes neither direction**, and the tests above pin both so that it
cannot start to. Whether a session with no monitoring evidence should be able to
settle at `qa_score 1.0` is a real question, but it is a pre-existing policy
question and it is not this spec's to answer.

## 6. Falsification criteria

Registered before implementation. Each is a condition that, if met, means this
design is wrong and must not ship.

| # | Criterion |
|---|---|
| F1 | Any session with `on_topic_percentage < 40` stops reaching `DISPUTE` |
| F2 | The zero-window path changes verdict in either direction (it must behave exactly as it does today) |
| F3 | Dataset-A false digression count changes at all — it must be **identical**, since `semantic_analysis/main.py` is untouched |
| F4 | `apply_settlement` behaviour changes for any given `qa_score` |
| F5 | A `severe` warning stops being written, displayed, or broadcast in-session |

F3 and F5 are the load-bearing ones: Half A must change *what a warning costs*,
and nothing about *when a warning fires* or how it is shown.

## 7. Why Half B is not in this spec

Half B's architecture is settled — cue marker as a cheap trigger, verified
vehicle-term→topic-term mappings as evidence, an LLM judge permitted only to
**veto** credit and never to grant it, so that a judge outage or error can cost
credit but never escrow. What is missing is the evidence to set its constants.

Two measurements over the corpus, run 2026-09-28:

**7.1 The discourse marker separates, and it is not the obvious one.**
Classic analogy connectives (`imagine`, `for example`, `similar to`, `analogy`)
are anti-discriminative: they fire on 2 of 23 declared drift span onsets and on
only 1 of the 4 analogy scripts. The class that separates is **announced detour
plus return promise** — "put the maths aside … I promise we'll map every part of
it back onto the code" — at **4 of 4 analogy scripts, 0 of 23 drift spans**.

It is also forgeable for the price of one sentence, so it can serve as a trigger
for adjudication and must never be evidence for credit.

**7.2 The mapping count cannot be calibrated on this corpus.**
Two constructions were measured. Bag-of-words co-occurrence around a copula:
analogy `[4, 8, 8, 12]`, drift `[0 … 17]` — no separating threshold. Directional
copula (vehicle vocabulary strictly left of the verb, topic vocabulary strictly
right): far higher precision, 18 of 20 drift scripts scoring zero — but
analogy `[0, 2, 3, 3]` against drift `[0 ×18, 1, 3]`. **Again no separating
threshold.**

Two causes, of different kinds:

- `sess_CAL05` scoring 3 is a **lexicon-construction defect**, not a signal
  failure: its hits are genuine on-topic teaching mappings, admitted because the
  vehicle lexicon was built script-wide instead of span-locally. Fixing it needs
  the replay harness.
- `sess_C05` scoring 0 is **real and already known**. Its analogy is two windows
  long; the prior design records the same limit — below roughly four windows
  this whole approach has nothing to say.

**7.3 The blocker is the corpus, not the design.** There are exactly four
analogy-bearing scripts, and **none of them declares its analogy spans** — all
four are `CATEGORY: code_switch` with no `DIGRESSION:` header, so `gold_labels.py`
has nothing to map. Any constant fitted today would be fitted to four unlabelled
examples: the same evidential shape for which `RHO` was rejected, and on which
the prior design's F5 and F7 both triggered.

Half B is therefore blocked on two pieces of corpus work, in this order:

1. **Declare analogy spans** in `sess_B08`, `sess_C05`, `case_B1_incremental`
   and `case_B2_late`, so span-local lexicons and gold labels become derivable.
2. **Write the missing adversary** (prior design §5.1): sustained
   *single-domain* off-topic speech — defeating a domain-cluster gate — carrying
   a **forged** detour marker and cheap retrofit mappings. The corpus contains
   wandering-drift-plus-retrofit and single-domain-drift-without-retrofit, but
   not their combination, which is the script that attacks every gate at once.

Neither is required for Half A, which is why Half A ships first.

## 8. Evidence

| Claim | Source |
|---|---|
| 5/10 honest sessions reach `severe`; 9/15 drift-bearing; `interleaved_abuse` 0/5 | ladder replay over the 25-session calibration corpus, recorded in the prior design's §6 |
| 12 of 25 sessions move `DISPUTE` → `PARTIAL`; 5 false, 7 genuine | same |
| 66 of 69 false digressions come from `abstract_on_topic` + `depressed_baseline` | flat `LOWER = 0.14` replay, prior design §1.2 |
| marker classes: 4/4 analogy vs 0/23 drift onsets; connectives 1/4 vs 2/23 | probe over `eval_dataset/scripts/**`, 2026-09-28 |
| mapping counts: co-occurrence `[4,8,8,12]` vs `[0…17]`; directional `[0,2,3,3]` vs `[0×18,1,3]` | same probe run |
| counter is already span-local | direct read of `apps/warning_engine/main.py:190-195` |

**Caveat.** The §2 and §3.4 replay figures are carried over from the prior
design document's recorded runs. The production code those runs describe was
re-read and verified for this spec; the replays themselves were not re-executed.
The §7 marker and mapping figures were produced fresh for this spec.
