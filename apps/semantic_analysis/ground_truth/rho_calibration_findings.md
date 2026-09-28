# `RHO` calibration: the relative threshold `thr = RHO*R`

Date: 2026-09-28

The deliverable of the analogy-corpus project (spec
[2026-09-27-analogy-corpus-design.md](../../../docs/superpowers/specs/2026-09-27-analogy-corpus-design.md)):
a corpus, a measurement harness, and a verdict on the relative threshold the
handoff proposed. Companion docs:
[design_decisions.md](./design_decisions.md) (D8 records this decision),
[threshold_experiment_findings.md](./threshold_experiment_findings.md) (the
global `UPPER` / `LOWER` calibration this builds on),
[analogy_handling_findings.md](./analogy_handling_findings.md) (the nine-script
evidence that motivated `RHO`).

## Result

**`RHO` is rejected.** No value of it is cleared to ship, and the mechanism
`thr = RHO*R` should not be built. Four independent findings support this, and
the first settles it on its own.

1. **`RHO = 0.45`, the value the handoff actually proposed, is worse than the
   flat `LOWER = 0.14` control it has to beat.** In-sample over all 25 sessions
   it catches 56 of 66 digression windows against flat's 50, but produces
   **96 false digressions out of 289 on-topic windows** against flat's 69 —
   informedness 0.5163 against 0.5188. It buys 6 more caught digressions for 27
   more false accusations. D2's stated bias is the opposite trade: give up a
   caught digression to avoid a false accusation. Spec §7.2 — "if `thr = RHO·R`
   does not beat a flat threshold there is nothing to ship."
2. **Falsification criterion C1 triggers.** The argmax differs across author
   folds: 1 of 4 folds matches the overall 0.26.
3. **Falsification criterion C3 triggers.** The argmax is not identical across
   session or topic folds — 23/25 each, against D1's *identical* bar.
4. **Criterion C2's advantage reverses out of sample.** The mechanism's
   false-digression advantage on the two on-topic categories holds in-sample and
   under session/topic folds, but under the author fold — the fold spec §7.3
   calls "the one most likely to expose idiom-fitting" — it is worse than flat.

C4 (WER robustness) was **not evaluated**; see below. It is neither a pass nor a
fail.

Stated narrowly and without overstating it: the best-fitting value on this
corpus is **0.26**, not 0.45, and at 0.26 the mechanism genuinely does beat the
flat control in-sample and under session folds. What it does not do is beat flat
at the value proposed, satisfy D1's stability bar, survive the author fold, or
transfer to the agent-authored half of the corpus. Per spec §11 a documented
rejection is a **successful completion** of this project.

No production code changed. No mechanism code ships. `thr = RHO*R` exists
nowhere outside `ground_truth/`.

## Data and method

- Corpus: 25 sessions, **364 windows**, in `eval_dataset/scripts/calibration/`.
  Five categories of five sessions each: `gradual_drift`, `adversarial`,
  `interleaved_abuse` (digression-bearing), `abstract_on_topic`,
  `depressed_baseline` (declare no digressions). 66 off-topic windows, 289
  on-topic windows, 9 excluded as boundary cases.
- Authorship: 15 sessions LLM-authored under the §5.3 blinding protocol across
  three models (5 each: `agent:opus-5`, `agent:sonnet-5`, `agent:haiku-4.5`);
  10 sessions are real human transcripts excerpted from AnnoMI
  (`real:annomi`), carrying AnnoMI's own third-party topic labels.
- Durations are **synthetic** throughout (`--durations synthetic`). No real
  audio and no ASR transcripts were involved in any number here.
- Mechanism as pinned by spec §7.1: `thr_i = RHO * R`, with `R` read before
  window `i` may update it, `R` starting at `UPPER = 0.36`, capped at 0.80, and
  updated `R = min(0.80, max(R, sim_i))` when `sim_i >= thr_i`.
- Control on every table and every fold: the shipped flat `LOWER = 0.14`.
- Grid: `RHO` from 0.20 to 0.80 at 0.01, **61 values**, built with integer
  arithmetic.
- Objective (committed before the run): **informedness** = caught rate minus
  false-digression rate; ties broken toward the lower false-digression rate,
  then the lower `RHO`. `asymmetric_cost` is the committed secondary, reported
  but not used to pick the argmax.
- Folds: leave-one-session-out (25), leave-one-topic-out (25),
  leave-one-author-out (4). Held-out sessions are scored at **that fold's own
  training-fitted argmax**, then pooled.
- Boundary windows in the `[0.4, 0.6]` off-topic-fraction band are excluded per
  spec §5.2 rather than forced to a label.

Full per-fold, per-session, per-category and per-objective detail is in
[rho_calibration_results.json](./rho_calibration_results.json) (668 KB; query
keys rather than reading it whole).

## The error mix, in-sample, all 25 sessions

Presented in D2's shape, because D2's lesson is the one that applies: the
aggregate score ties across a plateau while the error *mix* does not.

| `RHO` | digressions caught | **false digressions** | on-topic affected | informedness |
|---|---:|---:|---:|---:|
| 0.20 | 47/66 | 44/289 | 0.152 | 0.5599 |
| 0.25 | 50/66 | 53/289 | 0.183 | 0.5742 |
| **0.26** (argmax) | **52/66** | **56/289** | **0.194** | **0.5941** |
| 0.30 | 52/66 | 66/289 | 0.228 | 0.5595 |
| 0.36 | 55/66 | 73/289 | 0.253 | 0.5807 |
| 0.45 (proposed) | 56/66 | **96/289** | 0.332 | 0.5163 |
| 0.60 | 60/66 | 139/289 | 0.481 | 0.4281 |
| 0.80 | 66/66 | 204/289 | 0.706 | 0.2941 |
| **flat `LOWER = 0.14`** | **50/66** | **69/289** | **0.239** | **0.5188** |

Read down the false-digression column. The mechanism's caught rate is bought
almost linearly with false accusations, and past roughly 0.30 the price rises
faster than the return. At `RHO = 0.80` it catches every digression in the
corpus and falsely accuses 70.6% of on-topic windows.

The proposed 0.45 sits on the wrong side of the flat control on informedness,
and far on the wrong side of it on the one error D2 argues is asymmetrically
expensive: a false `incorrect` is the only classification that can escalate a
warning, and warnings feed trust score and escrow.

## Out of sample

The table above is **in-sample**: 0.26 was fitted on all 25 sessions, including
the ones it is scored on. Spec §7.2's gate belongs to the pooled cross-validated
numbers, where each fold's held-out sessions are scored at that fold's own
training-fitted argmax.

| | caught | false digressions | informedness | beats flat? |
|---|---:|---:|---:|:--|
| in-sample @0.26 (headline) | 52/66 | 56/289 | 0.5941 | yes |
| pooled LOO-session | 50/66 | 61/289 | 0.5465 | yes |
| pooled LOO-topic | 50/66 | 61/289 | 0.5465 | yes (but see M-B) |
| pooled LOO-**author** | 50/66 | **71/289** | **0.5119** | **NO** |
| flat `LOWER = 0.14` control | 50/66 | 69/289 | 0.5188 | — |

**The caught-rate advantage vanishes entirely out of sample.** 50/66 in every
arm, identical to flat. The two extra digressions the mechanism catches at 0.26
are an artefact of fitting on the sessions being scored; held out, it catches
exactly what a flat threshold catches. The only quantity that still differs is
the false-digression count — 8 fewer than flat under session/topic folds, and
**2 more than flat** under author folds.

So the honest description of what survives holdout is: not a better detector, at
best a slightly quieter one, and not even that when the held-out sessions come
from an unseen author.

The pooled LOO-author row's "beats flat? NO" rests on 71 false digressions
against flat's 69 — a 2-window difference out of 289, with no confidence
interval or paired test computed. That delta is within single-window
resolution and is not what the rejection rests on: C1 and C3 are structural
(argmax spread 0.25–0.36 across folds), and §7.2's in-sample failure at the
proposed `RHO = 0.45` is 27 false accusations, not 2.

## Argmax stability, per fold family

D1's bar for calling a fitted constant stable was *identical* across folds —
`UPPER = 0.36` was identical in 60/60 example folds and 6/6 topic folds. `RHO`
does not meet that bar in any family.

| fold family | folds | matching overall argmax (0.26) | identical? | distinct argmaxes |
|---|---:|---:|:--|---|
| leave-one-session-out | 25 | 23/25 (92%) | **no** | 0.25, 0.26, 0.36 |
| leave-one-topic-out | 25 | 23/25 (92%) | **no** | 0.25, 0.26, 0.36 |
| leave-one-**author**-out | 4 | **1/4 (25%)** | **no** | 0.25, 0.26, 0.36 |

Per-author fold argmaxes, which is where the instability is severe:

| held-out author | fitted argmax |
|---|---:|
| `agent:haiku-4.5` | 0.25 |
| `agent:opus-5` | 0.36 |
| `agent:sonnet-5` | 0.26 |
| `real:annomi` | 0.36 |

Hold out any one of the three agent authors and the fitted constant moves; hold
out the real-data half and it moves to 0.36, a 38% shift from the overall value.
The spread 0.25–0.36 is not a rounding wobble — at 0.36 the in-sample false
digressions are 73/289 against 56/289 at 0.26.

## The WER robustness leg: not run

| target WER | argmax | plateau | movement vs WER-0 |
|---|---|---|---|
| 0% | — | — | — |
| 10% | — | — | — |
| 20% | — | — | — |
| 30% | — | — | — |

**This sweep was not run, and the cells above are absent measurements, not
zeros.** Two reasons, both outside the harness:

1. **The AWS spend was declined.** Re-voicing 25 sessions through Polly and
   transcribing them through AWS Transcribe at four noise targets was not
   authorised.
2. **S3 staging is blocked regardless.** `eval_dataset/tools/transcribe.py`
   stages audio in the bucket `core-barter-audio-tmp`, which belongs to a
   **different AWS account** — `get_bucket_location` returns 403 /
   `AccessDenied` for this credential. Even with the spend approved the sweep
   could not run as written. See the spec correction below: §6.2 misidentifies
   the gate as Polly IAM.

Consequently no real-duration transcripts exist
(`eval_dataset/transcripts/synthetic` is empty) and `durations="timed"` cannot
be exercised.

What *is* built and unit-tested, so this leg can be filled in later without
redoing anything else:

- `timed_durations()` in `eval_dataset/tools/run_stt_sweep.py`, which derives
  real window durations from a transcript's word timings, covered by
  `eval_dataset/tools/tests/test_run_stt_sweep.py`.
- `align_wer_runs()` in `rho_calibration.py`, which aligns two runs by
  `ReplayWindow.index` rather than by position and reports window-count
  mismatches — necessary because a window emitted at WER-0 may be skipped at
  WER-30 for falling under `MIN_CONTENT_TOKENS`, which would silently misalign a
  positional comparison. Both behaviours are covered in
  `test_rho_calibration.py`.
- `load_sessions()` and `evaluate()` refuse `durations="timed"` explicitly
  rather than silently scoring synthetic durations, and
  `rho_calibration.py --all-wer` is accepted and refuses, printing the reason.

Filling this leg needs an S3 bucket in this account plus the Transcribe spend;
nothing in the corpus or the harness has to be rebuilt.

**How to unblock it.** `eval_dataset/tools/transcribe.py` already reads the
bucket from `AWS_S3_BUCKET` (the hard-coded `core-barter-audio-tmp` is only its
default), so the leg needs no code change — point that variable at a bucket the
run's credential owns, in the same region as the Transcribe calls, and run
`rho_calibration.py --all-wer`. Polly is *not* a blocker: that permission is
already granted (see the spec correction below). `timed_durations()` and
`align_wer_runs()` are built, unit-tested and guarded against being fed
synthetic durations, so the sweep can be filled in without redoing any of it.

## Verdict against each pre-registered criterion (spec §7.5)

The four criteria were registered before the run. Each is resolved below on its
own terms.

### C1 — "The argmax differs across author folds" — **FAIL (triggers)**

Triggers. Per-fold argmaxes are 0.25 / 0.36 / 0.26 / 0.36; **1 of 4** folds
matches the overall 0.26. The criterion's stated reasoning is that this means
the constant is fitting one model's idiom rather than conversational structure,
and the pattern is consistent with that: the fitted value moves when the author
population changes, including when the real-human half is held out.

### C2 — "False-digression rate on the two on-topic categories no better than flat" — **does not trigger in-sample; REVERSES out of sample**

This does **not** trigger in-sample. On `abstract_on_topic` +
`depressed_baseline` combined, the relative threshold produces 52/156 false
digressions (0.333) against flat's 66/156 (0.423) — a real improvement on
exactly the quantity this criterion measures.

Out of sample the picture splits by fold family. Pooled on-topic-only
false-digression delta versus flat is **-9 under session and topic folds** (57
against 66) but **+1 under author folds** — worse than flat. The criterion is
therefore **not recorded as a clean pass**: it holds under the fold families
that share authors and reverses under the one fold spec §7.3 identifies as most
likely to expose idiom-fitting. This is the leg the rejection leans on least; it
is reported as ambiguous, not as a failure.

### C3 — "Argmax unstable across session or topic folds, by D1's identical-argmax bar" — **FAIL (triggers)**

Triggers. 23/25 session folds and 23/25 topic folds match the overall argmax.
D1's bar is *identical* — 60/60 and 6/6 — so 92% fails it. Note that under M-B
the topic result is not independent confirmation of the session result.

### C4 — "WER-30 moves the argmax outside the WER-0 plateau" — **NOT EVALUATED**

Not evaluated. The sweep was not run for the two reasons in the section above.
This criterion is neither satisfied nor violated, and nothing here should be
read as evidence either way. It does not need to be resolved for the verdict,
because C1 and C3 trigger outright and §7.2's basic gate fails at the proposed
value; but it remains genuinely open, and anyone who revisits `RHO` should run
it rather than assume it.

### Overall

Two criteria trigger outright. A third reverses out of sample under the author
fold. The value the handoff actually proposed fails §7.2's control gate before
any fold is considered. The fourth criterion is unevaluated and would not change
the outcome. **`RHO` is rejected.**

## Windows excluded as boundary cases

Spec §5.2 excludes a window whose off-topic fraction falls in `[0.4, 0.6]`
rather than forcing it to a label, and requires the count reported per category.

| category | windows | on-topic | off-topic | excluded |
|---|---:|---:|---:|---:|
| gradual_drift | 66 | 36 | 30 | 0 |
| adversarial | 73 | 46 | 26 | 1 (1.4%) |
| interleaved_abuse | 69 | 51 | 10 | **8 (11.6%)** |
| abstract_on_topic | 75 | 75 | 0 | 0 |
| depressed_baseline | 81 | 81 | 0 | 0 |
| **total** | **364** | **289** | **66** | **9** |

All but one exclusion comes from `interleaved_abuse`, for the reason given in
LIM-5.

## Findings about the mechanism and the method

These were found while executing the plan. They are reported because they change
what a future attempt should do, not as arguments for the verdict.

### M-A — spec §7.1's "above-threshold" safeguard is a provable no-op

§7.1 specifies that a window updates `R` "only if that window was at or above
the current `thr`, so a digression can never raise the ceiling it is judged
against", and states that the variant where any window may update `R` "is a
different mechanism and would need its own fit". **For any `RHO < 1` the two are
mathematically identical.**

`thr = RHO*R` with `RHO < 1` implies `thr < R`. A window can only raise `R` if
`sim >= R`, and `sim >= R > thr`, so such a window always passes the gate
anyway. A window with `thr <= sim < R` passes the gate but leaves
`max(R, sim) = R`, so `R` is unchanged. In both directions the guard never
alters `R`.

Verified by brute force over **20,000 random sessions** across the whole grid:
**zero differences** between the guarded and unguarded implementations (an
out-of-band check run during execution; it is not a field in the results JSON). `R` is
therefore simply the running maximum of all window cosines, floored at `UPPER`
and capped at 0.80, and the distinction §7.1 treats as load-bearing does not
exist anywhere in the range the grid covers. This is a correction to the spec's
reasoning, not a defect in the implementation — the harness implements §7.1 as
written.

**If the mechanism is ever revisited, redesign this first.** The safeguard is
the part of §7.1 the spec argues hardest for, and it buys nothing: whatever
protection was intended against a digression raising its own ceiling is not
present, at any swept `RHO`. A successor mechanism has to get that property
from somewhere else — gating the update on the *gold* or predicted label, on a
ceiling that decays, or on an `R` that is not a plain running maximum — or drop
the claim. Re-running the grid with the guard as written will keep reproducing
the unguarded result.

### M-B — there are two fold families, not three

Leave-one-topic-out is **arithmetically identical** to leave-one-session-out on
this corpus, because 25 distinct topics over 25 sessions makes every topic fold
a singleton — the same held-out set by a different name. Their agreement in
every table above is not independent evidence, and must not be counted twice.

### M-C — cross-family argmax agreement is structural, not replication

The overall grid is computed over all sessions regardless of fold family, so the
*overall* argmax cannot vary between families by construction. That the three
blocks report the same 0.26 is an identity, not corroboration.

### M-D — the argmax is objective-dependent

The committed primary objective (informedness, false-accusation tie-break) picks
**0.26**. The **committed secondary** (`asymmetric_cost`), plus accuracy and
digression-class F1, all pick **0.20**.

| objective | argmax |
|---|---:|
| informedness (primary, committed) | 0.26 |
| `asymmetric_cost` (secondary, committed) | 0.20 |
| accuracy | 0.20 |
| digression-class F1 | 0.20 |
| weighted informedness, w = 0.25 | 0.78 |
| weighted informedness, w = 0.50 | 0.36 |
| weighted informedness, w = 1.00 | 0.26 |
| weighted informedness, w = 2.00 | 0.20 |
| weighted informedness, w = 3.00 | 0.20 |

The weight `w` (equivalently `K`) scales the false-digression side of
informedness, not the caught-digression side — larger `w` penalises false
accusations more heavily. `asymmetric_cost`, the committed secondary, is the
**same measure** as `weighted_informedness_w2.00`: both weight false
digressions by a factor of 2, so their agreement on 0.20 is one result read
twice, not two independent objectives corroborating each other.

Across weightings the argmax lands on 0.20 / 0.26 / 0.36 / 0.78. Any objective
more averse to false accusations than plain informedness — which is D2's stated
bias — pulls it to the bottom of the grid. There is no objective-independent
"the" value of `RHO` on this corpus.

### M-E — the entire net gain comes from the 10 real-data sessions

Split by authorship, against the flat control:

| half | sessions | caught delta | false-digression delta |
|---|---:|---:|---:|
| agent-authored | 15 | +2 | **+1** |
| real data (AnnoMI) | 10 | 0 | **-14** |

On the agent-authored half the relative threshold catches 2 more digressions and
produces **one extra** false digression — no improvement. The whole -14
false-digression improvement comes from the 10 AnnoMI sessions. Those 10 are
exactly the `abstract_on_topic` and `depressed_baseline` categories, so this and
the per-category split describe the same windows from two directions. The
apparent gain tracks which sessions the constant was fitted on, not a property
of the mechanism, which is the same thing C1 detects from the fold side.

The agent-authored false-digression delta (+1) is a 1-window difference (4
against flat's 3), also within single-window resolution and carrying no
uncertainty estimate. It is reported here as descriptive, not as a leg the
verdict depends on.

## Two spec corrections

Recorded as corrections. Both specs were written before this evidence existed.

- **Polly access is granted; S3 staging is the real blocker.** Spec §6.2 calls
  the Polly IAM permission "the gate", and
  `2026-08-11-eval-dataset-design.md` records it as never added. It is in fact
  available: `describe_voices` succeeds and returns 14 `en-US` voices for IAM
  user `mgmt-tool-admin`. The actual blocker is S3 staging — the bucket
  `core-barter-audio-tmp` that `transcribe.py` stages audio in belongs to
  another AWS account (403 / `AccessDenied` on `get_bucket_location`) — and the
  spec does not mention it at all.
- **Spec §5.4's topic-pool claim cannot hold.** §5.4 says all 25 sessions take
  distinct topics "from the 40-topic pool". That is not possible: the 10 AnnoMI
  scripts carry AnnoMI's own declared topics, which are deliberately *not* in
  the pool, and spec §8 relies on exactly that third-party provenance for the
  two categories that make `RHO` falsifiable. The corpus satisfies the correct,
  weaker invariant: **25 distinct topics corpus-wide, pool topics for the
  blinded 15 only.**

## Limits, stated plainly

### Reproduced from spec §8, in full

- **Labels are structural**, derived from author intent, not human annotation.
  No kappa exists for this corpus. Mitigating factor: for `abstract_on_topic`
  and `depressed_baseline` — the two categories that make `RHO` falsifiable —
  the topic labels are **third-party**, assigned by AnnoMI's authors with no
  knowledge of this mechanism.
- **Prosody is synthetic throughout.** AnnoMI scripts are excerpted human text
  re-voiced through Polly, so the corpus tests ASR-on-synthetic-speech, not
  ASR-on-human-speech. The condition is uniform across the `AUTHOR` fold, which
  is what matters for attributing differences, but it is not real audio.
- **AnnoMI is counselling, not teaching.** A therapist reflecting is
  structurally unlike a teacher explaining. It is the closest real match to
  `abstract_on_topic` that exists and it is still an approximation.
- **AnnoMI has no licence file** — no grant at all, not merely a messy one, and
  its transcripts derive from third-party video. Fetch-don't-redistribute is a
  mitigation, not a resolution. **Action item: contact the authors.**
- Three of five categories remain LLM-authored. Blinding and model rotation
  reduce, but do not eliminate, synthetic-idiom bias.
- 25 sessions is small. It is roughly 5x the evidence the handoff rests on and
  well short of the 40-session research dataset.

### Additional limits found during execution

- **LIM-1 — the `abstract_on_topic` splice seams are all the easiest shape.**
  All four seams are a full session restart: record 1 ends on a closing line and
  record 2 opens greeting a newly *named* client ("Hi, Sandra.", "Hi, Mrs.
  Smith."). Spec §5.5 warns that an abrupt cut is easier for a classifier than
  real drift; "new named client" is a lexical giveaway on top of that. Any
  result that depends on catching these seams reads easier than genuine
  mid-conversation drift.
- **LIM-2 — §5.4's low-vocabulary-overlap justification does not hold
  uniformly.** §5.4 justifies AnnoMI on the grounds that its content is on-topic
  while sharing almost no vocabulary with its topic label. `sess_CAL16`
  ("smoking cessation") has median cosine **0.422**, with **11 of 13** windows
  already at or above `UPPER` — it contributes only 2 false digressions and can
  barely be false-flagged at all. Only `sess_CAL18` (median 0.045) and
  `sess_CAL19` (0.130) genuinely exercise the low-overlap case, and together
  they supply 19 of the category's 25 false flags. The headline
  false-accusation number is flattered by 2–3 of its 5 sessions.
- **LIM-3 — `sess_CAL23`'s topic string is probably an upstream typo.**
  "avoiding DOI" is AnnoMI's own label, carried verbatim under the
  never-invent rule; it is almost certainly their typo for DUI.
- **LIM-4 — the therapist→teacher mapping is unevenly teaching-like.**
  `sess_CAL21` (diabetes) and `sess_CAL16` (smoking) read genuinely
  instructional; `sess_CAL19` (gambling) and `sess_CAL18` (probation) read as
  reflective counselling with almost no explaining. This confirms §8's
  counselling-not-teaching limit in practice and adds that it is uneven rather
  than uniform across the 10 sessions.
- **LIM-5 — the digression signal is unbalanced and `interleaved_abuse` barely
  contributes.** Of 66 digression windows corpus-wide, `gradual_drift` supplies
  30, `adversarial` 26, and `interleaved_abuse` only **10** — while carrying
  **8 of the 9** total exclusions (11.6% of its windows). Both facts share one
  cause: its alternating digressions are short relative to a 25-second window,
  so most off-topic turns are either absorbed into majority-on-topic windows or
  straddle the `[0.4, 0.6]` exclusion band. This is exactly what §5.2 said to
  watch for. The category exists to test alternating drift and contributes the
  least evidence of any drift category.
- **LIM-6 — the blinded corpus still shows a shared LLM attractor.** All five
  `gradual_drift` scripts used the same single-pivot device across all three
  author models; a coffee/espresso tangent recurs in 3 of 15 scripts; and a
  *Free Solo* documentary tangent appears near-verbatim in 2 of 15 by different
  models. Blinding and rotation reduce but do not eliminate synthetic-idiom
  convergence.
- **LIM-7 — one digression sits semantically adjacent to another session's
  topic.** `sess_CAL09`'s digression subject is chess positional theory, while
  `sess_CAL13`'s declared *topic* is "Chess strategy (advanced)". Within-script
  labelling is correct in both, and the per-session design contains it, but a
  negative-labelled digression sitting next to another session's
  positive-labelled teaching is a noise source for any cross-script
  aggregation.
- **LIM-8 — the `_experimental` secondary checks in `test_conformance_oracle.py`
  are not uniformly discriminating.** Per R14, the conformance oracle's
  `test_experimental_span_means` checks were left with a wide ±0.02 tolerance
  rather than tightened, on condition that this weakness be carried into the
  findings' limits — this bullet does that. `case_B2_late` (expected 0.059,
  actual 0.0594) is a genuinely tight match and real evidence the harness
  reproduces the oracle. `adv3_long` (expected 0.008, actual 0.0045) is not: a
  ±0.02 bound around 0.008 admits roughly `[-0.012, 0.028]`, i.e. essentially
  any low-similarity span, so the ~44% relative miss still passes and the
  passing test should not be counted as evidence for that span that it cannot
  supply.

## Consequence for handoff §6 step 1

Handoff §6 lists the relative threshold `thr = RHO·R` as step 1 of the
recommended sequencing, on the grounds that "it survived every test, degrades to
current behaviour where evidence is absent, and everything else depends on it".
**Step 1 is now closed as rejected, not blocked.** The mechanism did not survive
the tests this corpus was built to run: at the proposed `RHO = 0.45` it loses to
the flat threshold already shipped, no value of `RHO` is stable across author
folds, and what advantage the best-fitting value shows in-sample neither
survives holdout nor appears at all on the agent-authored half of the corpus.
The dependency that made step 1 first in the sequence is therefore also
dissolved rather than satisfied: the three services §6 would have built on this
constant should not be built on it. Steps 2 and 3 remain independently
worthwhile — the corpus from step 2 now exists and is what produced this result,
and step 3 (span-local ladder, `severe` no longer vetoing payout) never depended
on `RHO` and still removes the escrow-loss failure without needing to tell
analogy from drift. Anyone who wants to revisit a relative threshold should
start from a corpus with more than four author strata and real-audio durations,
and should treat M-A, M-D and C4 as open work rather than settled.

Per spec §11, this documented rejection is a successful completion of the
project.

## Reproduce

From the repository root, with the semantic service dependencies available
(system `python3` has no `sentence_transformers`; use the venv below):

```bash
cd apps/semantic_analysis && ./venv/bin/python -m ground_truth.rho_calibration \
  --durations synthetic \
  --output ground_truth/rho_calibration_results.json
```

The harness reads the corpus from `eval_dataset/scripts/calibration/*/sess_*.txt`
and writes every table above plus per-fold, per-session, per-topic,
per-category, per-author and per-objective detail. `--all-wer` is accepted and
refuses with the reason; `--durations` accepts only `synthetic`, because no
timed transcripts exist.

## Verification

```bash
python3 -m pytest apps/semantic_analysis/ground_truth/test_rho_calibration.py -q
```

Tests cover `R`'s floor at `UPPER`, its cap at 0.80, that a below-threshold
window cannot raise it and that it never decreases within a session; the
two-decimal 0.20–0.80 grid; excluded windows counting in neither class; the flat
control being reported alongside every `RHO`; per-fold argmax fitted on training
sessions only; each session pooled exactly once; the objective-sensitivity
argmaxes; refusal of `durations="timed"`; and `align_wer_runs()` aligning by
window index and reporting count mismatches.

The STT-side helper is tested separately:

```bash
python3 -m pytest eval_dataset/tools/tests/test_run_stt_sweep.py -q
```
