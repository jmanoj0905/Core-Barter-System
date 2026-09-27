# Handoff — separating pedagogical analogies from topic drift

**Status: NOT an approved spec. This is a design handoff.** Two options survived a
design investigation on 2026-09-26/27; the user liked both and wants them carried
into a fresh session. Nothing here has been implemented. No approval has been
given to write code.

**Read first:** `apps/semantic_analysis/ground_truth/analogy_handling_findings.md`
— the instrumented-run evidence, with every per-window number. This document
assumes it.

---

## 1. The problem, in one paragraph

A teacher explaining a concept by analogy speaks, for the duration of the
analogy, in vocabulary that shares almost nothing with the contract topic. The
topic-adherence classifier compares each ~25 s window's embedding against the
topic embedding by cosine, so a sustained analogy looks exactly like going
off-topic. Short analogies slip under the radar; long ones get flagged and can
cost the teacher their escrow. The user's specific case to protect: **an analogy
lasting several windows while the learner keeps asking engaged questions.**

## 2. Why this matters — the punishment chain

Traced through the live code, not assumed:

| Step | Location | Effect |
|---|---|---|
| `classify()` | `apps/semantic_analysis/main.py:225` | `>=0.36 correct`, `0.14–0.36 weakly_correct`, `<0.14 incorrect` |
| escalation | `apps/warning_engine/main.py:178` `run_warning_decision` | on `consecutive_incorrect`: 1 silent, 2 strong, **>=3 severe**. ANY non-incorrect window resets the counter to 0 |
| topic quality | `apps/backend/app/routes.py:132` `_evaluate_topic_quality` | `weakly_correct` counts as ON-topic; `on_topic% = (correct + weakly_correct) / total` |
| verdict | `apps/backend/app/routes.py:160` `_decide_verdict_type` | `on_topic% < 40` **OR any `severe` warning** → `DISPUTE`. `< 70` → capped at `PARTIAL` |
| settlement | `apps/backend/app/escrow.py:245` `apply_settlement` | `DISPUTE` → `qa_score 0.0` → full escrow penalty + trust hit |

**`severe` is an absolute veto on the teacher's payout.** That is the failure
mode everything below is designed against.

## 3. What was measured

Two instrumented end-to-end runs (real services over HTTP, no mocks) plus a
nine-script probe corpus. Durations synthesised at 150 wpm; `scope=""`; no STT.

### 3.1 The two runs

| | `sess_C05` (2-window analogy) | `sess_B08` (5-window analogy) |
|---|---|---|
| Analogy window cosines | 0.172, 0.155 | 0.179, 0.130, 0.197, 0.008, 0.029 |
| `incorrect` windows | **0** | **3** (25.0%) |
| Warnings | **0** | **1 — STRONG** |
| Escalation reached | none | strong, **not** severe |

`sess_B08` escaped `severe` → `DISPUTE` → `qa_score 0.0` **only** because one
mid-analogy window (0.197) was not `incorrect` and reset the counter. **A 0.057
cosine difference separated "mild scolding" from "lose the escrow."**

### 3.2 The nine-script corpus — seven candidate signals

The decisive control is **ADV-3**: ten windows of sustained *single-domain*
off-topic speech (a football season, told properly, recurring cast, no mapping),
length-matched to B2's eleven-window analogy on the same topic.

| Signal | B2 (real analogy) | ADV3 (pure off-topic) | Result |
|---|---|---|---|
| mean cosine | 0.059 | 0.008 | gap 0.051 — below the 0.057 already judged too fragile |
| window-to-window coherence | 0.406 | **0.453** | **BACKWARDS** |
| vocabulary turnover | 0.046 | 0.042 | flat |
| entity-recurrence share | 0.38 | **0.42** | **BACKWARDS** |
| domain clusters | 1 | 1 | flat |
| relational isomorphism (Gentner) | 0.189 | **0.223** | **BACKWARDS** |
| substitution test | −0.049 | −0.047 | flat |

**Conclusion: no cheap signal separates a sustained analogy from sustained
off-topic speech.** Four signals flat, three favour the adversary — a well-told
football story is *more* internally coherent than a working analogy.

Structural isomorphism was implemented properly (difference vectors over
recurring entity pairs) and recovers nonsense: `sess_B08`'s best alignment is
`wine:mentor ~ machine:training`, not the real mapping. Gentner is likely right
about what an analogy *is*; it is not recoverable from MiniLM's 384-dim space.
Caveat on that test: ADV1/ADV3's opening window scored above threshold and leaked
football vocabulary into the topic-entity set, so the absolute S6 numbers are not
a clean test — but the B2-vs-A comparison is backwards regardless.

### 3.3 The two results that DID hold

**(a) Session-relative thresholds.** The absolute on-topic/analogy gap differs by
2× between sessions; the ratio does not.

| | on-topic range | mean | vehicle mean | ratio |
|---|---|---|---|---|
| C05 | 0.373–0.784 | 0.549 | 0.164 | **0.30** |
| B08 | 0.191–0.453 | 0.308 | 0.109 | **0.35** |

So `thr = RHO · R`, `R` = running max of above-threshold cosines, floored at
`UPPER`, capped 0.80, `RHO = 0.45`. Where a session shows no better ceiling it
floors at 0.162 — within 0.022 of today's `LOWER` — so **it degrades to current
behaviour exactly where there is no evidence to do better.** This component
survived every test and is a prerequisite for everything else.

**(b) Domain-cluster count catches *wandering* drift.** Wandering drift (kitchen
→ holiday → car → commute → station roof) yields **9 single-linkage clusters
across 10 windows**; every sustained span in the corpus yields **1**. No overlap,
large margin. It cleanly identifies `gradual_drift`. It does **not** identify
analogies — single-domain coherence is not evidence of analogy; ADV3 is the most
coherent span in the corpus.

### 3.4 Other measured failures worth not repeating

- **Learner comprehension trajectory** (topic-vocabulary rise, 1st vs 2nd half of
  learner turns): B08 +0.128, B1 +0.056, but **B2 −0.111**, C05 −0.104 — while
  pure drift A and C both score **+0.125**, because a drifting teacher's learner
  starts asking on-topic questions once teaching resumes. Useless as a drift
  signal. It does separate one thing: ADV1/ADV3 sit at **0.000** — a learner who
  was never taught produces no topic vocabulary. Keep as a "did anything land"
  verdict input.
- **Lexical binding mass is forgeable.** Case C — ten windows of kitchen chat
  plus six throwaway retrofit sentences — scores **280s** of bond, in among the
  real analogies (B2 330s, B1 390s, ADV3 20s). Fifteen retrofit sentences costs
  ~90 seconds of talking. Case A earned **180s of bond from pure noise**
  (`['large'] × ['law','second','eleven']`) with zero mapping.
- **A cumulative debt ladder punishes recovery.** In Case A the teacher returns
  and teaches at cosines 0.269/0.408/0.305/0.281 while the ladder holds them at
  STRONG for two more windows. Any ladder must be **span-local**, resetting when
  a window comes back above threshold.
- **`MASS_MIN = 0.50` is falsified.** To pass B1 (32%) the floor must sit below
  32%, which also passes A (33%) and C (38%). There is no single mass threshold
  that admits the protected case and rejects pure drift. Mass must be one term in
  a conjunction, never a scalar veto. Denominate in **teacher-speech seconds**:
  elapsed-seconds takes `sess_B08` from 50% to 44%, vetoing the protected case
  *because the learner asked 17 questions*.
- **Baseline for scale:** the current production system `DISPUTE`s all four
  long-form scripts (A, B1, B2, C) — `max_consecutive` 6–10, `severe` in every
  case, including both genuine analogies. It separates nothing at this length.

---

## 4. OPTION 1 — Two-layer architecture (live advisory + adjudicated verdict)

### 4.1 The idea in plain terms

Split the job by budget. **While people talk**, the system only nudges — it may
tap the teacher on the shoulder, and it may never touch credits. **When the
session ends**, an LLM reads each off-topic stretch with the whole transcript
available and decides whether it was a teaching analogy or genuine wandering;
*that* decision touches credits. The LLM must quote the transcript for every
claim, and the server verifies each quote exists.

The live problem (25 s, no hindsight) and the money problem (no latency budget,
full transcript) are different problems. §3.2 proves the live problem is
unsolvable; the money problem is not.

### 4.2 Live layer — advisory, span-scoped, financially inert

Allowed to cost the teacher **nothing but attention**. No row it writes may reach
`_decide_verdict_type`.

- `thr = 0.45 · R` (§3.3a) in `semantic_analysis`.
- Span-local debt in `warning_engine`; span = contiguous run of below-`thr`
  windows, reset on return above `thr`.
- **Single-domain span → nudge only**, with copy honest about the system's own
  ignorance: *"You've been on one tangent for 2m 20s. If it's an analogy, start
  tying it back."* Correct message to send when the system cannot tell, and
  useful to an analogist and a drifter alike.
- **Multi-domain span (clusters >= 3 over >= 3 windows) → recorded warning.** The
  only live claim the data licenses (§3.3b).
- `consecutive_incorrect` and the 1/2/3 ladder retire. `severe` stops being a
  payout veto.

### 4.3 Verdict layer — one adjudication per digression span

Runs in the existing session-end path (`drift-summary` → `confirm_session`).
**One LLM call per span, not per window** — typically 1–3 per session.

**Input per span:** contract topic and scope; the span's window texts in order;
2 windows before and up to 4 after (so a return is visible); every learner
utterance inside and just after the span; and the cheap features as *evidence to
be checked, not conclusions* — span seconds, domain-cluster count, candidate
`(anchor, vehicle)` pairs from the lexical pass.

**Output — an evidence extraction, not an opinion:**

```json
{ "span_type": "pedagogical_vehicle | off_topic | uncertain",
  "mapped_pairs": [ {"vehicle_entity": "", "topic_concept": "", "quote": ""} ],
  "return_quality": "substantive | token_retrofit | none",
  "vehicle_entities_total": 0,
  "teaching_delivered": 0.0,
  "confidence": 0.0,
  "rationale": "<=40 words" }
```

**The device that makes this safe for money:** every claimed mapping carries a
verbatim quote, and the server verifies each quote is a substring of the stored
transcript. Unverified pairs are discarded before scoring, and
`mapping_completeness = verified_pairs / vehicle_entities_total` is recomputed
server-side rather than trusted. **The judge is a checkable extractor, not an
oracle** — a hallucinated mapping fails verification and falls to abstention.

`return_quality` is the Case C discrimination no cheap signal could make:
bare assertions with no elaboration, no prior topic-bearing use of the vehicle,
and no learner uptake are `token_retrofit`; B2's four mapping turns are
`substantive`.

**Mapping to the verdict:**

| Adjudication | Effect |
|---|---|
| `pedagogical_vehicle` + `substantive`, confidence >= 0.6 | span seconds count **on-topic**; span's live warnings marked `retracted_at`; eligible for `SUCCESSFUL` |
| `off_topic`, or `return_quality` in {`none`, `token_retrofit`} | span seconds count off-topic; may trigger the `DISPUTE` veto |
| `uncertain`, confidence < 0.6, or >half the quotes unverified | **abstain** — seconds count off-topic in the percentage, but the span **may not** trigger `DISPUTE`; verdict capped at `PARTIAL` (qa_score 0.5), trust deltas damped 0.5× |
| judge unavailable / call fails | fall back to cheap features, cap at `PARTIAL`, never `DISPUTE` on topic grounds (extends the existing ISSUE-019 principle: missing evidence proves nothing) |

`_decide_verdict_type` keeps its shape; `has_severe_warning` leaves the veto,
replaced by `adjudicated_off_topic_seconds / teacher_seconds`.

**Abstention is the load-bearing choice.** `PARTIAL` is deliberately worse than a
correct `SUCCESSFUL` and much better than a wrong `DISPUTE` — the system's
uncertainty is paid for out of the middle of the range, not out of the teacher's
escrow.

### 4.4 Adversarial analysis

- **Prompt injection is the real exposure** — the transcript is attacker-authored
  text. Mitigations: fenced data channel, closed output schema, server-side
  recomputation of `mapping_completeness`, quote verification. Worst case an
  injection achieves is `pedagogical_vehicle` with zero verified pairs → scored
  as abstention.
- **Collusion:** to earn `substantive`, a pair must produce a quotable
  per-entity mapping of vehicle onto topic — several minutes of genuine
  explanation, which *is* the lesson. Case C's retrofit costs six sentences and
  fails.
- **Residual, probably irreducible:** a mapping that is plausible but
  pedagogically worthless.

### 4.5 Cost

Live layer: zero new model calls beyond the embedding already computed; two
token-count dicts and a ledger per session. Judge: B2's span is ~900 words →
~2.5k input / ~300 output tokens per span, 1–3 spans per session, well under a
cent. Real costs are operational: a new external dependency in the settlement
path (must be queued and retried without blocking escrow) and an audit
obligation — store the judge's JSON plus verified quotes in `drift_summary` so a
disputed verdict can be re-examined by a human.

### 4.6 Why pick Option 1

- It is the only option that answers the actual question. It is the only
  mechanism that separates Case C's six retrofit sentences from B2's real
  mapping, and §3.2 shows no cheap signal can.
- It fixes both scenarios: the short analogy (recorded, not punished) and the
  long one (adjudicated, retractable).
- It puts uncertainty cost in the middle of the score range instead of on the
  teacher's escrow.
- The quote-verification design makes an LLM's output checkable rather than
  trusted — the judge cannot invent a mapping that survives the server.

### 4.7 Why NOT pick Option 1

- It puts an external model call in the path that moves real credits: new failure
  mode in the most sensitive part of the system, needing queue, retry, timeout
  policy and audit trail.
- **The judge is the one component with zero validation.** Everything else was
  measured across nine scripts; the judge has never been run once.
- Large surface: three services, a payload schema change, a new DB column
  (`retracted_at`), and a policy decision that whoever owns the credit rules must
  sign off — demoting `severe` from payout veto is not a technical call.
- Depends on `RHO = 0.45`, which Option 4 exists to make falsifiable.

---

## 5. OPTION 4 — Build the corpus first

### 5.1 The idea in plain terms

Build nothing yet. Nine scripts is not enough to trust, four were written by the
same agent that designed the mechanism they were used to test, and **both
instrumented runs bypassed speech-to-text entirely.** Write proper test material
first, run it through real STT, then calibrate.

### 5.2 Why the evidence is not yet sufficient

- **`RHO = 0.45` is unfalsifiable as things stand.** Without a genuinely clean
  session that naturally scores low throughout, a correct relative threshold
  cannot be distinguished from one that manufactures false digressions.
- **No STT, so no WER interaction.** Bindings and vehicle vocabulary are lexical;
  the rare distinctive nouns they depend on are exactly what a 20% WER transcript
  destroys. Entirely untested.
- **Single author, synthetic durations.** Durations come from word count at 150
  wpm. Under Option 1 those seconds would help determine escrow, so duration
  realism becomes financially load-bearing.
- **Nine parameters, none calibrated** (`RHO`, `K`, three ladder tiers,
  `MASS_MIN`, learner discount, learner cap, `R`'s cap) would replace two
  constants calibrated by grid search with leave-one-out and leave-one-topic-out
  folds (`ground_truth/design_decisions.md`, D1/D2). The repo's own standard
  would not accept the current evidence.
- The one result with margin enough to survive a thin corpus is the 9-vs-1
  domain-cluster separation. Everything else in §3.2 is a *negative* result —
  which is the one direction where a thin corpus is still informative: signals
  this flat on nine scripts will not sharpen on forty.

### 5.3 Categories to build

Existing categories (`clean`, `gradual_drift`, `adversarial`, `code_switch`,
`silence`) cannot validate any of this. Needed, with what each must contain:

1. **`depressed_baseline`** — a fully clean session on an abstract, low-overlap
   topic whose every window sits in 0.15–0.30 (`sess_B08`'s non-analogy prose is
   already close). **Load-bearing: without it `RHO` is unfalsifiable.** Confirms
   the mechanism does not manufacture debt in a session that simply embeds low.
2. **`elevated_baseline`** — the mirror: one exceptional opening window (~0.78)
   then ordinary windows near 0.37. Probes the +0.020 margin found at C05 W6 and
   `R`'s bootstrap, the weakest joint in the relative threshold.
3. **`abandoned_analogy`** — announced ("let me give you an analogy… I'll map it
   back"), sustained 4+ windows, **no return**, learner asks only vehicle-domain
   questions or goes silent. **Load-bearing.**
4. **`staged_analogy`** (collusion) — as above plus a learner producing one
   copular bridge per window using contract vocabulary; include variants where
   the *teacher* stages the bridges. **Load-bearing.**
5. **`interleaved_abuse`** — a pair alternating one on-topic window with one
   off-topic window at 60/40, 50/50, 40/60. Exists purely to calibrate where the
   mass policy call actually lives.
6. **`abstract_on_topic`** — >=10 windows of the `sess_B08` turn-24 kind:
   unambiguously on-topic prose scoring below `LOWER` (turn 24 scored **0.132**).
   Needed to measure the false-digression **rate**, which no artefact does today.
7. **`bonded_analogy`** — 3–6 window analogy with a dense explicit return (>=3
   vehicle→topic equations in one turn) and >=2 learner bridging questions.
   `sess_B08` is one; ~6 more across topics.
8. **`late_analogy`** — analogy in the final two windows so the return never
   arrives before session end. Probes whether settlement punishes a teacher who
   ran out of clock.

Plus: **multiple authors** (to break single-author idiom), and the existing
`synthesize` → `transcribe` → `wer_inject` pipeline so WER 10/20/30% variants
exist for every script.

### 5.4 Why pick Option 4

- It makes the two surviving results falsifiable instead of suggestive.
- It is a prerequisite for trusting Option 1's live layer, and the only way to
  calibrate nine parameters honestly.
- It matches the discipline the repo already applied to the current thresholds.
- STT interaction is a genuine unknown that could invalidate the lexical half of
  any binding mechanism.

### 5.5 Why NOT pick Option 4

- **It ships nothing.** The teacher-loses-escrow failure is live now, and Option 4
  does not remove it.
- Effort spent validating a mechanism not yet committed to.
- The one large-margin result (domain clusters) is arguably already safe to act on.

---

## 6. Recommended sequencing if both are wanted

1. **Relative threshold** (`thr = RHO·R`) alone — it survived every test,
   degrades to current behaviour where evidence is absent, and everything else
   depends on it.
2. **`depressed_baseline` + `elevated_baseline` + `abstract_on_topic`** — the
   three categories that make `RHO` falsifiable and measure the false-digression
   rate. Smallest slice of Option 4 that de-risks step 1.
3. **De-fang the live layer** — span-local ladder, `severe` stops vetoing payout.
   Removes the escrow-loss failure without needing to tell analogy from drift.
4. **`abandoned_analogy` + `staged_analogy` + `interleaved_abuse`** — the
   adversarial corpus the judge must be validated against.
5. **The judge**, last, once there is a corpus to test it on.

Steps 1 and 3 are independently shippable. The judge should not ship before
step 4 exists.

## 7. Open questions no evidence has settled

- Should the learner's *silence* during a digression add to debt? `sess_B08`'s
  engagement dipped to 0.50 during the analogy (learner share 7–9%) and recovered
  to 0.70. The signal exists, in a separate pipeline. Fusing engagement into
  drift is the obvious next question and there is no evidence either way.
- `MASS_MIN`'s value, and whether mass belongs in the veto at all given §3.4.
- Whether a colluding pair who genuinely performs a substantive ten-window
  mapping has cheated. Arguably they taught the lesson and should be paid.
- **Short spans are structurally blind.** C05's two-window analogy admits no span
  statistic at all (recurrence >=3 is arithmetically impossible). Below ~4
  windows the raw cosine is all there is.
- Whether the judge should ever grade mapping *quality*. Recommendation: no.

## 8. Artefacts in the repo

| Path | What |
|---|---|
| `apps/semantic_analysis/ground_truth/analogy_handling_findings.md` | the two instrumented runs, every per-window number |
| `apps/semantic_analysis/ground_truth/analogy_probes/` | measurement code (`sim*.py`, `corpus*.py`, `bridge.py`, `copula.py`, `baseline.py`, `an.py`) + both forensic reports |
| `eval_dataset/scripts/person_C/sess_C05.txt` | 2-window analogy (Node.js / restaurant kitchen) |
| `eval_dataset/scripts/person_B/sess_B08.txt` | 5-window analogy (ML intro / wine cellar) |
| `eval_dataset/scripts/_experimental/case_A.txt` | 10-window wandering drift, then legitimate return |
| `eval_dataset/scripts/_experimental/case_B1_incremental.txt` | 10-window analogy, tie-backs throughout |
| `eval_dataset/scripts/_experimental/case_B2_late.txt` | 10-window analogy, all mapping deferred to the end |
| `eval_dataset/scripts/_experimental/case_C_retrofit.txt` | Case A + 6 cheap retrofit sentences (the gaming probe) |
| `eval_dataset/scripts/_experimental/adv_never_return.txt` | announce-an-analogy, never return (4 windows) |
| `eval_dataset/scripts/_experimental/adv_collude.txt` | as above + colluding learner bridges |
| `eval_dataset/scripts/_experimental/adv3_long.txt` | **the decisive control** — 11 windows sustained single-domain off-topic |

All seven `_experimental` scripts parse under
`eval_dataset.tools.script_parser`. They are probes, not topic-pool sessions —
hence the `_experimental` directory rather than a `person_X` slot.

Reproduce a run: start `warning_engine` (8003) and `semantic_analysis` (8002)
from their venvs, `POST /session/{id}/contract`, post each turn to
`/ingest/segment` (teacher `user_id=1`, learner `user_id=2`, `duration = words ×
0.4`), then `POST /session/{id}/end`. `analogy_probes/` has working drivers.

## 9. State of the decision

The user reviewed four options (full two-layer, live-layer-only,
relative-threshold-only, corpus-first) and chose to carry **Option 1 and
Option 4** forward. Neither is approved for implementation. The next step is a
design discussion in a fresh session, then a real spec, then a plan — not code.
