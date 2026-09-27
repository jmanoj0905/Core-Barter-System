# Analogy corpus & `RHO` calibration — design spec

> Date: 2026-09-27
> Status: Approved (design). Implementation not yet planned.
> Implements: Option 4 of `2026-09-27-analogy-vs-drift-handoff.md`, scoped to
> the user's five categories.
> Precedes: handoff §6 step 1 (shipping `thr = RHO·R`), which is a separate
> task with its own approval.

## 1. Goal

Build evaluation material good enough to either **calibrate or reject**
`RHO = 0.45` in `thr = RHO · R`, to the standard `ground_truth/design_decisions.md`
applied to D1/D2 — grid search, multiple fold families, an error-mix table, and
a false-accusation rate reported separately from accuracy.

The handoff's §5.2 complaint is the brief: nine scripts is not enough to trust,
four were written by the same agent that designed the mechanism they test, and
both instrumented runs bypassed speech-to-text entirely.

**This project ships no mechanism code.** Its deliverable is a corpus, a
measurement harness, and a findings document containing a calibrated `RHO` or a
reasoned rejection of it.

## 2. Scope

### 2.1 In scope

Five categories, as specified by the user:

| Category | Status | What it must contain |
|---|---|---|
| `gradual_drift` | exists in parser | teacher progressively stops teaching |
| `adversarial` | exists in parser | sustained coherent off-topic speech |
| `depressed_baseline` | NEW | fully clean session, abstract topic, every window embeds low |
| `abstract_on_topic` | NEW | >=10 windows unambiguously on-topic, below `LOWER` |
| `interleaved_abuse` | NEW | alternating on-topic / off-topic stretches |

Plus: multiple authors, the `synthesize` -> `transcribe` -> `wer_inject`
pipeline so WER is in the loop, and the calibration itself.

### 2.2 Out of scope, deliberately

- **No mechanism code ships.** `classify()` untouched; `thr = RHO·R` does not
  enter production; the 1/2/3 escalation ladder stands; `severe` keeps its
  payout veto. Handoff §6 steps 1 and 3 are separate tasks.
- No judge, no LLM adjudication, nothing from handoff Option 1.
- The three judge-validation categories — `abandoned_analogy`,
  `staged_analogy`, `bonded_analogy` — deferred to handoff §6 step 4. The judge
  is last and does not exist yet.
- SwDA as a source (see §5.4).
- The annotation app and `compute_kappa` — a consequence of label scheme A (§5.2).
- `warning_engine`, `backend`, escrow, `window_chunker.py`: untouched.

## 3. Decisions taken during design

| # | Decision | Rationale |
|---|---|---|
| A | Harness before corpus | The one existing windower measures the wrong unit twice over (§4.1). The handoff's recorded cosines give a free oracle. Authoring 25 scripts against an unverified harness is the expensive failure mode. |
| B | Blinded fresh-context subagents as authors | Single-author bias cannot be fixed by the same agent writing more scripts. Blinding is the part that addresses it; model rotation breaks idiom. |
| C | Excerpt and splice only; free rewrite banned | A cleaner agent that rewrites real dialogue launders LLM text through a real corpus and makes the provenance field lie. |
| D | Verbatim-substring enforcement | Makes "real" a property the repo can prove. Same device as Option 1's quote verification, applied to corpus construction: the cleaner is a checkable extractor, not an author. |
| E | Structural labels (scheme A), no human annotation | Chosen by the user over a spot-check hybrid. Every category is defined by construction, so labels are exact w.r.t. intent. Limits recorded in §8. |
| F | AnnoMI only for the real-data leg | It is the only verified corpus passing the teacher-seconds test (§5.4) whose content is genuinely on-topic-but-low-overlap. |
| G | Falsification criteria fixed before the first run | A grid search always returns a number. Without pre-registered rejection conditions, "calibrate `RHO`" cannot fail, and the handoff's stated purpose is to make it falsifiable. |

## 4. The measurement harness

### 4.1 What production actually does

Four properties no existing tool reproduces:

| Property | Location |
|---|---|
| Windows close at **25 s of accumulated teacher speech** | `apps/semantic_analysis/main.py:92,524` |
| Learner turns never enter the buffer (routed to engagement) | `main.py:537` |
| Window text is the **concatenation** of teacher segments, then `clean_text` | `main.py:388-389` |
| A window under `MIN_CONTENT_TOKENS` is **skipped**, not classified | `main.py:396` |
| Topic side embeds `f"{topic}. {scope}"` | `main.py:441` |

By contrast `eval_dataset/tools/window_chunker.py:12` slices at **5.0 s of
wall-clock audio**, including learner speech. Those are different units twice
over. Cosine magnitude is strongly length-dependent, and `R` — the running max
the whole of `RHO` depends on — is defined over production windows. Labels or
cosines collected on 5 s wall-clock chunks cannot calibrate this threshold.

`window_chunker.py` is left untouched; it serves the 5 s annotation path of
`2026-08-11-eval-dataset-design.md`. We simply do not use it.

### 4.2 Structure

**(a) `apps/semantic_analysis/windowing.py`** — extraction of pure logic already
in `main.py`: `clean_text`, `classify`, the accumulate-and-close rule, and topic
text construction. `main.py` imports it; behaviour changes by zero. Must stay
import-light (no `sentence_transformers` at module scope) so the harness can
import it outside Docker. This is the only production file this project touches,
and it is a move, not a rewrite.

**(b) `eval_dataset/tools/replay.py`** — parsed script + per-turn durations ->
production windows. Emits `{index, text, teacher_seconds_start, teacher_seconds_end, source_turns}`.
Two duration sources behind one interface:

- `synthetic` — `words x 0.4` (150 wpm), what the handoff used.
- `timed` — real word timestamps, joined to turns by timestamp containment
  against `{session_id}_timing.json` (see §6.1).

The same windower serves both, so WER-0 vs WER-30 differences are attributable
to the transcript and nothing else.

**(c) `eval_dataset/tools/embed_windows.py`** — windows -> cosines, loading
MiniLM directly. No HTTP, no running services: a full corpus sweep is a script,
not the multi-service procedure in handoff §8.

### 4.3 The conformance gate

Before any script is authored, `replay` + `embed_windows` must reproduce, from
the scripts alone at `synthetic` durations and `scope=""`:

| Session | Expected per-window cosines | Expected |
|---|---|---|
| `sess_B08` | 0.179, 0.130, 0.197, 0.008, 0.029 (analogy windows) | 3 `incorrect`, max 2 consecutive |
| `sess_C05` | 0.172, 0.155 (analogy windows) | 0 `incorrect`, max 0 consecutive |

Committed as `test_conformance_oracle.py`. Tolerance **±0.005** to absorb float
and MiniLM patch drift, tightened to ±0.001 if the first run comes in exact.
Secondary weaker checks against handoff §3.2's means (B2 0.059, ADV3 0.008)
across the seven `_experimental` scripts.

The harness runs **no `warning_engine`**, so the gate asserts cosines,
classifications, and the **maximum consecutive-`incorrect` run** — the input the
1/2/3 ladder consumes — rather than warning rows. `sess_B08`'s recorded STRONG
warning follows from a max run of 2 by the rule at
`apps/warning_engine/main.py:178`, and recomputing that arithmetic locally keeps
the gate a single-process test.

**If the gate fails, the corpus is not written.** That result would mean the
handoff's numbers are not reproducible, which is more important than anything
downstream of it.

## 5. Corpus construction

### 5.1 Taxonomy and script format

`script_parser.py:3` gains three categories: `depressed_baseline`,
`abstract_on_topic`, `interleaved_abuse`. Four new header fields, all optional
so the two existing topic-pool scripts keep parsing:

| Field | Purpose |
|---|---|
| `SCOPE:` | Production embeds `f"{topic}. {scope}"`. The handoff pinned `scope=""` silently; making it explicit lets us measure whether it matters. |
| `AUTHOR:` | `agent:opus-5`, `agent:sonnet-5`, `agent:haiku-4.5`, `real:annomi`. Enables leave-one-author-out (§7.3). |
| `DIGRESSION:` | Turn ranges the author declares off-topic. This is label scheme A. |
| `SOURCE:` | Real-data scripts only; points at the provenance sidecar. |

### 5.2 Structural labels

`DIGRESSION` names turns; `replay` maps turns to windows; a window's gold label
is set by the **majority of its teacher-seconds**.

Windows straddling a boundary near 50/50 are **excluded from calibration, not
guessed**. A boundary window has no honest label, and rounding it would place
noise exactly where the threshold decision lives. The excluded count is
reported: a large one means digressions are being authored too raggedly to
measure, which is a finding about the corpus, not a nuisance.

### 5.3 Blinding protocol

One fresh-context subagent per script. It receives:

- a topic from `eval_dataset/topics/topic_pool.md`
- the category described **behaviourally** ("the teacher progressively stops
  teaching and never comes back")
- the script format and a teacher-word-count target
- an instruction to declare its own `DIGRESSION` ranges

It never receives: the cosine mechanism, `RHO`, the thresholds, the handoff, the
fact that a classifier exists, or any other script. Model is rotated across
authors.

The prompt template is **committed** at `eval_dataset/tools/authoring/prompt_template.md`
so the blinding is auditable rather than asserted. If `RHO` later looks too
good, a reader can check what the authors were told.

### 5.4 Sizing and sources

**25 sessions** — 5 per category, on 25 distinct topics from the 40-topic pool.
Teacher-word target 1,000–1,400, giving 10–14 production windows each, so
roughly 275–350 labeled windows. For scale, D1 calibrated on 60 single sentences
across 6 topics; 25 distinct topics makes leave-one-topic-out a real fold.

**The teacher-seconds test.** Any real source must supply >=10 windows x 25 s =
**~250 s ~= 625 teacher words in one session**. This is the binding constraint,
not turn count, and it eliminates most dialogue corpora:

- **QATD** (already vendored at `eval_dataset/curated/qatd/`, CC-BY-NC-SA-4.0,
  1,576 real tutor/student interventions): median 31 turns but **median 5 words
  per message**, ~100 tutor words per session. Roughly one sixth of the
  requirement. **Rejected on length**, not provenance.
- **AnnoMI**: 133 human-transcribed two-party motivational-interviewing
  sessions, 10–25 min, declared topic per session, long reflective turns.
  **Passes.** Content is genuinely on-topic while sharing almost no vocabulary
  with its own topic label — precisely the `abstract_on_topic` case.
- **SwDA**: the only real source of naturally-occurring drift, but speakers are
  peers with no teach/learn asymmetry — the one structural property every
  category here depends on. Splicing peer chat into teacher/learner sessions
  would also mix authorship inside a single script and muddy the `AUTHOR` fold.
  **Deferred**, against the scout's ranking.

So the real-data leg is **AnnoMI only, serving `abstract_on_topic` and
`depressed_baseline`** — the two categories where real data helps and where
third-party topic labels de-risk scheme A. The three drift categories stay
blinded-agent work and are labeled as such via `AUTHOR`.

### 5.5 The real-data pipeline

Following `curated/qatd/` exactly: `eval_dataset/curated/annomi/` with a
gitignored `raw/`, a `source_manifest.json` pinning repository revision and
per-file sha256, vendored upstream licence and README, and a `build_dataset.py`
that fetches on demand. **Nothing third-party is redistributed.**

Extraction runs as a cleaner agent under the excerpt-and-splice budget
(decision C): it selects spans and never writes words. Per script it emits
`eval_dataset/provenance/<session>.provenance.json` recording source record IDs,
exact character spans, and every seam offset. `verify_provenance.py` asserts
**every dialogue line is a verbatim substring of the fetched source** and
rejects the script otherwise.

Splices must fall on natural speaker-turn boundaries, and every seam offset is
recorded, so the calibration can test whether any result depends on them. An
abrupt topic cut is easier for a classifier than a real drift; if results hinge
on seams, the corpus is optimistically easy and we will know.

## 6. The STT leg

### 6.1 Speaker attribution

`transcribe.py:16` emits `{"words": [{text, start, end}]}` with **no speaker
labels**, and the harness needs teacher-only accumulation. Diarization is not
required: we synthesize the audio, and `synthesize.py:36` already writes
`{session_id}_timing.json` with `{speaker, text, start, end}` per turn.
Attributing each transcribed word to a turn by timestamp containment is exact
and free; `replay`'s `timed` source consumes that join.

`wer_inject` preserves word timestamps through all three operations
(`wer_inject.py:50-60`: substitute keeps them, delete drops the word, insert
stamps the filler at `w["end"] + 0.2`), so the alignment survives corruption and
WER-30 windows remain attributable.

### 6.2 Pipeline and gate

`synthesize` -> `transcribe` -> `wer_inject`, per `eval_dataset/README.md`,
giving WER 0/10/20/30 per session. `.wav` is already LFS-tracked via
`eval_dataset/.gitattributes`. Cost for 25 sessions of ~8 minutes is a few
dollars, so cost is not a design constraint. The gate is the **Polly IAM
permission**, which `2026-08-11-eval-dataset-design.md` lists as never having
been added; it is verified before anything else in this section runs.

### 6.3 What the STT leg tests, stated narrowly

Handoff §5.2 worried that WER destroys the rare distinctive nouns lexical
bindings depend on. **This project builds no bindings** — `RHO` is pure cosine —
so that specific worry is not what is being tested here. Two things are:

1. **Does `RHO` survive ASR damage?** `R` is a running max over above-threshold
   cosines. WER lowers cosines unevenly; if it lowers the ceiling more than the
   floor, the ratio moves and `RHO` is more fragile than no-STT runs can show.
2. **Real durations change the windows themselves.** The handoff used
   `words x 0.4`. Polly's actual speech rate and Transcribe's actual timestamps
   will not agree with that, so window boundaries land elsewhere and the same
   script yields a different window sequence. Since handoff §3.2 turns on a
   0.057 cosine gap, whether that gap survives realistic windowing is exactly
   what a synthetic duration model can hide.

### 6.4 Determinism

AWS Transcribe is a hosted service and is not reproducible across runs, so
**WER-0 transcripts are committed** as artifacts rather than treated as
regenerable — the same reasoning that pins shas in `source_manifest.json`.
`wer_inject` is seeded and deterministic, so the 10/20/30 variants are
reproducible from the committed WER-0; the seed is recorded, the outputs are not
stored.

## 7. Calibration

### 7.1 Pinning the mechanism

The handoff defines `R` as "the running max of above-threshold cosines" where
the threshold is `RHO · R` — which is recursive. The reading specified here:

- `R` initialises at `UPPER` (0.36).
- A window's cosine updates `R` only if that window was **at or above the
  current `thr`**, so a digression can never raise the ceiling it is judged
  against.
- `R` is capped at 0.80 and never decreases within a session.

This is consistent with the handoff's stated degradation point
(0.45 x 0.36 = 0.162). It is a **choice, not a deduction**. An alternative in
which any window may update `R` is a different mechanism and would need its own
fit.

**Parameters in scope: three** — `RHO`, `R`'s cap, and `R`'s update rule above.
The handoff's other six belong to the ladder and mass policy, which this project
does not build.

### 7.2 The control

Every result is reported against **current production** — fixed `LOWER = 0.14` —
on the same windows. If `thr = RHO·R` does not beat a flat threshold there is
nothing to ship, and that comparison is the deliverable whichever way it lands.

### 7.3 Grid and folds

`RHO` swept 0.20–0.80 at 0.01.

The held-out unit is the **session**, not the window: `R` is running per-session
state, so holding out a single window is incoherent. Three fold families:

- leave-one-session-out (25 folds)
- leave-one-topic-out (25 folds, distinct topics by construction)
- **leave-one-author-out** (4 folds: three models plus `real:annomi`) — the fold
  single-author work could never run, and the one most likely to expose
  idiom-fitting

D1's stability bar was the argmax being *identical* across all folds. Same bar
here; fold-stability of the argmax is reported as a number, not asserted.

### 7.4 Scoring

On the **error mix**, not accuracy. D2's lesson was that accuracy ties across a
plateau while the error mix does not, and that false accusations are the
asymmetrically expensive error. Headline table is D2-shaped:

| `RHO` | Digression windows caught | **False digressions** | On-topic windows affected |

`RHO` is chosen from that table with D2's bias: give up a caught digression to
avoid a false accusation. The **false-digression rate on `abstract_on_topic` and
`depressed_baseline` is the headline number** — the thing handoff §5.2 says no
artefact measures today.

Robustness: the whole calibration re-run at WER 0/10/20/30, reporting how far
the argmax moves.

### 7.5 Falsification criteria (pre-registered)

`RHO` is **rejected** if any of the following holds:

1. The argmax differs across **author folds** — it is fitting one model's idiom,
   not conversational structure.
2. Its false-digression rate on the two on-topic categories is **no better than
   flat `LOWER = 0.14`** — the relative threshold buys nothing for its
   complexity.
3. The argmax is **unstable across session or topic folds**, by D1's
   identical-argmax bar.
4. **WER-30 moves the argmax outside the plateau** that is flat at WER-0 — it
   does not survive real ASR.

**A rejection is a successful outcome of this project.** It closes handoff §6
step 1 and saves building three services on a constant that does not hold.

## 8. Limits, stated in advance

To be repeated in the findings document, in the manner of D1's limits section:

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

## 9. Repo layout

```
apps/semantic_analysis/
  windowing.py                        NEW  pure logic extracted from main.py
  main.py                             MOD  imports it; zero behaviour change
  ground_truth/
    rho_calibration.py                NEW
    rho_calibration_results.json      NEW
    rho_calibration_findings.md       NEW

eval_dataset/
  tools/replay.py                     NEW
  tools/embed_windows.py              NEW
  tools/verify_provenance.py          NEW
  tools/script_parser.py              MOD  +3 categories, +4 header fields
  tools/authoring/prompt_template.md  NEW  committed blinding prompt
  tools/tests/test_replay.py          NEW
  tools/tests/test_embed_windows.py   NEW
  tools/tests/test_verify_provenance.py NEW
  tools/tests/test_conformance_oracle.py NEW
  scripts/calibration/<category>/     NEW  25 sessions
  curated/annomi/                     NEW  manifest + build_dataset.py, raw/ gitignored
  provenance/                         NEW  per-script sidecars
  audio/, transcripts/raw/                 populated; .wav via existing LFS rule
```

## 10. Execution order

1. Verify Polly IAM; extract `windowing.py`; build `replay` + `embed_windows`;
   **conformance gate** (§4.3). Hard stop on failure.
2. Extend `script_parser`; commit the blinding template; author 15 blinded
   scripts across the three drift categories.
3. Fetch AnnoMI; run the cleaner agent; `verify_provenance` must pass on every
   line; produce 10 real-data scripts.
4. STT sweep across all 25 sessions -> WER 0/10/20/30.
5. Calibrate; write findings against §7.5.

TDD throughout, matching the repo: every existing tool in `eval_dataset/tools/`
has a test beside it.

**Two notes on the plan's shape.** Step 1 can invalidate everything after it, by
design — it is the cheapest step and the likeliest to fail. Steps 2 and 3 are
the token-expensive parts: 25 authored sessions at 1,000–1,400 teacher words,
one fresh-context agent per script, plus a cleaner pass over AnnoMI.

## 11. Success criteria

- The conformance gate passes, or the discrepancy is documented and the project
  stops.
- 25 sessions exist across five categories and 25 distinct topics, each with
  WER 0/10/20/30 transcripts.
- Every real-data script passes verbatim-substring verification.
- `rho_calibration_findings.md` reports the D2-shaped error-mix table, all three
  fold families, the WER robustness table, and an explicit verdict against each
  of §7.5's four criteria.
- A calibrated `RHO` with stated limits, **or** a documented rejection. Both are
  successful completions.
