# Window labeling rubric

Date: 2026-09-26

Applies to `human_label` on `window_results`, written by the "rate this" control
in `LiveSession.jsx` and exported by `export_real_labels.py`. One label per
window: roughly 25 seconds of accumulated teacher speech, judged against the
session contract's `topic` **and** `scope`.

This replaces the informal descriptions in `topics.py`'s docstring. The old
wording ("squarely inside the scope", "adjacent/tangential") gave two reviewers
no way to converge on the same answer, and the `correct` / `weakly_correct` line
was the dominant error mode in every experiment run so far.

## Label the scope, not the topic

The contract has two parts and the scope is the operative one. "Beginner
acoustic guitar chords" with scope "open chord shapes, transitions, and basic
strumming patterns" does **not** cover restringing the instrument, even though
restringing is obviously about guitars. Judge each window against the scope
clause; fall back to the topic only when the scope is empty.

## Which boundary actually matters

Get the **weakly_correct / incorrect** line right above all else. That is the
only boundary the product acts on: `warning_engine` escalates on *consecutive*
`incorrect` windows and treats `correct` and `weakly_correct` identically, both
resetting the counter. A mislabeled `incorrect` manufactures a warning against
an honest teacher; a mislabeled `correct`-vs-`weakly_correct` currently changes
nothing a participant can observe.

Spend your attention accordingly. If you are unsure between `correct` and
`weakly_correct`, pick `weakly_correct` and move on — it costs nothing today. If
you are unsure between `weakly_correct` and `incorrect`, stop and apply the test
below properly.

## The four labels

Apply these in order and take the first that fits.

**`incorrect` — no topical relation.** The window is not teaching this subject
at all: small talk, logistics ("can you hear me now", "let's wrap up at six"),
personal news, or an unrelated domain. Test: a listener who was told only the
topic would say the speaker has stopped covering it.

**`out_of_scope` — coherent teaching, wrong subject.** Real instructional
content about something the contract does not cover: teaching Java in a Python
session, or knife sharpening in a bread-baking session. This is *not* the same
as `incorrect` — the teacher is working, just not on the agreed thing — and not
the same as `weakly_correct`, which stays inside the topic. Exported labels with
this value are **excluded** from three-class calibration rather than remapped
(`export_real_labels.py`), because folding them into either neighbour would
corrupt the boundary being fitted.

**`weakly_correct` — on topic, outside the named scope.** Prerequisites,
adjacent sub-topics, tooling asides, or a digression the teacher could justify
if asked. Test: **one bridging sentence** connects it to something the scope
names. "Generator expressions use parentheses and are lazy" needs one sentence
to tie it back to list-comprehension syntax — weakly correct.

**`correct` — advances something the scope names.** No bridge needed; the
utterance is directly about a scope item. Test: you can point at the words in
the scope clause it belongs to.

## Tie-breaks

Written down so two reviewers resolve the same case the same way.

1. **Mixed windows: label the majority of the speech.** Twenty-five seconds
   often spans two thoughts. If one thought clearly dominates the window, label
   that. If it is genuinely half and half, take the *more* on-topic half — the
   teacher was doing the work for part of the window, and warnings should not
   fire on a window that contained real teaching.
2. **Answering a learner's off-topic question is `weakly_correct`, not
   `incorrect`,** provided the answer is still about the topic. The teacher is
   responding to the room, not drifting.
3. **Transcription garbage is not drift.** If the window is unintelligible
   because STT failed, skip it rather than rating it. A wrong label here teaches
   the model that noise means off-topic.
4. **Do not read the model's prediction before you label.** The UI shows the
   model's `classification` beside the window; anchoring on it makes the labels
   agree with the model rather than with the truth, which is worthless for
   calibration.
5. **When still torn, prefer the label that would not trigger a warning.** A
   false accusation costs a participant trust score and escrow; a missed
   off-topic window costs one window of evidence, and escalation needs several
   consecutive ones anyway.

## Scope counterfactuals

`topics.SCOPE_COUNTERFACTUALS` holds utterances labeled twice — once under each
of two scopes for the same topic — because unrelated small talk is too easy to
prove the model reads scope at all. `scope_sensitivity.py` scores them as a
paired, threshold-free probe. Current result: **11/12 (92%)** of pairs move
toward the labeled scope, mean margin `+0.10`, against 50% chance. These rows
are a diagnostic and are never used to fit thresholds.

## Before scaling collection

Have a second reviewer independently label an overlapping sample and measure
agreement on the `weakly_correct` / `incorrect` boundary specifically. If two
careful people disagree there, no model can be scored above that disagreement
rate, and the rubric needs another pass before more labels are worth collecting.
