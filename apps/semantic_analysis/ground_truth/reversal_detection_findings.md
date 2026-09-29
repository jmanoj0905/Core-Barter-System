# Meaning-reversal detection: evaluation on a synthetic corpus

Date: 2026-09-29

## Result

On an 80-sentence synthetic corpus, the lexicon-and-proximity detector
(`detect_meaning_reversal` in `apps/semantic_analysis/reversal_detection.py`)
has **precision 0.406, recall 0.650, F1 0.500** (tp 13, fp 19, fn 7, tn 41).
It never fires on plain commitments (20/20 correct) and catches 13 of 20
straightforward reversals, but it is **wrong on 17 of 20 double-negation
sentences**, and those account for 17 of its 19 false positives. Precision is
poor on this corpus because double negation was deliberately over-represented
(a quarter of the records); it is not an estimate of precision on real speech.

This is a descriptive measurement of a small heuristic, not a validation. No
threshold or lexicon was tuned on this corpus.

## Data and method

- Input: `reversal_eval_corpus.json`, 80 records, 20 per category:
  `commitment` (on-topic, no negation, label: no reversal), `reversal`
  (meaning-reversed commitment, label: reversal), `distractor` (negation
  present but unrelated to, or far from, a commitment; label: no reversal),
  `double_negation` (stacked negations that affirm the commitment; label: no
  reversal).
- Provenance: **synthetic and LLM-generated**, written by a separate agent that
  had not seen the detector code, labeled by true meaning. Not drawn from real
  sessions, not human-reviewed, not transcribed speech (punctuation and
  capitalization are clean; real STT output is not).
- No embedding model is involved; the detector is a pure text function. The
  corpus was not edited after seeing scores.
- Precision and recall default to 1.0 when their denominator is zero. That did
  not happen here (tp + fp = 32, tp + fn = 20), but the script prints a note if
  it ever does so a vacuous 1.0 is not read as a result.

## Printed output

Exactly as printed by
`cd apps/semantic_analysis/ground_truth && ../venv/bin/python evaluate_reversal_detection.py`:

```text
Corpus: 80 records (synthetic, LLM-generated; not real session data)
tp=13 fp=19 fn=7 tn=41
precision=0.406 recall=0.650 f1=0.500

category          correct  total  accuracy
commitment             20     20    100.0%
reversal               13     20     65.0%
distractor             18     20     90.0%
double_negation         3     20     15.0%

double_negation is a known, expected-imperfect category: the detector has no
double-negation handling, so it is reported to measure the limitation, not as
a target for 100% accuracy.

Misclassified (26):
[distractor]
  false positive: I'm not sure yet, but I'll pay you Friday.
  false positive: I can't wait to give you your first guitar lesson.
[double_negation]
  false positive: I would never refuse to pay you for good tutoring.
  false positive: I can't not give you credit for all the coding help you've provided.
  false positive: It's not that I won't deliver the lessons; I absolutely will.
  false positive: I'm not unable to provide the notes, they'll be in your inbox tonight.
  false positive: I wouldn't refuse to transfer the credits once the session is done.
  false positive: I never said I wouldn't refund your deposit if the lesson fell through.
  false positive: I couldn't not honor our agreement after everything you've done for me.
  false positive: I would never say I can't pay you what we agreed.
  false positive: I'm not going to refuse to fulfill my side of the swap.
  false positive: I refuse to not send you the recordings; you earned them.
  false positive: It isn’t true that I can’t return the money; I can and I will.
  false positive: I won't deny that I owe you three lessons, and I'll repay them soon.
  false positive: I'd never be unable to accept a fair offer like yours.
  false positive: I'm not refusing to commit to the full four-week tutoring plan.
  false positive: I wouldn't never guarantee the quality of my lessons, I always do.
  false positive: You won't find me refusing to deliver what I promised.
  false positive: I couldn't possibly not pay you for the photography lessons.
[reversal]
  false negative: I never delivered the design and I'm not going to.
  false negative: I don't owe you anything for those Spanish sessions.
  false negative: I'm backing out of the deal, so there will be no payment and no lessons.
  false negative: Forget what I said earlier. I'm walking away from the swap entirely.
  false negative: I never promised to tutor your daughter, you must have misheard me.
  false negative: I'm keeping the money I owe you; you can go ahead and complain.
  false negative: I wasn't paying for something that bad, so I'm withholding the credits.
```

## Failure modes

**Double negation (17 false positives, 3/20 correct).** The detector has no
scope handling, so any negation cue within four tokens of a commitment word
fires. This is the documented, pinned limitation, and it is the single largest
error source. It is also broader than "not refuse to": it covers "never say I
can't", "couldn't not honor", "not unable to", "won't deny that I owe", and
"never said I wouldn't refund". The 3 correct ones are correct by accident, not because negation was
resolved: "no way I'm refusing to complete" and "can't say no to that deal"
use cues or verb forms outside the lexicon ("no way", "refusing", "say no"),
and "never refused to teach you piano" uses the inflected "refused"/"teach",
neither of which matches.

**Missed reversals (7 false negatives, all in `reversal`).**
- Unlisted cues: "don't" ("I don't owe you anything") is not in
  `NEGATION_CUES`; neither are "wasn't", "no payment", "backing out",
  "walking away", "forget what I said", "withholding".
- Backing-out phrasing with no negation word at all ("I'm walking away from the
  swap entirely", "I'm keeping the money I owe you").
- Inflections: lexicon entries are exact tokens, so "delivered" and "paying"
  do not match "deliver" / "pay" ("I never delivered the design and I'm not
  going to", "I wasn't paying for something that bad").
- More inflections: "I never promised to tutor your daughter" is missed
  because "promised" does not match "promise".

**Distractor false positives (2 of 20).** Proximity traps: "I'm not sure yet,
but I'll pay you Friday" (the "not" scopes "sure", but "pay" is within four
tokens) and "I can't wait to give you your first guitar lesson" ("can't" is a
cue that here expresses enthusiasm). The other 18 distractors, where the
negation was far from or unrelated to the commitment, were correct.

Every misclassified text, grouped by category, is listed in the printed output
above.

## Labeling caveats

Labels were not changed. Two are debatable: "I wouldn't never guarantee the
quality of my lessons, I always do" is a double negation only under
standard-English logic (in some dialects it is an emphatic single negation),
and "I wasn't paying for something that bad, so I'm withholding the credits"
is past tense and could be read as a statement about the past rather than a
reversal of a future commitment. Neither changes the conclusions: the first is
one of 20 double-negation records and the second is one of 7 misses.

## What this does and does not show

- **Shows:** the exact failure modes of the exact-token lexicon plus
  four-token proximity rule on hand-constructed clean text, including that the
  double-negation limitation is real and large when double negation occurs.
- **Does not show:** the false-positive rate on real barter conversation, where
  double negation is presumably rare (the 25% share here is by construction);
  behavior on noisy STT output; recall on real reversals, which are likely more
  varied than 20 LLM-written ones; or any effect on verdicts or warnings.
  Confidence intervals are not reported and would be wide at n = 20 per
  category.
- **Next input needed:** real, human-labeled session windows. As with the
  semantic thresholds, do not use an LLM's suggested labels as the gold
  standard for that step.

## Reproduce

```bash
cd apps/semantic_analysis/ground_truth && ../venv/bin/python evaluate_reversal_detection.py
cd apps/semantic_analysis && PYTHONPATH=. venv/bin/pytest -q tests/test_evaluate_reversal_detection.py
```
