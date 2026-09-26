# Research: alternatives to cosine-regression SBERT fine-tuning

Date: 2026-09-18

Follow-up, 2026-09-19: [threshold experiment results](./threshold_experiment_findings.md)
are available. On sentence-held-out synthetic fixtures, global calibration scored
45/60 and per-topic upper thresholds 42/60. This does not support adopting
per-topic thresholds yet. Independent real-window evaluation remains blocked by
missing labeled session data; the recommendation below describes the experiment
design, not a validated runtime change.

## Recommendation

Start with **option 0: topic- or topic-family-specific thresholds on frozen
SBERT**, compared with independently calibrated global thresholds. This is the
cheapest deployment candidate: it changes the decision boundary without encoder
training or an additional model call. Focus first on `UPPER`, because the spike's
reported errors are between `correct` and `weakly_correct`; hold `LOWER` at its
global validation-calibrated value. The spike's perfect `incorrect` separation
does not establish that production's hardcoded `LOWER=0.35` is already optimal.

Use a **rubric-driven, context-aware LLM judge offline** to investigate errors
and propose training labels for human review. Give it the contract, scope, a
short preceding-turn history, and the current window; require one of the three
labels plus a short justification tied to the rubric. Evaluate it against
independently human-labeled examples. Its suggestions are not ground truth, and
it must not generate the test labels used to score itself.

Putting that judge on the live warning path is a separate experiment, not the
recommendation to deploy. The 25-second buffer measures accumulated teacher
speech, not an acceptable inference-latency budget. A runtime candidate needs
measured p95/p99 end-to-end latency under concurrent sessions, cost per session,
timeout/failure behavior, and ordered handling of results so late classifications
cannot corrupt consecutive-warning counts. Compare false-warning sequences as
well as window accuracy. Begin with recorded-session replay, then shadow scoring
that does not issue warnings, before considering live enforcement.

G-Eval is primary evidence for the general pattern of form/rubric-based LLM
evaluation, including its limitations and evaluator biases; it is not evidence
that an LLM judge wins this topical-adherence task
([paper](https://arxiv.org/abs/2303.16634)). NVIDIA's NeMoGuard topic-control
model is a relevant specialized baseline because it reads a topical instruction
and conversation history, but it produces binary on/off-topic judgments and is
designed for user-assistant conversations, so it does not natively express this
project's `weakly_correct` class
([model card](https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-topic-control)).

The leading trainable comparator is a **small three-class CrossEncoder** over
`(topic + scope, window)`. It models the contract and transcript jointly and
directly optimizes the labels, whereas the current bi-encoder compresses each
side independently and reduces their relation to one cosine. No cited result
establishes which approach will win on this dataset; both are hypotheses that
need a locked topic-held-out comparison.

The compatibility baseline should be **explicit within-contract ordinal SBERT
triplets using cosine distance**:

- `(topic + scope, correct, weakly_correct)`
- `(topic + scope, weakly_correct, incorrect)`

This directly expresses the two distinctions the product needs, preserves the
current cached-topic-embedding inference path, and avoids pretending that the
three labels are exact cosine targets of `1.0`, `0.5`, and `0.0`. It is still
only an experiment: triplet margin and example construction can dominate a
dataset this small, and ranking does not produce calibrated class boundaries.

Before training, tighten the `correct`/`weakly_correct` rubric and add reviewed
**scope counterfactuals**: utterances that are correct for one nearby scope but
weak or wrong for another. Random unrelated negatives are already easy; the
unresolved question is whether the model responds to the supplied topic and
scope. Split by topic family before generating pairs or triplets so variants of
one contract cannot cross train, validation, and test.

## Why the previous result is not a model-selection result

The aggregate counts in the [spike note](./finetune_spike_findings.md) can be
reconstructed. Treating the reported accuracies as exact, **40 is the only
sample size from 20 through 60 compatible with both**, and matches four held-out
topics with ten examples each:

- Stock: `0.825 = 33/40` correct predictions.
- Fine-tuned: `0.800 = 32/40` correct predictions.
- Difference: `1/40 = 2.5` percentage points, a **net difference of one correct
  prediction**.

This identifies the original "28 held-out examples" as the sample-count typo
under the documented design. The missing artifacts still prevent recovery of
the individual predictions: a net one-prediction difference could mean one loss,
or several gains and losses that cancel. The aggregate counts alone neither
establish a generalization regression nor prove the models equivalent.

The spike also says thresholds were selected by grid search on those held-out
topics; the committed [calibration script](./calibrate_thresholds.py) chooses and
reports performance on the same rows. These are best-fit accuracies for that
sample, not independent test performance. Selection makes the estimates
optimistic; it does not make them a guaranteed upper bound on unseen accuracy
([model-selection guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)).

The committed fixture has only 6 topics and 60 single-sentence examples, while
runtime classifies accumulated teacher windows after 25 seconds. There are no
committed real transcript labels. Topic coverage, label ambiguity, and this
sentence-to-window distribution shift are more important uncertainties than the
choice among closely related losses.

## Options

| Option | What it changes | Fit here | Main limitation |
|---|---|---|---|
| **0. Topic / topic-family thresholds** | Keep SBERT frozen; calibrate `UPPER` per supported topic or family, with a global fallback. | Cheapest first deployment candidate; no encoder training or additional inference call. | Needs independent calibration labels and a defined family lookup; tiny topic samples overfit, and thresholds cannot repair reversed score ordering. |
| **1a. Offline LLM judge / label assistant** | Evaluate contract, scope, recent turns, and current window together; suggest labels and reasons for human review. | Investigates scope errors and supports training-data curation without affecting live warnings. | Evaluator bias, prompt sensitivity, and labeling cost; independent human test labels remain necessary. |
| **1b. Runtime LLM classifier** | Send each current window and preceding context to a judge whose label feeds warning escalation. | A separate accuracy candidate if offline results justify a deployment study. | Unresolved tail latency, concurrency, cost, failure handling, and effects of delayed or misordered results on warning sequences. |
| **2. Small CrossEncoder, 3 classes** | Jointly encode `(topic + scope, window)` and predict three logits. | Leading supervised accuracy candidate; token-level cross-attention can model scope details that one cosine cannot. | New runtime path, no cached topic embedding, one transformer pass per pair, and still too little data for a trustworthy full fine-tune today. |
| **3. Ordinal cosine triplets** | Train the bi-encoder so `correct > weak > incorrect` within each contract. | Best SBERT-specific probe; matches the ordinal target and keeps current inference. | Margin-sensitive; optimizes relative order rather than absolute thresholds; tiny hand-built triplet sets can overfit. |
| **4. CoSENT loss** | Replace cosine MSE with batchwise ranking of labeled pair similarities. | Smallest code/model change and still aligned with cosine inference. | Rankings are formed across all differently scored pairs in a batch, including different topics; batching affects the signal, and class thresholds remain a separate calibration problem. |
| **5. Frozen SBERT + pair-feature classifier** | Keep embeddings frozen and train a strongly regularized 3-way head on pair features such as cosine, `|u-v|`, and `u*v`. | Low-cost control that tests whether a learned boundary helps without updating 22M encoder parameters. | High-dimensional features versus 60 rows; unlike vanilla SetFit, this must be a custom pair-conditioned head to handle arbitrary topics. |

### Threshold baseline: scope and evaluation

Topic-specific calibration is useful for recurring topics with labeled examples;
topic-family calibration can share data across related contracts. Define the
family assignment using contract metadata before testing. Use global thresholds
for unseen or insufficiently calibrated families; do not tune a threshold on a
held-out topic's labels and call that unseen-topic generalization.

Report two settings separately: new sessions within calibrated topics/families,
and entirely unseen families using the fallback. Keep source sessions and their
paraphrases together when splitting. The current single-sentence fixtures do not
provide the independent sessions needed for the first setting.

This baseline has a concrete limit: in the saved `synthetic_dataset.csv`, all six
topics contain a `correct` example scoring below a `weakly_correct` example.
No per-topic upper threshold can perfectly separate those examples. Calibration
may fix topic-to-topic score offsets, but those within-topic ordering errors
remain a reason to investigate representations, context, and labels.

### Loss-specific cautions

`CosineSimilarityLoss` minimizes MSE between cosine similarity and a supplied
float label. Sentence Transformers documents CoSENT and AnglE as stronger-signal
drop-in alternatives; CoSENT orders pair scores, while AnglE changes the
similarity geometry to reduce cosine-gradient saturation
([loss reference](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)).
For this service, CoSENT is the cleaner ablation because production decisions
still use cosine. Neither method guarantees calibrated three-class thresholds,
and the documentation's general performance claim is not evidence of a gain on
this dataset.

`MultipleNegativesRankingLoss` is a poor fit for the present labels. It requires
each topic's matched positive to outrank all other documents in the batch. With
several valid utterances per topic, other correct utterances become semantic
false negatives; weakly-correct examples also cannot be represented as an
ordered middle class. `NO_DUPLICATES` prevents exact duplicates, not semantic
false negatives ([official algorithm](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss)).

Vanilla SetFit is also not the pair classifier this task needs. SetFit first
forms same-class/different-class pairs to tune a sentence encoder, then trains a
classifier on each single text embedding
([conceptual guide](https://huggingface.co/docs/setfit/main/en/conceptual_guides/setfit)).
One could serialize `topic [SEP] window` into a string, but that discards the
cached bi-encoder design and should be treated as a separate empirical baseline,
not as evidence that SetFit's published few-shot results transfer here.

A CrossEncoder natively consumes a sentence pair and supports three-label
classification ([training guide](https://sbert.net/docs/cross_encoder/training_overview.html)).
It jointly processes the pair and cannot precompute independent embeddings
([model reference](https://sbert.net/docs/package_reference/cross_encoder/model.html)).
The original SBERT work and Augmented SBERT document the usual tradeoff: joint
pair encoders can model the relation more directly, while bi-encoders are much
cheaper when one side can be cached
([Augmented SBERT](https://arxiv.org/abs/2010.08240)). Since this service scores
one contract/window pair rather than retrieving over millions of candidates,
the latency cost may be acceptable, but it must be benchmarked in the actual
25-second window pipeline.

Teacher distillation is a later-stage option, not a remedy for the current data.
Augmented SBERT trains a CrossEncoder teacher, pseudo-labels additional pairs,
then trains a faster bi-encoder. That is materially useful only after a teacher
has reliable topic-held-out performance and there is representative unlabeled
traffic; otherwise it mainly transfers the teacher's errors.

## Bounded experiment design

1. Rewrite the label rubric and have a second reviewer adjudicate the
   `correct`/`weakly_correct` boundary. Add nearby-scope counterfactuals and
   multi-sentence windows resembling production.
2. Reserve entirely unseen topic families for a locked generalization test.
   Within the remaining families, split by source session for training,
   calibration, and evaluation of known-family thresholds. Make these splits
   **before** generating triplets, paraphrases, hard negatives, or pseudo-labels.
3. Compare frozen SBERT with global versus topic/family `UPPER` thresholds first,
   keeping the globally calibrated `LOWER` fixed for this comparison. Run the
   rubric-driven LLM judge offline on the same evaluation examples, blind to
   their gold labels; freeze its prompt and decoding settings before testing.
   Review its errors and training-label suggestions independently. Compare a
   small CrossEncoder when reviewed training coverage supports it; ordinal
   triplet SBERT, CoSENT, and the frozen pair-feature head remain alternatives.
4. Select prompts, thresholds, margins, and stopping points on validation only.
   Report macro-F1, weakly-correct F1, false-`incorrect` rate, per-topic results,
   and variation across training seeds or repeated judge runs as applicable.
   Replay complete sessions to measure false-warning sequences: the current
   warning engine escalates only consecutive `incorrect` classifications, so
   resolving `correct` versus `weakly_correct` alone need not improve warnings.
5. Consider a runtime LLM only after offline evaluation, session replay, and a
   separate latency/cost/failure study. Offline label assistance does not imply
   readiness to replace the live classifier.

There is no justified numerical collection target yet. Define the minimum
worthwhile improvement and acceptable false-warning rate, then use a development
pilot to estimate paired model disagreements, class balance, and dependence
within topics/sessions. Use those estimates to size a fresh test for the chosen
metric and desired power or precision. Count independent topics/sessions as well
as windows; adding paraphrases or triplet combinations does not create equally
many independent observations. Keep exploratory pilot results separate from
claims that a small performance difference has been established.
