# Forensic report — sess_B08 (Machine learning intro), 5-window sustained analogy

Run 2026-09-27 · script `eval_dataset/scripts/person_B/sess_B08.txt` · topic `Machine learning intro (training data, features, overfitting, validation)` · TEACHER=B(uid 1) LEARNER=A(uid 2) · CATEGORY=code_switch · barter_id 9908

Thresholds live: UPPER=0.36 LOWER=0.14 window=25s. Durations = words x 0.4s (150wpm), gapless. 28 turns, 17 teacher.

## Windows

| # | Turns | Buf s | ts | Words | Cosine | Verdict | Distance | Warning | Consec |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,3 | 29.6 | 0.0–37.6 | 74→72 | **0.453** | correct | +0.093 vs UPPER | none | 0 |
| 2 | 5 | 32.4 | 42.8–75.2 | 81→79 | **0.229** | weakly_correct | +0.089 vs LOWER | none | 0 |
| 3 | 6 | 29.6 | 75.2–104.8 | 74→74 | **0.179** | weakly_correct | +0.039 vs LOWER | none | 0 |
| 4 | 7 | 27.6 | 104.8–132.4 | 69→69 | **0.130** | incorrect | -0.010 vs LOWER | silent (1 consec) | 1 |
| 5 | 8 | 29.2 | 132.4–161.6 | 73→73 | **0.197** | weakly_correct | +0.057 vs LOWER | none | 0 |
| 6 | 9 | 30.8 | 161.6–192.4 | 77→77 | **0.008** | incorrect | -0.132 vs LOWER | silent (1 consec) | 1 |
| 7 | 10 | 29.2 | 192.4–221.6 | 73→73 | **0.029** | incorrect | -0.111 vs LOWER | STRONG (2 consec) | 2 |
| 8 | 12,14 | 39.6 | 224.4–269.6 | 99→97 | **0.191** | weakly_correct | +0.051 vs LOWER | none | 0 |
| 9 | 16,18 | 39.2 | 275.6–320.0 | 98→96 | **0.271** | weakly_correct | -0.089 vs UPPER | none | 0 |
| 10 | 20,22 | 36.4 | 323.2–367.6 | 91→91 | **0.321** | weakly_correct | -0.039 vs UPPER | none | 0 |
| 11 | 24,26 | 32.0 | 370.0–406.0 | 80→79 | **0.230** | weakly_correct | +0.090 vs LOWER | none | 0 |
| 12 | 28 | 17.2 | 409.6–426.8 | 43→43 | **0.381** | correct | +0.021 vs UPPER | none | 0 |

## Per-turn cosine (isolated)

| Turn | Spk | Role | Words | Cosine | In isolation |
|---|---|---|---|---|---|
| 1 | B | teacher | 31 | 0.496 | correct |
| 2 | A | learner | 20 | 0.3 | weakly_correct |
| 3 | B | teacher | 43 | 0.191 | weakly_correct |
| 4 | A | learner | 13 | 0.053 | incorrect |
| 5 | B | teacher | 81 | 0.229 | weakly_correct |
| 6 | B | teacher | 74 | 0.179 | weakly_correct |
| 7 | B | teacher | 69 | 0.13 | incorrect |
| 8 | B | teacher | 73 | 0.197 | weakly_correct |
| 9 | B | teacher | 77 | 0.008 | incorrect |
| 10 | B | teacher | 73 | 0.029 | incorrect |
| 11 | A | learner | 7 | -0.014 | incorrect |
| 12 | B | teacher | 56 | 0.197 | weakly_correct |
| 13 | A | learner | 14 | 0.094 | incorrect |
| 14 | B | teacher | 43 | 0.318 | weakly_correct |
| 15 | A | learner | 15 | -0.077 | incorrect |
| 16 | B | teacher | 54 | 0.182 | weakly_correct |
| 17 | A | learner | 13 | 0.048 | incorrect |
| 18 | B | teacher | 44 | 0.304 | weakly_correct |
| 19 | A | learner | 8 | 0.37 | correct |
| 20 | B | teacher | 42 | 0.28 | weakly_correct |
| 21 | A | learner | 20 | 0.112 | incorrect |
| 22 | B | teacher | 49 | 0.219 | weakly_correct |
| 23 | A | learner | 6 | 0.112 | incorrect |
| 24 | B | teacher | 35 | 0.132 | incorrect |
| 25 | A | learner | 10 | 0.255 | weakly_correct |
| 26 | B | teacher | 45 | 0.272 | weakly_correct |
| 27 | A | learner | 9 | 0.128 | incorrect |
| 28 | B | teacher | 43 | 0.381 | correct |

## Window texts

**W1** (turns [1, 3], cos 0.453, correct):

> Today I want to cover what training actually is, what a feature is, and then the two ways a model can look brilliant in the lab and be worthless in production. That's the right thing to be stuck on. Nobody writes the rules. You show the model a lot of examples with the right answers attached, and it adjusts itself a little bit after each one until the answers stop being wrong so often.

**W2** (turns [5], cos 0.229, weakly_correct):

> I'm going to put the maths aside for a while and tell you about an apprentice instead, because every single mistake a model can make, a person makes too, and it is much easier to see the mistake when a person is the one making it. So stay with me through a fairly long detour, don't worry if it feels like we've wandered off, and I promise we'll map every part of it back onto the code afterwards, piece by piece.

**W3** (turns [6], cos 0.179, weakly_correct):

> Picture an apprentice in a wine cellar. Her mentor pours twelve glasses every single morning, and her job is to name the grape, the region, and the year for each one before he tells her the answer. In the first week she is guessing wildly. She calls a cool coastal white something grown four hundred miles inland, and she calls a young bottle old because the colour looked deeper than she expected it to.

**W4** (turns [7], cos 0.13, incorrect):

> Every morning the mentor pours again, and every morning he tells her what she got wrong. Slowly she stops reaching for the label in her memory and starts noticing the things sitting underneath it. A certain sharpness right at the finish. A faint smell of wet stone. The way the oak sits behind the fruit instead of sitting on top of it and smothering everything else in the glass.

**W5** (turns [8], cos 0.197, weakly_correct):

> After a year she can walk into a blind flight, swirl a glass, hold it up to the window, and place it within a few years and a hundred miles. Her mentor never handed her a rule sheet. He never once said a word about acid levels or soil composition or harvest dates. He simply poured, she guessed, he corrected her, and she adjusted. Thousands of pours, thousands of tiny corrections, no rules.

**W6** (turns [9], cos 0.008, incorrect):

> But there is a failure mode in that cellar and it is worth dwelling on properly. Suppose the mentor only ever pours from his own collection, and his own collection happens to be heavy on one estate and two very good decades. She will become genuinely extraordinary at those bottles and quietly useless everywhere else. Hand her something from a region he never bought and she is guessing wildly again, except now with real confidence behind it.

**W7** (turns [10], cos 0.029, incorrect):

> There is a worse version of that too. Imagine she starts recognising the bottles themselves rather than the wine. A chip on the rim of one decanter. The mentor's habit of always pouring the Riesling into the narrow glass. The fact that Thursday morning is always the expensive flight. She will score beautifully every morning in that cellar and fall apart completely the moment a stranger pours for her in a different room.

**W8** (turns [12, 14], cos 0.191, weakly_correct):

> Exactly, and she can't tell that she's doing it, which is the dangerous part. Now let me map all of it. The morning pours are your training data. The mentor's answer is the label. The sharpness and the wet stone and where the oak sits are features. The adjusting after each correction is the training loop. That's a biased training set, and it shows up as a model that scores well on everything resembling what it saw and collapses on everything else. The general name for the gap is distribution shift — production simply doesn't look like your cellar.

**W9** (turns [16, 18], cos 0.271, weakly_correct):

> It is, and it has its own name. That's leakage. Your model found a signal that happens to correlate with the answer in your data but has nothing to do with the thing you actually care about. A timestamp, a file path, a column that was filled in after the outcome was already known. You hold data back. You never let the model see it during training, and you only score against it at the end. If it does well in training and badly on the held-out set, something is wrong — either it memorised, or it leaked.

**W10** (turns [20, 22], cos 0.321, weakly_correct):

> Validation is what you check against while you're still making decisions — which model, which settings. The test set is stricter: you touch it once, at the very end. Look at it repeatedly and you've quietly turned it into another validation set. The difference is whether what she picked up transfers. Noticing that wet stone means a particular soil transfers to any glass anywhere. Noticing that Thursday is expensive transfers nowhere. Overfitting is when a model has enough capacity to memorise the specifics instead of being forced to find the pattern.

**W11** (turns [24, 26], cos 0.23, weakly_correct):

> Often, yes, and that's one lever. More data is another, because specifics get harder to memorise as they multiply. Regularisation is a third — you make memorising expensive so the pattern becomes the cheaper option. Yes, and you should. Plot training error and validation error together as training proceeds. Training error keeps falling. Validation error falls, bottoms out, then starts climbing. That turning point is where memorising took over from learning, and it's the most useful picture in the field.

**W12** (turns [28], cos 0.381, correct):

> Take a small tabular dataset, hold out twenty percent, and fit something deliberately too large. Watch the validation curve turn upward. Then shrink the model and watch the turn move. You'll understand overfitting better in an afternoon than from any amount of reading.
