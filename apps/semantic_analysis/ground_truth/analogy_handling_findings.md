# How the classifier handles sustained analogies

Two instrumented end-to-end runs, 2026-09-26 and 2026-09-27, measuring what the
live pipeline does when a teacher sustains a single analogy for several windows.
An analogy is on-topic pedagogy, but for the duration of the vehicle the speech
shares almost no vocabulary with the contract topic — so it is the natural
hard-negative case for a cosine-against-topic classifier.

Both runs drove the real services (`semantic_analysis` on 8002, `warning_engine`
on 8003) over HTTP. No mocks, no fixtures. Every number below came off a run.

## Method

- Scripts parsed with `eval_dataset.tools.script_parser`, turns posted in order
  to `POST /ingest/segment` — teacher as `user_id=1`, learner as `user_id=2` —
  then `POST /session/{id}/end` to flush the tail buffer.
- Durations synthesised from word count at 150 wpm (`words * 0.4s`), timestamps
  gapless and monotonic from 0.0. No audio, no STT: this isolates the semantic
  layer from transcription error.
- `scope` registered empty, so the topic embedding is exactly the script's TOPIC
  line. **Every cosine here is sensitive to scope wording** — see
  `scope_sensitivity.py`. Margins as tight as 0.010 should not be read as stable.
- Window composition, cleaned text and per-turn cosines reproduced by importing
  `clean_text` / `classify` / `embed` from `apps/semantic_analysis/main.py` and
  loading the same `all-MiniLM-L6-v2`. Reproduced cosines matched the live
  service to three decimals in both runs.

## Constants in effect (read back from the live service banner)

| Constant | Value | Effect |
|---|---|---|
| `UPPER` | 0.36 | cosine >= UPPER → `correct` |
| `LOWER` | 0.14 | LOWER..UPPER → `weakly_correct`; < LOWER → `incorrect` |
| `WINDOW_DURATION_THRESHOLD` | 25.0 s | teacher buffer flushes at >= 25 s accumulated |
| `MIN_CONTENT_TOKENS` | 3 | fewer cleaned tokens → window skipped, not scored |
| escalation | 1 → silent, 2 → strong, >=3 → severe | on `consecutive_incorrect` |

Note the buffer does **not** carry excess seconds forward: `process_window`
clears `segments` and resets `accumulated_seconds` to 0.0, so any turn of 63+
words (>= 25 s at this rate) forms a window by itself.

## Headline result

| | Run 1 — `sess_C05` | Run 2 — `sess_B08` |
|---|---|---|
| Topic | Node.js basics (event loop, non-blocking I/O, callbacks) | Machine learning intro (training data, features, overfitting, validation) |
| Analogy vehicle | restaurant kitchen / waiter | wine cellar / sommelier apprentice |
| Analogy length | 2 windows (154 words) | 5 windows (366 words) |
| Turns / windows | 20 / 7 | 28 / 12 |
| Analogy window cosines | 0.172, 0.155 | 0.179, 0.130, 0.197, 0.008, 0.029 |
| Lowest cosine | 0.155 | **0.008** |
| `incorrect` windows | **0** | **3** (25.0%) |
| Warnings issued | **0** | **1 — STRONG** |
| `max_consecutive_incorrect` | 0 | 2 |
| Escalation reached | none | strong (not severe) |

**Sensitivity to analogy is a function of duration, and the cliff sits between
2 and 5 windows.** A two-window analogy is invisible to the system. A
five-window analogy is caught, but under-escalated by one tier.

---

# Run 1 — `sess_C05`, two-window analogy

`eval_dataset/scripts/person_C/sess_C05.txt` · TEACHER=C (uid 1) · LEARNER=D (uid 2) ·
CATEGORY=`code_switch` · barter_id 9905 · 20 turns · 224.4 s total (teacher 190.4, learner 34.0)

## Windows (7)

| # | Turns | Buf s | ts | Words raw→clean | Cosine | Verdict | Distance | Warning | Consec |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,3,5 | 34.4 | 0.0–52.8 | 86→83 | **0.784** | `correct` | +0.424 vs UPPER | none | 0 |
| 2 ⬅ | 6 | 34.4 | 52.8–87.2 | 86→86 | **0.172** | `weakly_correct` | **+0.032 vs LOWER** | none | 0 |
| 3 ⬅ | 7 | 27.2 | 87.2–114.4 | 68→68 | **0.155** | `weakly_correct` | **+0.015 vs LOWER** | none | 0 |
| 4 | 9,11 | 32.4 | 117.2–163.2 | 81→81 | **0.481** | `correct` | +0.121 vs UPPER | none | 0 |
| 5 | 13,15 | 26.4 | 165.2–193.2 | 66→66 | **0.669** | `correct` | +0.309 vs UPPER | none | 0 |
| 6 | 17,18 | 25.6 | 196.8–222.4 | 64→63 | **0.373** | `correct` | +0.013 vs UPPER | none | 0 |
| 7 | 20 | 10.0 | 228.0–238.0 | 25→24 | **0.436** | `correct` | +0.076 vs UPPER | none | 0 |

⬅ = analogy window. Both analogy turns exceed 25 s alone (86 words = 34.4 s,
68 words = 27.2 s), so each got its own window with **no on-topic speech mixed
in to rescue it** — the cleanest possible detection case, and still no flag.

Filler cleaning was near-inert: window 1 lost `Okay,` `actually` `so`; windows
6 and 7 lost one `so` each; **windows 2–5 lost nothing**. The analogy was
embedded verbatim.

## Per-turn cosine, scored alone (Run 1)

| Turn | Spk | Role | Words | Cosine | In isolation |
|---|---|---|---|---|---|
| 1 | C | teacher | 27 | 0.773 | `correct` |
| 2 | D | learner | 13 | 0.393 | `correct` |
| 3 | C | teacher | 31 | 0.607 | `correct` |
| 4 | D | learner | 13 | 0.285 | `weakly_correct` |
| 5 | C | teacher | 28 | 0.272 | `weakly_correct` |
| 6 ⬅ | C | teacher | 86 | **0.172** | `weakly_correct` |
| 7 ⬅ | C | teacher | 68 | **0.155** | `weakly_correct` |
| 8 | D | learner | 7 | 0.131 | `incorrect` |
| 9 | C | teacher | 47 | 0.478 | `correct` |
| 10 | D | learner | 10 | 0.292 | `weakly_correct` |
| 11 | C | teacher | 34 | 0.399 | `correct` |
| 12 | D | learner | 14 | 0.403 | `correct` |
| 13 | C | teacher | 35 | 0.650 | `correct` |
| 14 | D | learner | 5 | 0.055 | `incorrect` |
| 15 | C | teacher | 31 | 0.414 | `correct` |
| 16 | D | learner | 9 | 0.379 | `correct` |
| 17 | C | teacher | 28 | 0.259 | `weakly_correct` |
| 18 | C | teacher | 36 | 0.412 | `correct` |
| 19 | D | learner | 14 | 0.151 | `weakly_correct` |
| 20 | C | teacher | 25 | 0.436 | `correct` |

The two lowest teacher turns in the whole script are exactly the two pure-analogy
turns; the next lowest (turn 17, 0.259) is 0.087 above turn 6. The embedding
discriminates correctly — the thresholds are simply not placed to act on it.
Learner turns are never topic-scored (they route to engagement), so turns 8 and
14 falling below LOWER influenced nothing.

## Engagement (Run 1)

| Learner turn | Words | Dur s | Q cum | Ack cum | Learner share | Score |
|---|---|---|---|---|---|---|
| 2 | 13 | 5.2 | 1 | 0 | 32.5% | 0.500 (placeholder, <30 s data) |
| 4 | 13 | 5.2 | 3 | 0 | 31.0% | 0.800 |
| 8 | 7 | 2.8 | 4 | 1 | 12.1% | 0.736 |
| 10 | 10 | 4.0 | 6 | 1 | 13.0% | 0.726 |
| 12 | 14 | 5.6 | 9 | 1 | 15.1% | 0.757 |
| 14 | 5 | 2.0 | 12 | 1 | 14.8% | 0.737 |
| 16 | 9 | 3.6 | 14 | 1 | 15.5% | 0.745 |
| 19 | 14 | 5.6 | 15 | 1 | 15.9% | 0.746 |

Final summary: `learner_engagement_score 0.746`, teacher 190.4 s vs learner
34.0 s, 15 questions, 1 acknowledgment, 8 learner segments. Never below 0.3, so
`post_engagement_alert` never fired.

## Session end (Run 1)

```
[Barter 9905  SESSION END]
→  Windows: 7 total  0 incorrect  (0.0% off-topic)
→  Warnings issued: 0  max_consecutive: 0
```

---

# Run 2 — `sess_B08`, five-window analogy

`eval_dataset/scripts/person_B/sess_B08.txt` · TEACHER=B (uid 1) · LEARNER=A (uid 2) ·
CATEGORY=`code_switch` · barter_id 9908 · 28 turns (17 teacher)

Analogy = turns 6–10, 74/69/73/77/73 words, each >= 25 s so each claims its own
window. Turn 5 was deliberately padded to 81 words so it would flush separately
and leave the five analogy turns as five uncontaminated windows.

## Windows (12)

| # | Turns | Buf s | ts | Words raw→clean | Cosine | Verdict | Distance | Warning | Consec |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,3 | 29.6 | 0.0–37.6 | 74→72 | **0.453** | `correct` | +0.093 vs UPPER | none | 0 |
| 2 | 5 | 32.4 | 42.8–75.2 | 81→79 | **0.229** | `weakly_correct` | +0.089 vs LOWER | none | 0 |
| 3 ⬅ | 6 | 29.6 | 75.2–104.8 | 74→74 | **0.179** | `weakly_correct` | +0.039 vs LOWER | none | 0 |
| 4 ⬅ | 7 | 27.6 | 104.8–132.4 | 69→69 | **0.130** | `incorrect` | -0.010 vs LOWER | silent | 1 |
| 5 ⬅ | 8 | 29.2 | 132.4–161.6 | 73→73 | **0.197** | `weakly_correct` | +0.057 vs LOWER | none | 0 |
| 6 ⬅ | 9 | 30.8 | 161.6–192.4 | 77→77 | **0.008** | `incorrect` | -0.132 vs LOWER | silent | 1 |
| 7 ⬅ | 10 | 29.2 | 192.4–221.6 | 73→73 | **0.029** | `incorrect` | -0.111 vs LOWER | **STRONG** | 2 |
| 8 | 12,14 | 39.6 | 224.4–269.6 | 99→97 | **0.191** | `weakly_correct` | +0.051 vs LOWER | none | 0 |
| 9 | 16,18 | 39.2 | 275.6–320.0 | 98→96 | **0.271** | `weakly_correct` | -0.089 vs UPPER | none | 0 |
| 10 | 20,22 | 36.4 | 323.2–367.6 | 91→91 | **0.321** | `weakly_correct` | -0.039 vs UPPER | none | 0 |
| 11 | 24,26 | 32.0 | 370.0–406.0 | 80→79 | **0.230** | `weakly_correct` | +0.090 vs LOWER | none | 0 |
| 12 | 28 | 17.2 | 409.6–426.8 | 43→43 | **0.381** | `correct` | +0.021 vs UPPER | none | 0 |
⬅ = analogy window (turns 6–10).

## The counter reset — why 5 windows only reached `strong`

The five analogy windows scored **0.179 → 0.130 → 0.197 → 0.008 → 0.029**.
Three of the five are `incorrect`, but they are not contiguous: W5 (0.197) sits
between them, and `run_warning_decision` resets `consecutive_incorrect = 0` on
anything that is not `incorrect`.

| Window | Cosine | Verdict | Counter after | Action |
|---|---|---|---|---|
| W4 | 0.130 | `incorrect` | 1 | silent |
| W5 | 0.197 | `weakly_correct` | **0 — reset** | none |
| W6 | 0.008 | `incorrect` | 1 | silent |
| W7 | 0.029 | `incorrect` | 2 | **STRONG** |

Had W5 scored 0.057 lower, W4/W6/W7 would have been three consecutive
`incorrect` and the ladder would have reached **severe**. A five-window unbroken
digression escalated only to `strong`, and a single window differing by 0.057 of
cosine is what held it back. The strict-consecutive counter has no tolerance for
one noisy window inside an otherwise solid drift — which is the shape real drift
takes.

## Per-turn cosine, scored alone (Run 2)

| Turn | Spk | Role | Words | Cosine | In isolation |
|---|---|---|---|---|---|
| 1 | B | teacher | 31 | 0.496 | `correct` |
| 2 | A | learner | 20 | 0.300 | `weakly_correct` |
| 3 | B | teacher | 43 | 0.191 | `weakly_correct` |
| 4 | A | learner | 13 | 0.053 | `incorrect` |
| 5 | B | teacher | 81 | 0.229 | `weakly_correct` |
| 6 | B | teacher | 74 | 0.179 | `weakly_correct` |
| 7 | B | teacher | 69 | 0.130 | `incorrect` |
| 8 | B | teacher | 73 | 0.197 | `weakly_correct` |
| 9 | B | teacher | 77 | 0.008 | `incorrect` |
| 10 | B | teacher | 73 | 0.029 | `incorrect` |
| 11 | A | learner | 7 | -0.014 | `incorrect` |
| 12 | B | teacher | 56 | 0.197 | `weakly_correct` |
| 13 | A | learner | 14 | 0.094 | `incorrect` |
| 14 | B | teacher | 43 | 0.318 | `weakly_correct` |
| 15 | A | learner | 15 | -0.077 | `incorrect` |
| 16 | B | teacher | 54 | 0.182 | `weakly_correct` |
| 17 | A | learner | 13 | 0.048 | `incorrect` |
| 18 | B | teacher | 44 | 0.304 | `weakly_correct` |
| 19 | A | learner | 8 | 0.370 | `correct` |
| 20 | B | teacher | 42 | 0.280 | `weakly_correct` |
| 21 | A | learner | 20 | 0.112 | `incorrect` |
| 22 | B | teacher | 49 | 0.219 | `weakly_correct` |
| 23 | A | learner | 6 | 0.112 | `incorrect` |
| 24 | B | teacher | 35 | 0.132 | `incorrect` |
| 25 | A | learner | 10 | 0.255 | `weakly_correct` |
| 26 | B | teacher | 45 | 0.272 | `weakly_correct` |
| 27 | A | learner | 9 | 0.128 | `incorrect` |
| 28 | B | teacher | 43 | 0.381 | `correct` |
**Turn 24 is a false positive worth recording**: "Often, yes, and that's one
lever. More data is another, because specifics get harder to memorise as they
multiply. Regularisation is a third — you make memorising expensive so the
pattern becomes the cheaper option." Unambiguously on-topic ML content, scored
**0.132**, below LOWER. It escaped flagging only because it was buffered with
turn 26 into W11 (0.230). The embedding is noisier on abstract ML prose than the
threshold placement assumes.

## Engagement (Run 2)

| Order | Score | Learner share | Q cum | Ack cum |
|---|---|---|---|---|
| 1 | 0.80 | 31% | 3 | 0 |
| 2 | 0.53 | 7% | 3 | 1 |
| 3 | 0.50 | 9% | 5 | 1 |
| 4 | 0.63 | 10% | 7 | 1 |
| 5 | 0.64 | 11% | 9 | 1 |
| 6 | 0.64 | 11% | 10 | 1 |
| 7 | 0.67 | 13% | 12 | 1 |
| 8 | 0.66 | 13% | 13 | 1 |
| 9 | 0.70 | 13% | 15 | 2 |
| 10 | 0.70 | 13% | 17 | 2 |

Engagement dipped to 0.50 during the analogy — learner share fell to 7–9% while
the teacher monologued — then recovered to 0.70. Never below 0.3, so no alert.

## Session end (Run 2)

```
[Barter 9908  SESSION END]
→  Windows: 12 total  3 incorrect  (25.0% off-topic)
→  Warnings issued: 1  max_consecutive: 2
```

## Window texts (Run 2)

**W1** — turns [1, 3], cos **0.453**, `correct`

> Today I want to cover what training actually is, what a feature is, and then the two ways a model can look brilliant in the lab and be worthless in production. That's the right thing to be stuck on. Nobody writes the rules. You show the model a lot of examples with the right answers attached, and it adjusts itself a little bit after each one until the answers stop being wrong so often.

**W2** — turns [5], cos **0.229**, `weakly_correct`

> I'm going to put the maths aside for a while and tell you about an apprentice instead, because every single mistake a model can make, a person makes too, and it is much easier to see the mistake when a person is the one making it. So stay with me through a fairly long detour, don't worry if it feels like we've wandered off, and I promise we'll map every part of it back onto the code afterwards, piece by piece.

**W3** — turns [6], cos **0.179**, `weakly_correct`

> Picture an apprentice in a wine cellar. Her mentor pours twelve glasses every single morning, and her job is to name the grape, the region, and the year for each one before he tells her the answer. In the first week she is guessing wildly. She calls a cool coastal white something grown four hundred miles inland, and she calls a young bottle old because the colour looked deeper than she expected it to.

**W4** — turns [7], cos **0.130**, `incorrect`

> Every morning the mentor pours again, and every morning he tells her what she got wrong. Slowly she stops reaching for the label in her memory and starts noticing the things sitting underneath it. A certain sharpness right at the finish. A faint smell of wet stone. The way the oak sits behind the fruit instead of sitting on top of it and smothering everything else in the glass.

**W5** — turns [8], cos **0.197**, `weakly_correct`

> After a year she can walk into a blind flight, swirl a glass, hold it up to the window, and place it within a few years and a hundred miles. Her mentor never handed her a rule sheet. He never once said a word about acid levels or soil composition or harvest dates. He simply poured, she guessed, he corrected her, and she adjusted. Thousands of pours, thousands of tiny corrections, no rules.

**W6** — turns [9], cos **0.008**, `incorrect`

> But there is a failure mode in that cellar and it is worth dwelling on properly. Suppose the mentor only ever pours from his own collection, and his own collection happens to be heavy on one estate and two very good decades. She will become genuinely extraordinary at those bottles and quietly useless everywhere else. Hand her something from a region he never bought and she is guessing wildly again, except now with real confidence behind it.

**W7** — turns [10], cos **0.029**, `incorrect`

> There is a worse version of that too. Imagine she starts recognising the bottles themselves rather than the wine. A chip on the rim of one decanter. The mentor's habit of always pouring the Riesling into the narrow glass. The fact that Thursday morning is always the expensive flight. She will score beautifully every morning in that cellar and fall apart completely the moment a stranger pours for her in a different room.

**W8** — turns [12, 14], cos **0.191**, `weakly_correct`

> Exactly, and she can't tell that she's doing it, which is the dangerous part. Now let me map all of it. The morning pours are your training data. The mentor's answer is the label. The sharpness and the wet stone and where the oak sits are features. The adjusting after each correction is the training loop. That's a biased training set, and it shows up as a model that scores well on everything resembling what it saw and collapses on everything else. The general name for the gap is distribution shift — production simply doesn't look like your cellar.

**W9** — turns [16, 18], cos **0.271**, `weakly_correct`

> It is, and it has its own name. That's leakage. Your model found a signal that happens to correlate with the answer in your data but has nothing to do with the thing you actually care about. A timestamp, a file path, a column that was filled in after the outcome was already known. You hold data back. You never let the model see it during training, and you only score against it at the end. If it does well in training and badly on the held-out set, something is wrong — either it memorised, or it leaked.

**W10** — turns [20, 22], cos **0.321**, `weakly_correct`

> Validation is what you check against while you're still making decisions — which model, which settings. The test set is stricter: you touch it once, at the very end. Look at it repeatedly and you've quietly turned it into another validation set. The difference is whether what she picked up transfers. Noticing that wet stone means a particular soil transfers to any glass anywhere. Noticing that Thursday is expensive transfers nowhere. Overfitting is when a model has enough capacity to memorise the specifics instead of being forced to find the pattern.

**W11** — turns [24, 26], cos **0.230**, `weakly_correct`

> Often, yes, and that's one lever. More data is another, because specifics get harder to memorise as they multiply. Regularisation is a third — you make memorising expensive so the pattern becomes the cheaper option. Yes, and you should. Plot training error and validation error together as training proceeds. Training error keeps falling. Validation error falls, bottoms out, then starts climbing. That turning point is where memorising took over from learning, and it's the most useful picture in the field.

**W12** — turns [28], cos **0.381**, `correct`

> Take a small tabular dataset, hold out twenty percent, and fit something deliberately too large. Watch the validation curve turn upward. Then shrink the model and watch the turn move. You'll understand overfitting better in an afternoon than from any amount of reading.

---

# What the two runs establish

1. **A short analogy leaves no trace.** Run 1's two analogy windows sat +0.032
   and +0.015 above LOWER — inside `weakly_correct`, which carries no consequence
   anywhere. Only `incorrect` increments `consecutive_incorrect`, and only
   `incorrect` feeds `total_drift_incidents`. A deliberate, zero-vocabulary,
   two-window digression is completely absent from the drift summary.
2. **Length is what pushes an analogy under LOWER.** Run 1's floor was 0.155;
   Run 2 reached **0.008** and **0.029**, essentially orthogonal to the topic.
   Five windows of pure vehicle vocabulary with no on-topic anchor collapse the
   cosine; two windows stay in the 0.15–0.17 cushion.
3. **Buffering does not rescue a long analogy, because long turns get their own
   windows.** Any turn of 63+ words fills a 25 s window alone at this speaking
   rate, and there is no carry-forward. Both runs' analogy turns were scored in
   isolation whether or not that was intended.
4. **The escalation ladder under-reacts to real drift.** A contiguous five-window
   digression produced one `strong`, not `severe`, because one intermediate
   window landed 0.057 above LOWER and reset the counter.
5. **The classifier's ranking is sound; the thresholds and the counter are the
   weak links.** In both runs the analogy turns were the lowest-scoring teacher
   turns in the script, by a clear margin. The signal is there.

## Suggested changes these runs support

- Give `weakly_correct` some consequence — e.g. record runs of consecutive
  `weakly_correct` windows in the drift summary, or treat two consecutive windows
  below ~0.20 as a soft drift incident. Raising `LOWER` is the wrong lever: it
  would also punish genuine, useful analogies and would have flagged turn 24,
  which is correct ML content.
- Make the escalation counter tolerant of a single non-`incorrect` window — a
  decaying counter, or a fraction-over-last-N rule — so one blip cannot downgrade
  a sustained drift by a full tier.

## Known limitations of these runs

- Synthetic durations from word count; no audio and no STT, so no WER interaction.
- `scope=""` in both runs. A non-empty scope changes the topic embedding and
  therefore every cosine; margins of 0.010–0.057 are not stable across wordings.
- Two scripts, one run each. Nothing here is a distribution.
- Both runs had no barter backend on 8000, so `warning_engine`'s persistence
  POSTs failed (404 in Run 1 where another app was listening, connection-refused
  in Run 2). Classification and escalation are unaffected — but note that
  `end_session` posts the engagement summary without `raise_for_status()`, so a
  404 there is silent: Run 2's log printed `Failed to POST engagement summary`
  and then `engagement summary posted` on the following line. Worth fixing
  independently of anything above.
- The `warning_engine` startup banner prints `1→silent 2→mild 3-4→strong
  5+→severe`, which contradicts its own code (no `mild` tier exists; it is
  1→silent, 2→strong, >=3→severe). Stale banner text.
