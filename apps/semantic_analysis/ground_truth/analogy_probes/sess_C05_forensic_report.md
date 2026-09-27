# Forensic run report — sess_C05 (Node.js basics), sustained restaurant-kitchen analogy

Run date: 2026-09-26. Script: `/Users/manojj/Documents/CSE-Projects/core-barter-system/eval_dataset/scripts/person_C/sess_C05.txt`
Topic (as registered): `Node.js basics (event loop, non-blocking I/O, callbacks)`  ·  TEACHER=C (user_id 1)  ·  LEARNER=D (user_id 2)  ·  CATEGORY=`code_switch`  ·  barter_id=`9905`

## 1. Run setup

### Commands (run from the repo root, both backgrounded, stdout+stderr to the scratchpad)

```bash
apps/warning_engine/venv/bin/python -m uvicorn main:app \
    --host 127.0.0.1 --port 8003 --app-dir apps/warning_engine \
    > scratchpad/warning_engine.log 2>&1 &

apps/semantic_analysis/venv/bin/python -m uvicorn main:app \
    --host 127.0.0.1 --port 8002 --app-dir apps/semantic_analysis \
    > scratchpad/semantic_analysis.log 2>&1 &
```

No environment variables were set. Both services therefore used their in-code defaults:
`WARNING_ENGINE_URL=http://localhost:8003`, `BACKEND_URL=http://localhost:8000`. No repo file was modified.

`/health` returned `{"status":"ok"}` on both within ~1 s of the poll starting (sentence-BERT
`all-MiniLM-L6-v2`, 384-dim, loaded from local cache — the log shows `Loading weights: 103/103`).

### Thresholds actually in effect (read back from the live service banner)

```
Model ready  —  UPPER=0.36  LOWER=0.14  window=25s
```

| Constant | Value | Effect |
|---|---|---|
| `UPPER` | 0.36 | cosine >= 0.36 → `correct` |
| `LOWER` | 0.14 | 0.14 <= cosine < 0.36 → `weakly_correct`; < 0.14 → `incorrect` |
| `WINDOW_DURATION_THRESHOLD` | 25.0 s | teacher buffer flushes when accumulated audio >= 25 s |
| `MIN_CONTENT_TOKENS` | 3 | windows with fewer cleaned tokens are skipped, not scored |
| warning ladder | 1→silent, 2→strong, 3+→severe | `run_warning_decision`, on `consecutive_incorrect` |

Note: the warning-engine startup banner prints `1→silent 2→mild 3–4→strong 5+→severe`, which does
**not** match its own code. The code has no `mild` tier for topic drift: 1 consecutive → `silent`,
2 → `strong`, >=3 → `severe`. The banner is stale text.

### Session wiring

1. `POST /session/9905/contract` on 8002 — body `{barter_id: 9905, topic: "<topic line>", scope: "", teacher_user_id: 1, learner_user_id: 2}` → `{"status":"registered","barter_id":9905}`.
   `scope` was left empty so nothing beyond the script's own TOPIC line entered the topic embedding.
   The embedded topic string is therefore exactly `"Node.js basics (event loop, non-blocking I/O, callbacks). "`.
2. `POST /session/9905/init` on 8003 — body `{teacher_user_id: 1, learner_user_id: 2}` → `{"status":"initialized","barter_id":9905}`.
3. 20 segments posted to `POST /ingest/segment` on 8002, in script order, teacher turns as `user_id=1`, learner turns as `user_id=2`.
4. `POST /session/9905/end` on 8002 → `{"status":"ended","barter_id":9905}` (flushes the tail buffer, then notifies 8003).

### How durations were derived

`duration_seconds = word_count * 0.4` (150 wpm), rounded to 2 dp; `word_count = len(text.split())`.
Timestamps are gapless and monotonic: `ts_start` of each segment = `ts_end` of the previous one,
`ts_end = ts_start + duration_seconds`, starting at 0.0. Total session audio = 224.4 s
(teacher 190.4 s, learner 34.0 s).

### What failed, and why

* **Every warning-engine POST to the backend failed.** 8000 is not the barter backend in this
  environment — some other FastAPI app is listening there (PID 11526, pre-existing, not started by
  this run) and answers `404 Not Found` for all barter routes. Failures logged (each retried 3x per
  `post_to_backend`): `/window/result` x7, `/session/9905/engagement-log` x8, `/session/9905/drift-summary` x1.
  This is evidence-persistence only — it does not affect classification, escalation, or any number below.
* **`semantic_analysis`'s own POST of the engagement summary to the backend did not error.**
  `end_session` posts `/session/9905/engagement-summary` without `raise_for_status()`, so the 404
  came back silently and nothing was logged. That is a real hole: a rejected summary is
  indistinguishable from a stored one.
* Nothing else failed. No exception, no skipped window, no `MIN_CONTENT_TOKENS` skip.

### Reproduction of the internals

Window composition, `clean_text` output, word counts and per-turn cosines were computed in a
separate process that imports `clean_text`/`classify`/`UPPER`/`LOWER` directly from
`apps/semantic_analysis/main.py` and loads the same `all-MiniLM-L6-v2` model. It reproduced all
seven live window cosines to three decimals (0.784 / 0.172 / 0.155 / 0.481 / 0.669 / 0.373 / 0.436)
and all eight live engagement scores exactly, which confirms the window→turn attribution below.

## 2. Windows the system actually formed

Seven windows. Six fired on the 25 s threshold; window 7 was the tail flush at `/session/9905/end`.

| Window | Turns | Buffered s | ts range | Window text (full, as buffered) | Cleaned text (actually embedded) | Words raw→cleaned | Cosine | Band / Verdict | Distance to nearest threshold | Warning action | consecutive_incorrect |
|---|---|---|---|---|---|---|---|---|---|---|---|
| #1 | T1, T3, T5 | 34.4 | 0.0–44.8 | Okay, today I want to get you past the syntax and into the thing that actually makes Node different, which is the event loop and non-blocking I/O. Fair. The short version is that Node runs your JavaScript on a single thread, so if one operation sits there waiting for a file or a database, nothing else can run. It hands the slow work off and keeps going. Let me use a picture that usually makes this click, and then we'll map it straight back to code. | today I want to get you past the syntax and into the thing that makes Node different, which is the event loop and non-blocking I/O. Fair. The short version is that Node runs your JavaScript on a single thread, if one operation sits there waiting for a file or a database, nothing else can run. It hands the slow work off and keeps going. Let me use a picture that usually makes this click, and then we'll map it straight back to code. | 86→83 | **0.784** | above UPPER → **correct** | +0.424 above UPPER (0.36) | `none` (not incorrect) | 0 |
| #2 | T6 | 34.4 | 44.8–79.2 | Think of one waiter in a small restaurant. He walks to table four, takes the order, and instead of standing next to the table until the food is ready, he clips the ticket to the rail and walks away. He goes to table five, takes their order, clips that ticket up too, refills someone's water on the way back, and clears two plates from table two. The kitchen is doing the slow part, the grilling, the simmering, the plating, and none of that is his problem. | Think of one waiter in a small restaurant. He walks to table four, takes the order, and instead of standing next to the table until the food is ready, he clips the ticket to the rail and walks away. He goes to table five, takes their order, clips that ticket up too, refills someone's water on the way back, and clears two plates from table two. The kitchen is doing the slow part, the grilling, the simmering, the plating, and none of that is his problem. | 86→86 | **0.172** | LOWER..UPPER → **weakly_correct** | +0.032 above LOWER (0.14); other boundary -0.188 | `none` (not incorrect) | 0 |
| #3 | T7 | 27.2 | 79.2–106.4 | When a plate is finally up, the cook rings the bell, the ticket comes down off the rail, and the waiter carries that plate out. Then he's right back to whoever is waving at him. One waiter, dozens of tables, and nobody sits there watching a pan. The only way the whole floor stops is if he decides to stand at the stove and stir the sauce himself. | When a plate is finally up, the cook rings the bell, the ticket comes down off the rail, and the waiter carries that plate out. Then he's right back to whoever is waving at him. One waiter, dozens of tables, and nobody sits there watching a pan. The only way the whole floor stops is if he decides to stand at the stove and stir the sauce himself. | 68→68 | **0.155** | LOWER..UPPER → **weakly_correct** | +0.015 above LOWER (0.14); other boundary -0.205 | `none` (not incorrect) | 0 |
| #4 | T9, T11 | 32.4 | 109.2–145.6 | Exactly. The waiter is the single thread running your JavaScript, the kitchen is the operating system doing the file read or the network call, and the ticket rail is the queue. The bell is the completion event, and carrying the plate out is your callback finally running. Standing at the stove is any synchronous blocking call. A tight loop, readFileSync, a long JSON.parse over something huge. While that runs the whole server is unresponsive, because there is only ever one waiter. | Exactly. The waiter is the single thread running your JavaScript, the kitchen is the operating system doing the file read or the network call, and the ticket rail is the queue. The bell is the completion event, and carrying the plate out is your callback finally running. Standing at the stove is any synchronous blocking call. A tight loop, readFileSync, a long JSON.parse over something huge. While that runs the whole server is unresponsive, because there is only ever one waiter. | 81→81 | **0.481** | above UPPER → **correct** | +0.121 above UPPER (0.36) | `none` (not incorrect) | 0 |
| #5 | T13, T15 | 26.4 | 151.2–179.6 | Node hands the read to the thread pool, and your callback is parked until the read finishes. Then it gets queued, and the event loop picks it up on the next pass through that phase. A handful — timers, pending callbacks, poll, check, close. For now just remember that timers run setTimeout callbacks, poll is where most I/O completions land, and check is where setImmediate runs. | Node hands the read to the thread pool, and your callback is parked until the read finishes. Then it gets queued, and the event loop picks it up on the next pass through that phase. A handful — timers, pending callbacks, poll, check, close. For now just remember that timers run setTimeout callbacks, poll is where most I/O completions land, and check is where setImmediate runs. | 66→66 | **0.669** | above UPPER → **correct** | +0.309 above UPPER (0.36) | `none` (not incorrect) | 0 |
| #6 | T17, T18 | 25.6 | 183.2–208.8 | Underneath, yes, but microtasks are drained between phases, so a resolved promise runs before the next setTimeout even if the timeout was zero. That trips people up constantly. The practical rule is simple. Never block the loop. Push CPU-heavy work to a worker thread or a separate process, and always prefer the async version of an API even when the sync one looks easier. | Underneath, yes, but microtasks are drained between phases, a resolved promise runs before the next setTimeout even if the timeout was zero. That trips people up constantly. The practical rule is simple. Never block the loop. Push CPU-heavy work to a worker thread or a separate process, and always prefer the async version of an API even when the sync one looks easier. | 64→63 | **0.373** | above UPPER → **correct** | +0.013 above UPPER (0.36) | `none` (not incorrect) | 0 |
| #7 | T20 | 10.0 | 214.4–224.4 | Yes, do that next. Use http.createServer, read the file asynchronously inside the handler, and send the response from the callback so the loop stays free. | Yes, do that next. Use http.createServer, read the file asynchronously inside the handler, and send the response from the callback the loop stays free. | 25→24 | **0.436** | above UPPER → **correct** | +0.076 above UPPER (0.36) | `none` (not incorrect) | 0 |

Cleaning removed almost nothing — verified by token diff: window 1 lost 3 tokens (`Okay,`, `actually`, `so`),
windows 2, 3, 4 and 5 lost nothing at all, window 6 lost one (`so`), window 7 lost one (`so`).
The two analogy turns (T6, T7) were embedded verbatim, 86 and 68 words, no filler present at all.

Warning-engine side: `run_warning_decision` was reached seven times. No window carried
`classification == "incorrect"`, so every call took the `else` branch that resets
`consecutive_incorrect = 0` and returned `{"action": "none", "consecutive_incorrect": 0}`.
`warning_history` stayed empty. The warning engine's end-of-session log:

```
[Barter 9905  SESSION END]
→  Windows: 7 total  0 incorrect  (0.0% off-topic)
→  Warnings issued: 0  max_consecutive: 0
```

## 3. Per-turn cosine, each turn scored alone against the topic

Same path as a window: `clean_text(text)` → `model.encode` → cosine vs the topic embedding.

| Turn | Speaker | Role | Words raw→cleaned | Cosine | Would classify as (in isolation) |
|---|---|---|---|---|---|
| T1 | C | teacher | 27→25 | **0.773** | `correct` |
| T2 | D | learner | 13→13 | **0.393** | `correct` |
| T3 | C | teacher | 31→30 | **0.607** | `correct` |
| T4 | D | learner | 13→12 | **0.285** | `weakly_correct` |
| T5 | C | teacher | 28→28 | **0.272** | `weakly_correct` |
| T6 | C | teacher | 86→86 | **0.172** | `weakly_correct` ⬅ pure analogy |
| T7 | C | teacher | 68→68 | **0.155** | `weakly_correct` ⬅ pure analogy |
| T8 | D | learner | 7→6 | **0.131** | `incorrect` |
| T9 | C | teacher | 47→47 | **0.478** | `correct` |
| T10 | D | learner | 10→10 | **0.292** | `weakly_correct` |
| T11 | C | teacher | 34→34 | **0.399** | `correct` |
| T12 | D | learner | 14→13 | **0.403** | `correct` |
| T13 | C | teacher | 35→35 | **0.650** | `correct` |
| T14 | D | learner | 5→5 | **0.055** | `incorrect` |
| T15 | C | teacher | 31→31 | **0.414** | `correct` |
| T16 | D | learner | 9→9 | **0.379** | `correct` |
| T17 | C | teacher | 28→27 | **0.259** | `weakly_correct` |
| T18 | C | teacher | 36→36 | **0.412** | `correct` |
| T19 | D | learner | 14→14 | **0.151** | `weakly_correct` |
| T20 | C | teacher | 25→24 | **0.436** | `correct` |

Only learner turns T8 (`0.131`, "Okay, and the waiter is the thread?") and T14 (`0.055`,
"And standing at the stove is what, a while loop?") fall below `LOWER`. Learner turns are never
scored for topic relevance by the service — they are routed to engagement only — so neither
influenced any warning. Every teacher turn scored >= 0.155 in isolation.

## 4. Engagement scoring (learner turns) and end-of-session summary

| Learner turn | Words | Duration s | cum questions | cum acks | learner share of speech | Score returned by `/ingest/segment` |
|---|---|---|---|---|---|---|
| T2 | 13 | 5.2 | 1 | 0 | 32.5% | **0.500** |
| T4 | 13 | 5.2 | 3 | 0 | 31.0% | **0.800** |
| T8 | 7 | 2.8 | 4 | 1 | 12.1% | **0.736** |
| T10 | 10 | 4.0 | 6 | 1 | 13.0% | **0.726** |
| T12 | 14 | 5.6 | 9 | 1 | 15.1% | **0.757** |
| T14 | 5 | 2.0 | 12 | 1 | 14.8% | **0.737** |
| T16 | 9 | 3.6 | 14 | 1 | 15.5% | **0.745** |
| T19 | 14 | 5.6 | 15 | 1 | 15.9% | **0.746** |

T2 returned the 0.5 "not enough data yet" placeholder (total speech was 16.0 s < 30 s). T4 peaked at
0.800. From T8 on the score sat in a narrow 0.726–0.757 band: the learner's speaking share fell to
12–16% (below the 20–40% ideal, so `ratio_score` is in the interpolated 0.5–1.0 region), question
rate stayed saturated, and acknowledgments were scarce (1 across 8 segments).
Nothing ever dropped below 0.3, so `post_engagement_alert` was never called — no low-engagement
alert exists in this run.

End-of-session summary payload `semantic_analysis` sent to the (absent) backend:

```json
{
  "learner_engagement_score": 0.746,
  "learner_speaking_seconds": 34.0,
  "teacher_speaking_seconds": 190.4,
  "learner_question_count": 15,
  "learner_acknowledgment_count": 1,
  "learner_segment_count": 8
}
```

Session close-out: `POST /session/9905/end` on 8002 → `{"status":"ended","barter_id":9905}`;
log line `Session 9905 ended — buffer flushed, engagement summary posted` (the "posted" claim is
unverified — see the silent 404 above).

## 5. Findings — how the system handled the sustained analogy

**Short answer: the analogy was seen, scored dramatically lower than everything around it, and then
absorbed by the `weakly_correct` band. Nothing was flagged, nothing escalated, no warning of any
kind was issued.**

1. **The analogy was not diluted by buffering — it got its own windows.** Both analogy turns are
   long enough to trip the 25 s threshold by themselves: T6 is 86 words = 34.4 s and T7 is 68 words =
   27.2 s, each well past `WINDOW_DURATION_THRESHOLD = 25.0`. So window #2 = T6 alone and window #3 =
   T7 alone. There was no neighbouring on-topic speech in either window to rescue them. The buffering
   *hurt* the analogy's chances of being masked here, and it still was not caught. This is the
   cleanest possible test case and the system still returned no flag.

2. **The cosines collapsed exactly where the analogy starts, and by a lot.** Window #1 (T1, T3, T5 —
   real Node content) scored **0.784**. Windows #2 and #3 scored **0.172** and **0.155**. Window #4,
   the moment T9 maps the analogy back to code ("the waiter is the single thread... the ticket rail is
   the queue"), jumps back to **0.481**. The signal is unmistakable — a 0.61 drop and recovery — and
   the classifier reports both analogy windows as `weakly_correct`, the same label a mildly imprecise
   but on-topic explanation would get.

3. **`LOWER = 0.14` is what saved them, by a hair.** Window #3 sat **+0.015 above LOWER**; window #2
   **+0.028 above**. Both are far from `UPPER` (−0.188 and −0.205). Two windows within 0.03 of the
   `incorrect` boundary is not a comfortable margin: had `LOWER` been 0.18 (still well below the
   `UPPER` optimum) both would have classified `incorrect`, consecutively, and the ladder would have
   gone silent → **strong** on window #3.

4. **Escalation never armed.** `consecutive_incorrect` stayed at 0 for the entire session;
   `max_consecutive_incorrect = 0`, `incorrect_windows = 0`, `percent_incorrect = 0.0`,
   `warning_count = 0`. The verdict path recorded a perfectly clean session.

5. **Per-turn diagnosis confirms it is the analogy, not the window size.** Scored in isolation, T6 =
   **0.172** and T7 = **0.155** — identical to their window scores, since each was its own window.
   Every other teacher turn scored 0.259–0.773. The two lowest teacher turns in the whole script are
   exactly the two pure-analogy turns, and the next-lowest teacher turn (T17, 0.259) is 0.087 above
   T6. The embedding is discriminating correctly; the thresholds are simply not placed to act on it.

6. **What would have caught it.** Nothing in the current design would. `weakly_correct` has no
   consequence anywhere in `warning_engine` — only `incorrect` increments `consecutive_incorrect`,
   and `total_drift_incidents` only counts `incorrect`. A sustained, deliberate, zero-vocabulary
   digression that parks two consecutive windows in the bottom 8% of the `weakly_correct` band leaves
   no trace at all in the drift summary. If analogy-vs-drift is meant to be distinguishable, the
   evidence here argues for either a sub-band signal (e.g. two consecutive windows below ~0.20
   counting as a soft drift incident) or for recording `weakly_correct` runs in the summary, rather
   than for moving `LOWER` up — which would also catch genuine, useful analogies.

### Honest caveat

This run used one script, a synthetic word-count-derived duration model, and `scope=""`. A non-empty
scope string would change the topic embedding and hence every cosine. The 0.015/0.028 margins are
tight enough that they should not be treated as stable across scope wordings.
