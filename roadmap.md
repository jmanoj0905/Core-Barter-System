# Core Barter POC — Build Roadmap

> Build order is based on dependencies. Each stage must work before moving to the next.
> Estimated solo build time: **25–35 focused hours (~4–5 days)**

---

## Stage 1 — Project Skeleton
**Goal**: Everything boots. DB connects. No logic yet.
**Est. time**: 2–3 hours

- [ ] Initialize FastAPI project (port 8000)
- [ ] Set up PostgreSQL connection + `.env`
- [ ] Write and run DB migrations (all 6 tables)
- [ ] Seed Alice and Bob with trust scores of 0.30
- [ ] Confirm DB tables exist and seed data is correct
- [ ] Initialize 3 stub services: port 8001 (audio), 8002 (semantic), 8003 (warning) — just `hello world` endpoints for now

**Gate**: All 4 services start without errors. DB has the correct schema.

---

## Stage 2 — Session Lifecycle (Backend)
**Goal**: A barter can be created, started, confirmed, and terminated via API calls only (no frontend yet).
**Est. time**: 4–5 hours

- [ ] `POST /session/create` — creates barter + session contract in DB
- [ ] `POST /session/{id}/start` — sets `started_at`, status → `active`
- [ ] `POST /session/{id}/confirm` — tracks confirmations, marks `completed` when both confirm
- [ ] `GET /session/{id}/status` — returns elapsed time, both_confirmed flag
- [ ] `POST /session/{id}/terminate` — force closes, marks for DISPUTE verdict
- [ ] `POST /session/{id}/drift-summary` — stores JSON from warning engine
- [ ] `POST /verdict/{id}/generate` — rule-based: duration + confirmation checks
- [ ] `POST /trust/{id}/update` — applies trust formula after verdict

**Test**: Use Postman or curl to run through the full lifecycle manually.
`create → start → confirm (user 1) → confirm (user 2) → drift-summary → verdict → trust update`

**Gate**: Full session lifecycle works end to end via API. Trust scores update in DB.

---

## Stage 3 — Warning Engine (Service on port 8003)
**Goal**: Warning logic works in isolation. Can receive fake window results and produce correct warnings.
**Est. time**: 3–4 hours

- [ ] `POST /session/{id}/init` — creates empty session state in memory
- [ ] `POST /window/result` — updates counters, runs warning decision engine
- [ ] Warning logic: 1→silent, 2→mild, 3–4→strong, 5+→severe+terminate
- [ ] Out-of-scope scope_violation → strong warning immediately
- [ ] `POST /warnings/log` call to Person 1's backend on each warning
- [ ] `POST /session/{id}/terminate` call to Person 1 on severe
- [ ] `POST /session/{id}/end` — builds drift summary, POSTs to Person 1

**Test**: Send fake window results manually (mix of correct, incorrect, out_of_scope) and verify:
- Correct warnings fire at the right consecutive counts
- Drift summary is accurate
- Person 1's `/warnings/log` receives the payloads

**Gate**: Warning engine + Person 1 backend are fully connected and working together.

---

## Stage 4 — WebSocket Warning Relay (Backend → Frontend bridge)
**Goal**: Warnings logged by the warning engine broadcast in real time.
**Est. time**: 2 hours

- [ ] WebSocket endpoint at `ws://localhost:8000/ws/warnings/{barter_id}`
- [ ] When `POST /warnings/log` is called, broadcast payload to connected clients
- [ ] Test with a simple browser console WebSocket connection

**Gate**: Open a WebSocket in browser console, POST a warning via Postman, see it appear in the console in real time.

---

## Stage 5 — Semantic Analysis Engine (Service on port 8002)
**Goal**: Sentence-BERT is working. Windows are classified correctly against a topic.
**Est. time**: 5–6 hours

- [ ] Install `sentence-transformers`, load `all-MiniLM-L6-v2`
- [ ] `POST /session/{id}/contract` — stores contract, generates + caches topic embedding
- [ ] `POST /ingest/segment` — appends to buffer, checks if window is ready (~100s)
- [ ] Window assembly: combine segment texts, clean filler words
- [ ] Generate window embedding, compute cosine similarity vs topic embedding
- [ ] Scope violation check (keyword scan against disallowed concepts)
- [ ] Window classification: correct / weakly_correct / incorrect / out_of_scope
- [ ] `POST /window/result` call to Person 4 (port 8003) after each window
- [ ] `POST /session/{id}/end` — flush remaining buffer, notify warning engine

**Test**: Manually POST fake segments (some on-topic, some off-topic) and verify:
- Windows fire at ~100s of accumulated audio time
- On-topic text classifies as correct, off-topic as incorrect
- Out-of-scope keywords trigger scope_violation

**Calibrate thresholds**: Run 5 test windows manually and adjust UPPER (0.55) and LOWER (0.35) thresholds until classification matches expectation.

**Gate**: Semantic engine + warning engine + Person 1 are fully connected. POSTing fake segments produces warnings in Person 1's DB.

---

## Stage 6 — Audio Pipeline (Service on port 8001)
**Goal**: Browser mic audio streams in, Whisper transcribes it, segments reach the semantic engine.
**Est. time**: 5–6 hours

- [ ] Install `openai-whisper`, `ffmpeg` (system), `torch`
- [ ] WebSocket server at `ws://localhost:8001/audio/{barter_id}/{user_id}`
- [ ] Receive binary audio chunks from browser
- [ ] Buffer per `(barter_id, user_id)` until ~20–30 seconds of audio
- [ ] Convert `.webm` → 16kHz mono `.wav` using ffmpeg
- [ ] Run Whisper `base` model on the WAV file
- [ ] Attach speaker label + timestamps
- [ ] Discard empty/silent transcriptions
- [ ] DELETE temp audio files after transcription
- [ ] `POST /ingest/segment` call to port 8002 for each segment
- [ ] `POST /session/{id}/end` endpoint — flush buffers, notify port 8002

**Test**: Connect to the WebSocket from a browser tab, speak for 30 seconds, verify a transcript segment appears in the semantic engine's buffer.

**Gate**: Speaking into the browser mic produces window classifications in the warning engine and, eventually, warnings in Person 1's DB.

---

## Stage 7 — Frontend (React)
**Goal**: A real user can run a full barter session through the browser.
**Est. time**: 5–6 hours

**Screen 1: Session Setup**
- [ ] Form: topic, scope, allowed concepts, disallowed concepts, duration
- [ ] On submit → `POST /session/create` → navigate to Screen 2

**Screen 2: Live Session**
- [ ] "Start Session" button → `POST /session/{id}/start`
- [ ] Mic capture via `MediaRecorder` API → stream binary chunks to `ws://localhost:8001/audio/{id}/{user_id}` every 5 seconds
- [ ] Connect to `ws://localhost:8000/ws/warnings/{id}` → display warning banners
  - Yellow = mild, Orange = strong, Red = severe
- [ ] Live session timer (elapsed vs agreed duration)
- [ ] "Mark Complete" button → `POST /session/{id}/confirm`
- [ ] Poll `GET /session/{id}/status` every 10 seconds → navigate to Screen 3 when `both_confirmed: true`

**Screen 3: Post Session**
- [ ] Fetch `GET /verdict/{id}` → display verdict, duration check, confirmation check
- [ ] Display drift summary: % off-topic, warning count
- [ ] Display trust scores before and after for both users

**Gate**: Can run a complete session from setup to post-session verdict entirely through the browser UI.

---

## Stage 8 — Integration + End-to-End Test
**Goal**: All 4 services run simultaneously and the full pipeline works.
**Est. time**: 3–4 hours

- [ ] Run all 4 services simultaneously
- [ ] Open 2 browser tabs (simulating Alice and Bob)
- [ ] Create session, start it, speak into both mics
- [ ] Verify transcripts flow: Audio → Semantic → Warning → Backend → Frontend
- [ ] Deliberately go off-topic and verify warning banners appear
- [ ] Both users confirm complete → verify verdict and trust scores update

**Known things to debug at this stage**:
- Latency: Whisper on `base` model takes 5–15 seconds per chunk. If too slow, switch to `tiny`.
- Window timing: Windows may fire too early or too late. Adjust the 100s buffer threshold.
- CORS issues between services. Add CORS middleware to all FastAPI services.
- WebSocket disconnect handling. Browser tab refresh shouldn't crash the audio server.

**Gate**: A full barter session runs successfully. Warnings appear during the session. Verdict and trust scores are correct at the end.

---

## Service Port Reference

| Service | Owner | Port |
| -------------------- | -------- | ---- |
| Backend Core + DB    | Person 1 | 8000 |
| Audio Pipeline       | Person 2 | 8001 |
| Semantic Analysis    | Person 3 | 8002 |
| Warning Engine       | Person 4 | 8003 |

---

## Data Flow (End to End)

```
Browser mic
    │ audio chunks (WebSocket)
    ▼
Port 8001 — Whisper STT
    │ transcript segments (HTTP POST)
    ▼
Port 8002 — Sentence-BERT + Sliding Window
    │ window classifications (HTTP POST)
    ▼
Port 8003 — Warning Engine
    │ warnings (HTTP POST)          │ drift summary (HTTP POST)
    ▼                               ▼
Port 8000 — Backend Core ──────────────────────────────────────────────
    │ warning broadcast (WebSocket)     │ verdict + trust update
    ▼                                   ▼
Frontend (browser)                  PostgreSQL DB
```

---

## Build Checklist Summary

| Stage | What | Est. Time |
| ----- | ---- | --------- |
| 1 | Project skeleton + DB schema | 2–3 hrs |
| 2 | Session lifecycle API | 4–5 hrs |
| 3 | Warning engine | 3–4 hrs |
| 4 | WebSocket relay | 2 hrs |
| 5 | Semantic analysis + embeddings | 5–6 hrs |
| 6 | Audio pipeline + Whisper | 5–6 hrs |
| 7 | React frontend (3 screens) | 5–6 hrs |
| 8 | Integration + end-to-end test | 3–4 hrs |
| **Total** | | **29–36 hrs** |
