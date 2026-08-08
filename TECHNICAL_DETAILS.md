# Core Barter System -- Technical Documentation

> A real-time audio conversation monitoring platform that enforces topic adherence
> during barter/negotiation sessions using speech-to-text, semantic similarity
> analysis, and escalating warning systems.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Service Breakdown](#4-service-breakdown)
5. [Key Data Flows](#5-key-data-flows)
6. [ML Models](#6-ml-models)
7. [Domain Concepts](#7-domain-concepts)
8. [LAN Deployment](#8-lan-deployment)
9. [Database Schema](#9-database-schema)
10. [Security Features](#10-security-features)

---

## 1. Project Overview

### The Problem

In barter and knowledge-exchange platforms, two users agree to teach each other a
skill within a time-limited session. Without enforcement, participants may drift
off-topic, waste each other's time, or engage in bad-faith behavior. There is no
reliable automated mechanism to verify that both parties honored the agreed-upon
subject matter.

### The Solution

The Core Barter System monitors live audio conversations in real time. It
transcribes speech, computes the semantic similarity between what is being said
and the agreed session topic, classifies conversation windows as on-topic or
off-topic, and issues escalating warnings when drift is detected. At session end,
the system produces a verdict (SUCCESSFUL, PARTIAL, or DISPUTE) and updates each
participant's trust score accordingly.

### Key Capabilities

- **Real-time speech-to-text** using faster-whisper (CTranslate2 int8 quantized).
- **Semantic topic analysis** using Sentence-BERT cosine similarity against the
  session contract.
- **Escalating warnings** from silent monitoring through mild, strong, and severe
  alerts, with auto-termination for sustained drift.
- **Scope violation detection** using keyword matching against disallowed concepts.
- **Toxicity detection** via the Mistral Moderation API for harmful speech.
- **NSFW video frame detection** using NudeNet for visual content safety.
- **Learner engagement scoring** tracking speaking ratios, question frequency, and
  acknowledgment patterns.
- **WebRTC peer-to-peer video** between participants with signaling relay.
- **Trust score system** that rewards on-topic, mutually confirmed sessions and
  penalizes disputes.

---

## 2. System Architecture

The system is composed of four FastAPI microservices and a React single-page
application. Services communicate via HTTP POST for pipeline processing and
WebSockets for real-time data delivery to the browser.

### Architecture Diagram

```
+----------------------------------------------------------+
|                     BROWSER (React SPA)                  |
|                     https://host:5173                     |
|                                                          |
|  Setup.jsx ──> LiveSession.jsx ──> PostSession.jsx       |
|      |              |     |     |          |              |
|      |   WebRTC     |     |     |          |              |
|      |   P2P Video  |     |     |          |              |
|      |   & Audio    |     |     |          |              |
+------|--------------+-----+-----+----------|--------------+
       |              |     |     |          |
  REST |   WebSocket  |     |     |     REST |
  POST |   (signal)   |     |     |     GET  |
       |              |     |     |          |
       v              v     |     v          v
+------+------------------+ | +--+-----------+--------------+
| PORT 8000               | | |                             |
| BACKEND CORE            | | | REST: /session/create       |
| FastAPI + PostgreSQL    | | |       /session/{id}/start   |
|                         | | |       /session/{id}/confirm |
| - REST API (routes.py)  | | |       /verdict/{id}         |
| - WebSocket Manager     | | |       /trust/{id}/update    |
| - Signaling Server      | | |       /safety/check-frame  |
| - NudeNet Safety        | | |                             |
| - DB CRUD Operations    | | | WS: /ws/warnings/{id}      |
+------+------------------+ | |     /ws/signal/{id}/{uid}   |
       ^                    | +-----------------------------+
       |                    |
       | HTTP POST          | WebSocket (audio chunks)
       | /warnings/log      |
       | /window/result     |
       | /session/drift     v
       | /session/transcript|
       |              +-----+---------------------------+
       |              | PORT 8001                       |
       |              | AUDIO PIPELINE                  |
       |              | FastAPI + faster-whisper         |
       |              |                                 |
       |              | - WS: /audio/{barter}/{user}    |
       |              | - MediaRecorder chunk ingestion |
       |              | - FFmpeg webm-to-wav conversion |
       |              | - Whisper STT transcription     |
       |              | - Mistral Moderation (toxicity) |
       |              | - 15s audio buffering strategy  |
       |              +-----+---------------------------+
       |                    |
       |                    | HTTP POST /ingest/segment
       |                    v
       |              +-----+---------------------------+
       |              | PORT 8002                       |
       |              | SEMANTIC ANALYSIS               |
       |              | FastAPI + Sentence-BERT         |
       |              |                                 |
       |              | - Contract registration         |
       |              | - Topic embedding (cached)      |
       |              | - Cosine similarity scoring     |
       |              | - Window classification         |
       |              | - Filler word stripping         |
       |              | - Engagement scoring (learner)  |
       |              | - 30s window accumulation       |
       |              +-----+---------------------------+
       |                    |
       |                    | HTTP POST /window/result
       |                    v
       |              +-----+---------------------------+
       |              | PORT 8003                       |
       |              | WARNING ENGINE                  |
       |              | FastAPI                         |
       |              |                                 |
       |              | - Escalation logic              |
       |              | - Consecutive drift tracking    |
       |              | - Scope violation handling      |
       |              | - Safety alert reception        |
       |              | - Drift summary generation      |
       |              +--------------------------------+
```

### Communication Protocols

| From              | To                | Protocol       | Purpose                                    |
|-------------------|-------------------|----------------|--------------------------------------------|
| Browser           | Backend (8000)    | HTTPS REST     | Session CRUD, verdicts, trust, safety       |
| Browser           | Backend (8000)    | WSS            | Warning/transcript broadcast, signaling     |
| Browser           | Audio Pipe (8001) | WSS            | Raw audio chunk streaming                   |
| Browser           | Browser           | WebRTC (P2P)   | Peer-to-peer video and audio                |
| Audio Pipe (8001) | Semantic (8002)   | HTTP POST      | Transcript segments for analysis            |
| Audio Pipe (8001) | Backend (8000)    | HTTP POST      | Transcript storage                          |
| Audio Pipe (8001) | Warning (8003)    | HTTP POST      | Toxicity safety alerts                      |
| Semantic (8002)   | Warning (8003)    | HTTP POST      | Window classification results               |
| Semantic (8002)   | Backend (8000)    | HTTP POST      | Engagement summaries                        |
| Warning (8003)    | Backend (8000)    | HTTP POST      | Warnings, window results, drift summaries   |

### Data Flow Direction

The processing pipeline is strictly linear for topic analysis:

```
Browser --> Audio Pipeline --> Semantic Analysis --> Warning Engine --> Backend Core --> Browser
```

Safety and engagement flows have their own lateral paths back to the warning engine
and backend, as documented in Section 5.

---

## 3. Tech Stack

### Backend Services

| Technology               | Version / Variant        | Purpose                                              |
|--------------------------|--------------------------|------------------------------------------------------|
| Python                   | 3.11+                    | All four microservices                                |
| FastAPI                  | Latest                   | Async web framework for all services                  |
| Uvicorn                  | Latest                   | ASGI server                                           |
| SQLAlchemy               | 2.x (async)             | ORM with asyncpg driver                               |
| Pydantic                 | v2 (pydantic-settings)  | Request/response validation, configuration            |
| httpx                    | Latest                   | Async HTTP client for inter-service communication     |
| PostgreSQL               | 16                       | Persistent data store                                 |
| faster-whisper           | Latest (CTranslate2)    | Speech-to-text inference                              |
| sentence-transformers    | Latest                   | Sentence-BERT embedding model                         |
| PyTorch                  | Latest                   | ML runtime for Sentence-BERT                          |
| NudeNet                  | Latest                   | NSFW image detection                                  |
| FFmpeg                   | System install           | Audio format conversion (webm to wav)                 |
| python-dotenv            | Latest                   | Environment variable loading                          |

### Frontend

| Technology               | Purpose                                              |
|--------------------------|------------------------------------------------------|
| React                    | UI framework (functional components, hooks)          |
| Vite                     | Build tool and dev server with HTTPS + reverse proxy |
| @vitejs/plugin-basic-ssl | Self-signed TLS certificate generation               |
| WebRTC (RTCPeerConnection) | Peer-to-peer video and audio                       |
| MediaRecorder API        | Browser audio capture and chunked streaming          |
| WebSocket API            | Audio streaming and warning/transcript reception     |
| Canvas API               | Video frame capture for NSFW detection               |

### External APIs

| API                      | Provider    | Purpose                                  |
|--------------------------|-------------|------------------------------------------|
| Mistral Moderation API   | Mistral AI  | Toxicity detection on transcript text     |
| Mistral Small (chat)     | Mistral AI  | AI-generated disallowed concept suggestions |
| Google STUN servers      | Google      | WebRTC NAT traversal for ICE candidates   |

### Infrastructure

| Component                | Purpose                                              |
|--------------------------|------------------------------------------------------|
| Homebrew                 | macOS package manager for system dependencies        |
| PostgreSQL 16            | Database server (managed via Homebrew services)      |
| Python venv              | Isolated virtual environments per microservice       |
| npm                      | Frontend dependency management                       |
| Bash (start.sh)          | Unified startup script for all services              |

---

## 4. Service Breakdown

### 4.1 Backend Core (Port 8000)

**Source files**: `backend/app/main.py`, `backend/app/routes.py`,
`backend/app/websocket.py`, `backend/app/safety.py`, `backend/app/models.py`,
`backend/app/schemas.py`, `backend/app/database.py`, `backend/app/config.py`

The backend core is the central hub of the system. It owns the database, serves
the REST API to the frontend, manages WebSocket connections for real-time
broadcasting, and runs the WebRTC signaling server.

#### Startup Lifecycle

On application startup (via the `lifespan` context manager), the backend
initializes the NudeNet detector model for NSFW frame checking. If NudeNet fails
to load, it logs a warning and continues with NSFW checks disabled (fail-open).

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.safety import init_detector
    try:
        init_detector()
    except Exception as e:
        logger.warning("NudeNet init failed (NSFW checks disabled): %s", e)
    yield
```

#### REST API Endpoints

**Session Lifecycle**

| Method | Endpoint                         | Purpose                                              |
|--------|----------------------------------|------------------------------------------------------|
| POST   | `/session/create`                | Create session + contract, register with services    |
| POST   | `/session/{id}/start`            | Set session to active, record start timestamp         |
| POST   | `/session/{id}/confirm`          | Record user confirmation; end session if both confirm |
| GET    | `/session/{id}/status`           | Poll session state (elapsed time, confirmations)     |
| POST   | `/session/{id}/terminate`        | Force-terminate a session                            |
| POST   | `/session/suggest-concepts`      | AI-generated disallowed concepts via Mistral Small   |

**Data Ingestion (from downstream services)**

| Method | Endpoint                              | Source           | Purpose                          |
|--------|---------------------------------------|------------------|----------------------------------|
| POST   | `/window/result`                      | Warning Engine   | Store window result, broadcast   |
| POST   | `/warnings/log`                       | Warning Engine   | Store warning, broadcast via WS  |
| POST   | `/session/{id}/drift-summary`         | Warning Engine   | Store drift summary in verdict   |
| POST   | `/session/{id}/transcript`            | Audio Pipeline   | Store transcript segment         |
| POST   | `/session/{id}/engagement-summary`    | Semantic Analysis| Store engagement data in verdict |

**Verdicts and Trust**

| Method | Endpoint                    | Purpose                                              |
|--------|-----------------------------|------------------------------------------------------|
| POST   | `/verdict/{id}/generate`    | Compute verdict from duration + confirmation checks  |
| GET    | `/verdict/{id}`             | Retrieve verdict with drift summary                  |
| POST   | `/trust/{id}/update`        | Compute and apply trust score deltas                 |

**Data Retrieval**

| Method | Endpoint                        | Purpose                                  |
|--------|---------------------------------|------------------------------------------|
| GET    | `/session/{id}/windows`         | All window results for a session          |
| GET    | `/session/{id}/transcript`      | Full transcript segments for a session    |

**Safety**

| Method | Endpoint             | Purpose                                          |
|--------|----------------------|--------------------------------------------------|
| POST   | `/safety/check-frame`| NudeNet NSFW check on base64 video frame         |

#### WebSocket Endpoints

| Endpoint                          | Purpose                                          |
|-----------------------------------|--------------------------------------------------|
| `/ws/warnings/{barter_id}`        | Broadcast warnings, window results, transcripts  |
| `/ws/signal/{barter_id}/{user_id}`| WebRTC signaling relay (SDP + ICE candidates)    |

#### WebSocket Connection Manager

The `ConnectionManager` class (`websocket.py`) maintains a dictionary mapping
`barter_id` to a list of connected WebSocket clients. It supports:

- **connect**: Accept and register a WebSocket for a session.
- **disconnect**: Remove a WebSocket; clean up if no clients remain.
- **broadcast**: Send a JSON payload to all connected clients for a session.
  Dead connections are detected and removed automatically.

```python
class ConnectionManager:
    def __init__(self):
        self._connections: dict[int, list[WebSocket]] = defaultdict(list)

    async def broadcast(self, barter_id: int, payload: dict):
        message = json.dumps(payload)
        dead = []
        for ws in self._connections.get(barter_id, []):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(barter_id, ws)
```

#### WebRTC Signaling Server

The signaling server is a lightweight WebSocket-based relay implemented directly
in `main.py`. It maintains an in-memory dictionary:

```python
signal_peers: dict[int, dict[int, WebSocket]] = {}
# Structure: { barter_id: { user_id: WebSocket } }
```

When a user connects to `/ws/signal/{barter_id}/{user_id}`:

1. The server registers the user's WebSocket in `signal_peers`.
2. It notifies all already-connected peers with a `peer_joined` message.
3. It relays all subsequent messages (SDP offers, answers, ICE candidates) to the
   other peer by looking up the target user ID (simple toggle: if sender is user 1,
   target is user 2, and vice versa).

#### Session Creation Side Effects

When `POST /session/create` is called, the backend performs three operations:

1. Creates a `BarterSession` row (status: "proposed") and a `SessionContract` row
   in PostgreSQL.
2. Sends an HTTP POST to the Semantic Analysis service at
   `http://localhost:8002/session/{id}/contract` to register the contract and
   pre-compute the topic embedding.
3. Sends an HTTP POST to the Warning Engine at
   `http://localhost:8003/session/{id}/init` to initialize the escalation state.

Both downstream calls are fire-and-forget (errors are caught and ignored) to allow
the session to be created even if downstream services are not yet running.

#### Session Confirmation Logic

When a user calls `POST /session/{id}/confirm`:

1. A `Confirmation` row is inserted for the user.
2. The system checks if both users (2 total) have confirmed.
3. If both confirmed:
   - Session status is set to "completed" with `ended_at` timestamp.
   - An HTTP POST is sent to `http://localhost:8001/session/{id}/end` to flush
     audio buffers and trigger the end-of-session pipeline.

---

### 4.2 Audio Pipeline (Port 8001)

**Source file**: `audio_pipeline/main.py`

The audio pipeline is the entry point for all audio data. It receives raw audio
chunks from the browser via WebSocket, converts them to WAV, runs speech-to-text,
checks for toxicity, and forwards transcript segments downstream.

#### Startup Lifecycle

On startup, the service loads the faster-whisper model (`base`, CTranslate2 int8
quantization) and creates an `httpx.AsyncClient` for downstream communication.

```python
whisper_model = WhisperModel("base", compute_type="int8")
http_client = httpx.AsyncClient(timeout=30.0)
```

#### WebSocket Audio Ingestion

**Endpoint**: `ws://localhost:8001/audio/{barter_id}/{user_id}`

The browser's `MediaRecorder` captures audio in webm/opus format and sends binary
chunks every 5 seconds over WebSocket. Each connected user gets their own buffer:

```python
buffers: dict[tuple[int, int], dict] = {}
# Key: (barter_id, user_id)
# Value: { chunks, header_chunk, accumulated_seconds, segment_counter, wall_start }
```

**Buffering strategy**:

- The first chunk received is saved as `header_chunk` (contains the webm container
  header, which must be prepended to every concatenated segment for FFmpeg to
  decode it).
- Chunks accumulate until `accumulated_seconds >= BUFFER_THRESHOLD_SECONDS`
  (15 seconds).
- When the threshold is reached, the buffer is processed and reset.
- On WebSocket disconnect, any remaining buffered audio is flushed and processed.

#### Audio Processing Pipeline

When a buffer triggers processing:

1. **Concatenation**: All buffered chunks are joined into a single byte string.
2. **FFmpeg conversion**: The webm bytes are written to a temp file and converted
   to 16kHz mono WAV via subprocess call:
   ```bash
   ffmpeg -y -i input.webm -ar 16000 -ac 1 -f wav output.wav
   ```
3. **Whisper transcription**: The WAV file is passed to `whisper_model.transcribe()`
   with `language="en"`. Segments are joined into a single text string.
4. **Toxicity check**: The transcript text is sent to the Mistral Moderation API.
   If flagged, a safety alert is posted to the Warning Engine at
   `POST http://localhost:8003/safety/alert`.
5. **Transcript storage**: The segment is sent to
   `POST http://localhost:8000/session/{id}/transcript` for database persistence
   and real-time broadcast.
6. **Semantic forwarding**: The segment is sent to
   `POST http://localhost:8002/ingest/segment` for topic analysis, regardless of
   toxicity status.

#### Session End Handling

When `POST /session/{id}/end` is called (by the backend after both users confirm):

1. All audio buffers for the session are flushed (remaining audio is transcribed).
2. The semantic analysis service is notified at
   `POST http://localhost:8002/session/{id}/end`.

---

### 4.3 Semantic Analysis (Port 8002)

**Source file**: `semantic_analysis/main.py`

The semantic analysis service is the intelligence core. It computes embeddings,
measures topic relevance, classifies conversation windows, and tracks learner
engagement.

#### Startup Lifecycle

On startup, the service loads the `all-MiniLM-L6-v2` Sentence-BERT model and
creates an HTTP client:

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
http_client = httpx.AsyncClient(timeout=10.0)
```

#### In-Memory State

The service maintains three dictionaries:

```python
contracts: dict[int, dict] = {}       # barter_id -> contract + cached topic embedding
buffers: dict[int, dict] = {}         # barter_id -> window buffer (teacher segments)
engagement_state: dict[int, dict] = {} # barter_id -> learner engagement tracking
```

#### Contract Registration

**Endpoint**: `POST /session/{barter_id}/contract`

When a contract is registered, the service:

1. Concatenates the topic and scope: `"{topic}. {scope}"`.
2. Computes and caches the topic embedding using Sentence-BERT.
3. Stores allowed/disallowed concepts, and the teacher/learner user IDs.
4. Initializes a fresh window buffer and engagement state.

#### Segment Routing (Teacher vs Learner)

**Endpoint**: `POST /ingest/segment`

Incoming transcript segments are routed based on the user's role:

- **Teacher segments**: Enter the topic relevance pipeline. Segments are buffered
  until accumulated audio duration reaches `WINDOW_DURATION_THRESHOLD` (30 seconds),
  at which point a window is processed.
- **Learner segments**: Enter the engagement scoring pipeline (no topic analysis
  is performed on learner speech).

#### Window Processing (Topic Analysis)

When a teacher buffer reaches 30 seconds:

1. **Text combination**: All buffered segment texts are concatenated.
2. **Filler word stripping**: Common filler words are removed:
   ```
   uh, um, er, ah, like, you know, i mean, basically, literally,
   actually, so, well, okay
   ```
3. **Embedding**: The cleaned text is encoded using Sentence-BERT.
4. **Cosine similarity**: Similarity is computed between the window embedding and
   the cached topic embedding.
5. **Scope violation check**: The raw text is scanned for disallowed concept
   keywords (case-insensitive substring match).
6. **Classification**: The window receives one of four classifications:

   | Classification    | Condition                                      |
   |-------------------|------------------------------------------------|
   | `out_of_scope`    | Disallowed concept keyword found in text       |
   | `correct`         | Cosine similarity >= 0.55 (UPPER threshold)    |
   | `weakly_correct`  | 0.35 <= similarity < 0.55                      |
   | `incorrect`       | Cosine similarity < 0.35 (LOWER threshold)     |

7. **Forward**: The classification, similarity score, scope violation flag, and
   text preview are sent to the Warning Engine at `POST /window/result`.

#### Engagement Scoring (Learner Analysis)

The engagement system tracks learner participation quality through three signals:

**Speaking ratio** (50% weight):
- Ideal range: 20-40% of total speaking time belongs to the learner.
- Below 10% scores 0.2 (learner is passive).
- 10-20% is linearly scaled.
- Above 40% is penalized (learner may be dominating).

**Question frequency** (30% weight):
- Target rate: 1 question per 3 minutes (rate of 0.33/min).
- Questions are detected via regex:
  ```
  \b(how|what|why|when|where|which|can you|could you|is it|are there)\b|\?
  ```

**Acknowledgment presence** (20% weight):
- Acknowledgment words are detected via substring matching:
  ```
  uh-huh, okay, right, yes, yeah, got it, i see, makes sense,
  interesting, cool
  ```
- Score is calculated as `min(1.0, acknowledgments_per_segment * 2.0)`.

**Combined formula**:
```
engagement = (ratio_score * 0.5) + (question_score * 0.3) + (ack_score * 0.2)
```

If the score drops below 0.3, an engagement alert is posted to the Warning Engine,
subject to a 120-second cooldown between alerts.

#### Session End Handling

When `POST /session/{barter_id}/end` is called:

1. The remaining teacher buffer is flushed as a final window.
2. The Warning Engine is notified at `POST /session/{id}/end` to generate the
   drift summary (this creates the verdict row in the database).
3. The engagement summary is posted to the backend at
   `POST /session/{id}/engagement-summary` (after the verdict row exists).
4. All in-memory state for the session (contracts, buffers, engagement) is cleaned up.

---

### 4.4 Warning Engine (Port 8003)

**Source file**: `warning_engine/main.py`

The warning engine is the decision-making layer for escalation. It receives window
classifications, applies escalation rules, and forwards warnings and results to
the backend.

#### In-Memory State

```python
sessions: dict[int, dict] = {}
# Per-session state:
# {
#     total_windows, incorrect_windows, consecutive_incorrect,
#     max_consecutive_incorrect, total_drift_incidents,
#     warning_history, terminated
# }
```

#### Endpoints

| Method | Endpoint                       | Purpose                                          |
|--------|--------------------------------|--------------------------------------------------|
| POST   | `/session/{id}/init`           | Initialize session escalation state              |
| POST   | `/window/result`               | Receive window classification, run decision logic|
| POST   | `/safety/alert`                | Receive toxicity/NSFW safety alerts              |
| POST   | `/engagement/alert`            | Receive low-engagement alerts                    |
| POST   | `/session/{id}/end`            | Generate drift summary, clean up                 |

#### Warning Decision Engine

The `run_warning_decision` function is the core escalation logic. It processes
each incoming window result as follows:

1. **Guard**: If the session is already terminated, ignore the window.
2. **Counter update**: Increment `total_windows`. If the classification is
   `incorrect` or `out_of_scope`, increment `consecutive_incorrect` and
   `incorrect_windows`. Otherwise, reset `consecutive_incorrect` to 0.
3. **Severity determination**:

   | Condition                     | Severity | Action                              |
   |-------------------------------|----------|-------------------------------------|
   | Scope violation (keyword)     | `strong` | Immediate warning, overrides count  |
   | 3+ consecutive off-topic      | `severe` | Warning issued                      |
   | 2 consecutive off-topic       | `strong` | Warning issued                      |
   | 1 consecutive off-topic       | (silent) | Window forwarded, no warning        |
   | On-topic (correct/weakly)     | (none)   | Window forwarded, no warning        |

4. **Forwarding**: Every window result is posted to the backend at
   `POST /window/result` for database storage and WebSocket broadcast.
5. **Warning logging**: If a warning is issued, it is recorded in the session's
   `warning_history` and posted to `POST /warnings/log` on the backend for
   persistence and broadcast.

#### Safety Alert Handling

Safety alerts from the audio pipeline (toxicity) or backend (NSFW) are processed
separately from the topic escalation pipeline:

- If `hard_block` is true in the alert details, severity is `severe`.
- Otherwise, severity is `strong`.
- The warning is recorded in session state and forwarded to the backend.

Safety warnings do not affect the `consecutive_incorrect` counter (they are
orthogonal to topic drift).

#### Engagement Alert Handling

Low-engagement alerts from the semantic analysis service are issued as `mild`
warnings. These inform participants that the learner may not be actively engaged
but do not trigger escalation.

#### Drift Summary Generation

When `POST /session/{id}/end` is called, the engine computes:

```python
drift_summary = {
    "barter_id": barter_id,
    "total_windows": total,
    "incorrect_windows": incorrect,
    "percent_incorrect": (incorrect / total) * 100,
    "max_consecutive_incorrect": state["max_consecutive_incorrect"],
    "total_drift_incidents": state["total_drift_incidents"],
    "warning_count": len(state["warning_history"]),
    "warnings": state["warning_history"],
    "terminated_early": state["terminated"],
}
```

This summary is posted to `POST /session/{id}/drift-summary` on the backend, which
stores it in the `Verdict` row as a JSON string.

---

### 4.5 Frontend (Port 5173)

**Source files**: `frontend/src/App.jsx`, `frontend/src/screens/Setup.jsx`,
`frontend/src/screens/LiveSession.jsx`, `frontend/src/screens/PostSession.jsx`,
`frontend/vite.config.js`

The frontend is a React single-page application served via Vite with HTTPS enabled.
It manages a three-screen flow: Setup, LiveSession, and PostSession.

#### Screen Flow

```
App.jsx (state machine)
  |
  |-- screen='setup'   --> Setup.jsx
  |-- screen='live'    --> LiveSession.jsx
  |-- screen='post'    --> PostSession.jsx
```

State transitions:
- Setup --> LiveSession: when `onSessionCreated(barterId, minutes, userId)` fires.
- LiveSession --> PostSession: when `onComplete(barterId)` fires (both confirmed).
- PostSession --> Setup: when the user clicks "New Session".

#### Setup Screen (Setup.jsx)

Provides a form for session configuration:

- **User selection**: Alice (user_id=1) or Bob (user_id=2) via radio buttons.
- **Alice flow**: Creates a new session by filling in topic, scope, allowed concepts,
  disallowed concepts, and duration. Calls `POST /session/create`.
- **Bob flow**: Joins an existing session by entering a barter ID.
- **AI Suggest button**: Calls `POST /session/suggest-concepts` to auto-generate
  disallowed concepts using Mistral Small.

#### Live Session Screen (LiveSession.jsx)

The primary operational screen. It manages:

**WebSocket connections**:
- Warnings/transcript WebSocket: connects to `wss://host/ws/warnings/{barterId}`
  and dispatches incoming messages by type (`window`, `transcript`, or warning).
- Audio WebSocket: connects to `wss://host/audio/{barterId}/{userId}` and streams
  MediaRecorder chunks.

**WebRTC peer connection**:
- Creates an `RTCPeerConnection` with Google STUN servers.
- Connects to the signaling WebSocket at `wss://host/ws/signal/{barterId}/{userId}`.
- Implements the full SDP offer/answer exchange with ICE candidate queuing.
- User 1 (Alice) is always the offerer; user 2 (Bob) answers.
- ICE candidates that arrive before `setRemoteDescription` are queued and flushed.

**MediaRecorder**:
- Captures audio tracks from the media stream.
- Prefers `audio/webm;codecs=opus`, with fallback to other supported formats.
- Records in 5-second intervals (`mr.start(5000)`), sending each chunk over the
  audio WebSocket.

**NSFW frame capture**:
- Every 10 seconds, the local video element is drawn to an off-screen canvas.
- The canvas is exported as a JPEG at 50% quality and base64-encoded.
- Sent to `POST /safety/check-frame` for NudeNet analysis.

**Live UI elements**:
- Timer with overtime indicator.
- Side-by-side video feeds (local + remote) with name labels.
- Microphone status indicator (live/off).
- Live transcript display (most recent 30 segments, reverse chronological).
- Warning feed with severity-based styling.
- Window results with classification, similarity score, and text preview.
- "Mark Complete" and "Terminate" buttons.

**Session completion polling**:
- Every 10 seconds, polls `GET /session/{id}/status` to check if both users have
  confirmed, triggering transition to PostSession.

#### Post-Session Screen (PostSession.jsx)

Displays the session verdict and analytics across four tabs:

**Verdict tab**:
- Verdict badge (SUCCESSFUL / PARTIAL / DISPUTE).
- Duration check (pass/fail pill).
- Confirmation check (pass/fail pill).
- On-topic percentage, warning count, total windows, max consecutive off-topic.
- Trust score deltas for both users (signed, color-coded).
- Warning log with severity and timestamp.

**Windows tab**:
- All window results with ID, classification (color-coded), similarity score,
  and text preview.

**Transcript tab**:
- Full transcript with speaker labels (color-differentiated).

**Engagement tab**:
- Learner engagement score, speaking seconds (learner vs teacher), question count,
  acknowledgment count, and segment count.

**Verdict generation flow** (called on mount):
1. `POST /verdict/{id}/generate` -- computes verdict from checks.
2. `POST /trust/{id}/update` -- applies trust score deltas.
3. Parallel fetch of verdict, transcript, and windows data.

#### Vite Configuration

```javascript
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: '0.0.0.0',       // Bind to all interfaces (LAN accessible)
    port: 5173,
    https: true,            // Self-signed TLS via @vitejs/plugin-basic-ssl
    proxy: {
      '/session':  { target: 'http://localhost:8000', changeOrigin: true },
      '/verdict':  { target: 'http://localhost:8000', changeOrigin: true },
      '/trust':    { target: 'http://localhost:8000', changeOrigin: true },
      '/safety':   { target: 'http://localhost:8000', changeOrigin: true },
      '/window':   { target: 'http://localhost:8000', changeOrigin: true },
      '/warnings': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws':       { target: 'ws://localhost:8000',   ws: true },
      '/audio':    { target: 'ws://localhost:8001',   ws: true },
    },
  },
})
```

All REST API calls from the browser use relative paths (e.g., `/session/create`).
Vite's reverse proxy routes them to the appropriate backend service. WebSocket
connections to `/ws/*` are proxied to port 8000, and `/audio/*` is proxied to
port 8001.

---

## 5. Key Data Flows

### 5.1 Audio Chunk to Frontend Warning (End-to-End)

This is the primary pipeline. It traces a single audio chunk from the browser
microphone to a warning displayed on screen.

```
Step 1: Browser captures 5s audio chunk via MediaRecorder
        --> Sends binary data over WebSocket to ws://host/audio/{barter_id}/{user_id}

Step 2: Audio Pipeline receives chunk, appends to buffer
        Buffer accumulates until >= 15 seconds of audio

Step 3: Audio Pipeline flushes buffer:
        a) Concatenates chunks (with webm header prepended)
        b) FFmpeg: webm bytes --> 16kHz mono WAV file
        c) faster-whisper: WAV --> English transcript text
        d) Mistral Moderation API: transcript --> toxicity check
        e) POST /session/{id}/transcript to backend (storage + broadcast)
        f) POST /ingest/segment to Semantic Analysis

Step 4: Semantic Analysis receives segment:
        a) Routes based on role (teacher -> topic analysis, learner -> engagement)
        b) For teacher: appends to window buffer
        c) Window buffer accumulates until >= 30 seconds

Step 5: Semantic Analysis processes window:
        a) Concatenates segment texts, strips filler words
        b) Encodes cleaned text via Sentence-BERT
        c) Computes cosine similarity vs cached topic embedding
        d) Checks for disallowed concept keywords
        e) Classifies: correct / weakly_correct / incorrect / out_of_scope
        f) POST /window/result to Warning Engine

Step 6: Warning Engine receives window result:
        a) Updates consecutive_incorrect counter
        b) Determines severity (none / silent / strong / severe)
        c) POST /window/result to Backend (always)
        d) POST /warnings/log to Backend (if warning issued)

Step 7: Backend receives warning/window:
        a) Stores in PostgreSQL (warnings / window_results table)
        b) WebSocket broadcast to all connected clients for that barter_id

Step 8: Browser receives JSON over WebSocket:
        a) Warning messages update the warnings list (severity-styled)
        b) Window messages update the windows display
        c) Severe warning triggers auto-termination of the session
```

**Latency characteristics**: With 5-second MediaRecorder intervals, 15-second
audio buffering, and 30-second window accumulation, the minimum time from speech
to first window classification is approximately 30-45 seconds. Warnings for
sustained drift require at least two consecutive off-topic windows (60-90 seconds).

### 5.2 WebRTC Signaling Flow (Offer/Answer/ICE)

```
Step 1: Alice opens LiveSession, calls navigator.mediaDevices.getUserMedia()
        --> Connects signaling WebSocket to /ws/signal/{barter_id}/1

Step 2: Bob opens LiveSession on another device
        --> Connects signaling WebSocket to /ws/signal/{barter_id}/2

Step 3: Server detects two peers, sends "peer_joined" to both

Step 4: Alice (user_id=1) is the offerer:
        a) pc.createOffer() --> pc.setLocalDescription(offer)
        b) Sends { type: "offer", sdp: localDescription } via signaling WS

Step 5: Server relays offer to Bob (target_id = 2 when sender is 1)

Step 6: Bob receives offer:
        a) pc.setRemoteDescription(offer)
        b) Flushes any queued ICE candidates
        c) pc.createAnswer() --> pc.setLocalDescription(answer)
        d) Sends { type: "answer", sdp: localDescription } via signaling WS

Step 7: Server relays answer to Alice

Step 8: Alice receives answer:
        a) pc.setRemoteDescription(answer)
        b) Flushes any queued ICE candidates

Step 9: ICE candidate exchange (concurrent with steps 4-8):
        a) pc.onicecandidate fires --> send { type: "ice", candidate } via WS
        b) Server relays to other peer
        c) If remote description is not yet set, candidates are queued and
           flushed after setRemoteDescription succeeds

Step 10: RTCPeerConnection enters "connected" state
         --> Remote video/audio stream plays in the partner's video element
```

**ICE servers used**:
```javascript
iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun1.l.google.com:19302' },
]
```

No TURN server is configured. The system relies on direct peer-to-peer
connectivity, which works on LAN or when both peers are behind the same NAT.

### 5.3 Session Lifecycle (Create to Trust Update)

```
Phase 1: CREATION
  Alice fills Setup form --> POST /session/create
    Backend:
      1. Creates BarterSession (status: "proposed")
      2. Creates SessionContract (topic, scope, concepts, duration, roles)
      3. POST /session/{id}/contract to Semantic Analysis (pre-compute embedding)
      4. POST /session/{id}/init to Warning Engine (initialize escalation state)
    Returns: { barter_id, contract_id, status }
    UI transitions to LiveSession screen

Phase 2: START
  Alice clicks "Start Session" --> POST /session/{id}/start
    Backend:
      1. Sets status to "active"
      2. Records started_at timestamp
    Returns: { barter_id, started_at, status }
    Camera + mic activated, WebSocket and WebRTC connections established

Phase 3: ACTIVE SESSION
  Audio pipeline processes speech continuously
  Semantic analysis classifies windows
  Warning engine issues escalating warnings
  Live transcripts and warnings broadcast to both participants

Phase 4: CONFIRMATION
  Each user clicks "Mark Complete" --> POST /session/{id}/confirm
    Backend:
      1. Records Confirmation row for user
      2. Checks if both users confirmed
      3. If both: set status "completed", set ended_at, notify audio pipeline
    Returns: { barter_id, confirmed_by, both_confirmed }

  When audio pipeline receives end notification:
    1. Flush remaining audio buffers
    2. Notify semantic analysis --> semantic flushes window buffer
    3. Semantic notifies warning engine --> warning engine generates drift summary
    4. Warning engine posts drift summary to backend
    5. Semantic posts engagement summary to backend

Phase 5: VERDICT + TRUST
  PostSession screen loads:
    1. POST /verdict/{id}/generate
       - Duration check: actual_seconds >= agreed_seconds * 0.8
       - Confirmation check: both users confirmed
       - Verdict determination:
         * DISPUTE: terminated OR (no duration AND no confirmation)
         * SUCCESSFUL: duration pass AND confirmation pass
         * PARTIAL: duration pass OR confirmation pass (not both)

    2. POST /trust/{id}/update
       - QA score: SUCCESSFUL=1.0, PARTIAL=0.5, DISPUTE=0.0
       - quality_adjusted = (qa_score + 0.8) / 2
       - new_trust = (previous_trust * 0.3) + (quality_adjusted * 0.7)
       - Clamped to [0.0, 1.0]

    3. GET /verdict/{id} + GET /session/{id}/transcript + GET /session/{id}/windows
       - Display verdict, drift analytics, transcript, engagement data
```

### 5.4 Live Transcript Broadcast

```
Audio Pipeline (transcription complete)
  |
  POST /session/{id}/transcript
  Body: { barter_id, user_id, text, duration_seconds, timestamp_start, timestamp_end }
  |
  v
Backend (routes.py: save_transcript_segment)
  |
  1. Insert TranscriptSegment row in PostgreSQL
  2. manager.broadcast(barter_id, {
       type: "transcript",
       user_id: req.user_id,
       speaker: "Alice" or "Bob",
       text: req.text,
     })
  |
  v
WebSocket --> Browser
  |
  LiveSession.jsx: onmessage handler
    if data.type === 'transcript':
      prepend to liveTranscripts (max 30 entries)
```

### 5.5 Safety Pipeline (Toxicity + NSFW)

The safety pipeline operates on two channels in parallel with the topic analysis
pipeline.

#### Toxicity Detection (Text-Based)

```
Audio Pipeline: process_buffer()
  |
  Text transcript produced by Whisper
  |
  POST https://api.mistral.ai/v1/moderations
  Body: { model: "mistral-moderation-latest", input: text }
  |
  Response: { results: [{ flagged: bool, category_scores: {...} }] }
  |
  If flagged AND any category_score >= 0.7:
    POST /safety/alert to Warning Engine (port 8003)
    Body: {
      barter_id, user_id,
      warning_type: "toxicity",
      details: {
        flagged: true,
        categories: { category: score, ... },  // only scores >= 0.7
        hard_block: true/false,                 // true if any score >= 0.9
      }
    }
    |
    Warning Engine: POST /warnings/log to Backend
    Severity: "severe" if hard_block, "strong" otherwise
```

#### NSFW Frame Detection (Video-Based)

```
LiveSession.jsx: setInterval every 10 seconds
  |
  Draw local video frame to canvas
  Export as JPEG (50% quality) --> base64 string
  |
  POST /safety/check-frame to Backend (port 8000)
  Body: { barter_id, user_id, image_base64 }
  |
  Backend: check_frame() via NudeNet
    1. Decode base64 --> temp JPEG file
    2. NudeNet detector.detect(tmp_path)
    3. Filter detections:
       - Class in NSFW_CLASSES AND score >= 0.6
       - NSFW_CLASSES: FEMALE_BREAST_EXPOSED, FEMALE_GENITALIA_EXPOSED,
         MALE_GENITALIA_EXPOSED, BUTTOCKS_EXPOSED, ANUS_EXPOSED
    4. If flagged detections exist:
       POST /safety/alert to Warning Engine
       Body: {
         barter_id, user_id,
         warning_type: "nsfw",
         details: {
           flagged: true,
           detections: [{ class, score }, ...],
           hard_block: true/false,  // true if any score >= 0.9
         }
       }
```

---

## 6. ML Models

### 6.1 faster-whisper base (Local STT)

| Property      | Value                                          |
|---------------|------------------------------------------------|
| Model         | OpenAI Whisper "base" (74M parameters)         |
| Runtime       | CTranslate2 with int8 quantization             |
| Package       | `faster-whisper`                               |
| Execution     | Local (CPU)                                    |
| Input         | 16kHz mono WAV files                           |
| Output        | English transcript text                        |
| Language       | Hardcoded to `language="en"`                  |

**Why this model**: The base variant provides a balance between transcription
accuracy and inference speed suitable for near-real-time processing. CTranslate2
int8 quantization reduces memory footprint and speeds up inference on CPU
hardware without a significant accuracy loss. Larger models (small, medium, large)
would improve accuracy but increase latency beyond acceptable thresholds for
15-second audio buffers.

**How it runs**: Loaded once at service startup. Each buffer flush produces a WAV
file that is passed to `whisper_model.transcribe()`. The temp files are cleaned up
immediately after transcription.

### 6.2 Sentence-BERT all-MiniLM-L6-v2 (Local Semantic Similarity)

| Property      | Value                                          |
|---------------|------------------------------------------------|
| Model         | all-MiniLM-L6-v2 (22.7M parameters)           |
| Runtime       | PyTorch via sentence-transformers              |
| Package       | `sentence-transformers`                        |
| Execution     | Local (CPU)                                    |
| Input         | Text strings (cleaned transcript, topic+scope) |
| Output        | 384-dimensional embedding vectors              |
| Similarity    | Cosine similarity via `util.cos_sim()`         |

**Why this model**: all-MiniLM-L6-v2 is the standard choice for semantic textual
similarity tasks. It is small enough to run on CPU with sub-second inference,
produces high-quality 384-dim embeddings, and is pre-trained on over 1 billion
sentence pairs. It captures semantic meaning well enough to distinguish on-topic
from off-topic speech in the barter domain.

**How it runs**: Loaded once at service startup. The session topic+scope text is
embedded and cached when the contract is registered. Each conversation window's
cleaned text is embedded at classification time, and cosine similarity is computed
against the cached topic embedding.

### 6.3 NudeNet Detector (Local NSFW Detection)

| Property      | Value                                          |
|---------------|------------------------------------------------|
| Model         | NudeNet (YOLO-based object detection)          |
| Package       | `nudenet`                                      |
| Execution     | Local (CPU)                                    |
| Input         | JPEG image file path                           |
| Output        | List of detections with class and score        |
| Threshold     | Flag >= 0.6, hard block >= 0.9                 |

**Why this model**: NudeNet provides pre-trained NSFW detection that runs locally
without external API calls, ensuring privacy for video content. It detects
specific anatomical exposure classes with confidence scores, allowing tiered
response (warning vs hard block).

**Monitored classes**:
- `FEMALE_BREAST_EXPOSED`
- `FEMALE_GENITALIA_EXPOSED`
- `MALE_GENITALIA_EXPOSED`
- `BUTTOCKS_EXPOSED`
- `ANUS_EXPOSED`

**How it runs**: Loaded once at backend startup via `init_detector()`. Frames are
received as base64 strings, decoded to temp JPEG files, and passed to
`detector.detect()`. Results are filtered by class and confidence threshold.
Fail-open: if NudeNet fails to load or a check errors, the frame passes without
flagging.

### 6.4 Mistral Moderation API (Remote Toxicity Detection)

| Property      | Value                                          |
|---------------|------------------------------------------------|
| Model         | mistral-moderation-latest                      |
| API           | `https://api.mistral.ai/v1/moderations`        |
| Execution     | Remote (Mistral AI cloud)                      |
| Input         | Transcript text string                         |
| Output        | Flagged boolean + category scores              |
| Auth          | Bearer token (MISTRAL_API_KEY env var)         |
| Thresholds    | Flag >= 0.7, hard block >= 0.9                 |

**Why this model**: The Mistral Moderation API provides multi-category toxicity
detection (hate speech, violence, self-harm, sexual content, etc.) without
requiring a local model. It is called on every transcript segment, so it needs
to be fast and reliable.

**Fail-open behavior**: If the API call fails (network error, timeout, rate limit),
the system logs the error and does not block the transcript. This ensures the
session continues even if the toxicity API is temporarily unavailable.

### 6.5 Mistral Small (Remote Concept Suggestion)

| Property      | Value                                          |
|---------------|------------------------------------------------|
| Model         | mistral-small-latest                           |
| API           | `https://api.mistral.ai/v1/chat/completions`   |
| Execution     | Remote (Mistral AI cloud)                      |
| Input         | Session topic, scope, allowed concepts          |
| Output        | Comma-separated list of 6-8 disallowed concepts|
| Auth          | Bearer token (MISTRAL_API_KEY env var)         |

**Why this model**: Used only during session setup to suggest concepts that should
be disallowed. This is a non-critical convenience feature. The model receives a
structured prompt and returns a short comma-separated list.

---

## 7. Domain Concepts

### 7.1 Barter Session

A barter session represents a single knowledge-exchange interaction between two
participants. It has a defined lifecycle:

```
proposed --> active --> completed
                   \--> terminated
```

| Status       | Meaning                                                   |
|--------------|-----------------------------------------------------------|
| `proposed`   | Session created, waiting for start                        |
| `active`     | Session started, audio streaming and monitoring active    |
| `completed`  | Both users confirmed completion                           |
| `terminated` | Manually or automatically terminated (severe drift, etc.) |

### 7.2 Session Contract

The contract defines the rules for a session:

- **Topic**: The agreed subject matter (e.g., "Graphic design fundamentals").
- **Scope**: A broader description of the exchange context.
- **Allowed concepts**: Terms that are explicitly on-topic.
- **Disallowed concepts**: Terms that trigger immediate scope violation warnings.
- **Agreed duration**: The target session length in minutes.
- **Teacher/Learner IDs**: Which user teaches and which learns.

### 7.3 Windows

A "window" is a time-based chunk of conversation analyzed as a unit. Windows are
formed by accumulating teacher transcript segments until 30 seconds of speech
have been buffered. Each window receives a classification:

| Classification   | Cosine Similarity | Scope Violation | Meaning                         |
|------------------|-------------------|-----------------|---------------------------------|
| `correct`        | >= 0.55           | No              | On-topic speech                 |
| `weakly_correct` | 0.35 - 0.55       | No              | Marginally relevant speech      |
| `incorrect`      | < 0.35            | No              | Off-topic speech                |
| `out_of_scope`   | (any)             | Yes             | Disallowed concept detected     |

### 7.4 Semantic Thresholds

```
UPPER = 0.55    # cosine similarity >= UPPER --> "correct"
LOWER = 0.35    # LOWER <= similarity < UPPER --> "weakly_correct"
                 # similarity < LOWER --> "incorrect"
```

These thresholds are defined as constants in `semantic_analysis/main.py` and are
tunable. They were chosen empirically: 0.55 is high enough to require genuine
topical alignment, while 0.35 allows for tangential but related discussion.

### 7.5 Warning Escalation Rules

Warnings escalate based on consecutive off-topic windows:

| Consecutive Off-Topic Windows | Severity   | Behavior                              |
|-------------------------------|------------|---------------------------------------|
| 0                             | (none)     | Normal operation                      |
| 1                             | (silent)   | Logged internally, no user warning    |
| 2                             | `strong`   | Warning displayed to participants     |
| 3+                            | `severe`   | Warning displayed, auto-termination   |

**Scope violation override**: Any window with a scope violation (disallowed concept
keyword detected) immediately triggers a `strong` warning, regardless of the
consecutive counter.

**Safety override**: Toxicity or NSFW alerts bypass the escalation ladder entirely.
They issue `strong` (confidence >= 0.6) or `severe` (confidence >= 0.9) warnings
directly.

**Counter reset**: A single `correct` or `weakly_correct` window resets
`consecutive_incorrect` to 0, restarting the escalation ladder.

### 7.6 Trust Score Formula

Trust scores are updated after verdict generation:

```python
# QA score mapping
qa_scores = {"SUCCESSFUL": 1.0, "PARTIAL": 0.5, "DISPUTE": 0.0}
qa_score = qa_scores[verdict_type]

# Satisfaction rating (hardcoded to 4/5 = 0.8 for POC)
quality_adjusted = (qa_score + 0.8) / 2

# Trust update formula
new_trust = (previous_trust * 0.3) + (quality_adjusted * 0.7)
# Clamped to [0.0, 1.0]
```

**Interpretation**: The formula weights recent session performance (70%) more
heavily than historical trust (30%). A successful session with the default 0.8
satisfaction rating produces `quality_adjusted = (1.0 + 0.8) / 2 = 0.9`, which
pushes trust scores upward. A dispute produces `quality_adjusted = 0.4`, which
drives trust down.

**Starting trust**: Both seed users (Alice and Bob) begin with a trust score of
0.30.

### 7.7 Verdict Determination

| Condition                                    | Verdict Type  |
|----------------------------------------------|---------------|
| Session terminated (any reason)              | `DISPUTE`     |
| Duration pass AND confirmation pass          | `SUCCESSFUL`  |
| Duration pass OR confirmation pass (not both)| `PARTIAL`     |
| Neither duration nor confirmation passes      | `DISPUTE`     |

**Duration check**: passes if `actual_seconds >= agreed_duration_seconds * 0.8`
(session lasted at least 80% of the agreed time).

**Confirmation check**: passes if both users submitted a confirmation before the
session ended.

### 7.8 Engagement Scoring Formula

```python
# Combined engagement score (0.0 to 1.0)
engagement = (ratio_score * 0.5) + (question_score * 0.3) + (ack_score * 0.2)
```

| Component       | Weight | Ideal Value                     | Measurement                          |
|-----------------|--------|---------------------------------|--------------------------------------|
| Speaking ratio  | 50%    | Learner speaks 20-40% of time   | learner_seconds / total_seconds      |
| Questions       | 30%    | 1 question per 3 minutes        | question_count / minutes             |
| Acknowledgments | 20%    | Frequent acknowledgments        | ack_count / learner_segment_count    |

The score requires at least 30 seconds of total speaking time before producing
meaningful results. Below that threshold, it returns a neutral 0.5.

---

## 8. LAN Deployment

The system is designed to run on a local machine and be accessible to other devices
on the same LAN. This is necessary because WebRTC and `getUserMedia()` require a
secure context (HTTPS).

### HTTPS via Self-Signed Certificate

Vite's `@vitejs/plugin-basic-ssl` generates a self-signed TLS certificate at dev
server startup. The server binds to `0.0.0.0:5173`, making it accessible on all
network interfaces.

```javascript
// vite.config.js
plugins: [react(), basicSsl()],
server: {
    host: '0.0.0.0',
    port: 5173,
    https: true,
}
```

Devices on the LAN access the app at `https://<host-ip>:5173`. The browser will
warn about the self-signed certificate; the user must accept it to proceed.

### Vite Reverse Proxy

The Vite dev server acts as a reverse proxy, routing all API and WebSocket traffic
through port 5173. This eliminates CORS issues and means the browser only needs to
trust a single origin.

| Browser Path       | Proxied To                            |
|---------------------|---------------------------------------|
| `/session/*`        | `http://localhost:8000/session/*`      |
| `/verdict/*`        | `http://localhost:8000/verdict/*`      |
| `/trust/*`          | `http://localhost:8000/trust/*`        |
| `/safety/*`         | `http://localhost:8000/safety/*`       |
| `/window/*`         | `http://localhost:8000/window/*`       |
| `/warnings/*`       | `http://localhost:8000/warnings/*`     |
| `/ws/*`             | `ws://localhost:8000/ws/*`             |
| `/audio/*`          | `ws://localhost:8001/audio/*`          |

### WebRTC ICE with STUN

The system uses Google's public STUN servers to discover the host's externally
reachable IP address for WebRTC ICE candidate gathering:

```javascript
iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
]
```

On a LAN, STUN typically resolves the local IP address, allowing direct peer
connectivity. No TURN server is configured, so the system does not support
connections across different NATs or firewalls.

### Startup Script (start.sh)

The `start.sh` script automates the complete setup and launch:

1. **System dependencies**: Checks for (and installs via Homebrew if missing)
   Python 3.11+, Node.js, FFmpeg, PostgreSQL 16.
2. **PostgreSQL**: Starts the service, creates the `barter` role and `barter_db`
   database, runs all migration SQL files.
3. **Python virtual environments**: Creates venvs and installs `requirements.txt`
   for each of the four services.
4. **Frontend**: Runs `npm install` if `node_modules` is absent.
5. **Service launch**: Starts all four uvicorn services (bound to `0.0.0.0`) and
   the Vite dev server. All output is prefixed with service labels.
6. **Health checks**: Polls `/health` on ports 8000-8003.
7. **Cleanup**: Traps `EXIT`, `INT`, and `TERM` signals to kill all child
   processes on shutdown.

```bash
# Launch command for each service
uvicorn "$entry" --host 0.0.0.0 --port "$port" --log-level warning
```

---

## 9. Database Schema

PostgreSQL 16, database name `barter_db`, role `barter`.

### Entity Relationship Diagram

```
users
  |--- 1:N ---> barter_sessions (as user1_id or user2_id)
  |--- 1:N ---> session_contracts (as teacher_user_id or learner_user_id)
  |--- 1:N ---> confirmations (as user_id)
  |--- 1:N ---> transcript_segments (as user_id)

barter_sessions
  |--- 1:1 ---> session_contracts
  |--- 1:N ---> window_results
  |--- 1:N ---> warnings
  |--- 1:1 ---> verdicts
  |--- 1:N ---> confirmations
  |--- 1:N ---> transcript_segments
```

### Table Definitions

#### users

| Column       | Type           | Constraints                          |
|--------------|----------------|--------------------------------------|
| id           | SERIAL         | PRIMARY KEY                          |
| username     | VARCHAR(50)    | UNIQUE, NOT NULL                     |
| trust_score  | FLOAT          | NOT NULL, DEFAULT 0.30               |
| created_at   | TIMESTAMPTZ    | NOT NULL, DEFAULT NOW()              |

Seed data: `alice` (trust 0.30), `bob` (trust 0.30).

#### barter_sessions

| Column       | Type           | Constraints                          |
|--------------|----------------|--------------------------------------|
| id           | SERIAL         | PRIMARY KEY                          |
| user1_id     | INTEGER        | NOT NULL, FK -> users(id)            |
| user2_id     | INTEGER        | NOT NULL, FK -> users(id)            |
| status       | VARCHAR(20)    | NOT NULL, DEFAULT 'pending'          |
| started_at   | TIMESTAMPTZ    | NULLABLE                             |
| ended_at     | TIMESTAMPTZ    | NULLABLE                             |
| created_at   | TIMESTAMPTZ    | NOT NULL, DEFAULT NOW()              |

Status values: `proposed`, `active`, `completed`, `terminated`.

#### session_contracts

| Column                  | Type        | Constraints                      |
|-------------------------|-------------|----------------------------------|
| id                      | SERIAL      | PRIMARY KEY                      |
| barter_session_id       | INTEGER     | NOT NULL, FK -> barter_sessions  |
| topic                   | VARCHAR(200)| NOT NULL                         |
| scope                   | TEXT        | NULLABLE                         |
| allowed_concepts        | TEXT        | NULLABLE (comma-separated)       |
| disallowed_concepts     | TEXT        | NULLABLE (comma-separated)       |
| agreed_duration_seconds | INTEGER     | NOT NULL                         |
| teacher_user_id         | INTEGER     | NOT NULL, FK -> users(id)        |
| learner_user_id         | INTEGER     | NOT NULL, FK -> users(id)        |

#### transcript_segments

| Column             | Type        | Constraints                         |
|--------------------|-------------|-------------------------------------|
| id                 | SERIAL      | PRIMARY KEY                         |
| barter_session_id  | INTEGER     | NOT NULL, FK -> barter_sessions     |
| user_id            | INTEGER     | NOT NULL, FK -> users(id)           |
| text               | TEXT        | NOT NULL                            |
| duration_seconds   | FLOAT       | NOT NULL                            |
| timestamp_start    | FLOAT       | DEFAULT 0.0                         |
| timestamp_end      | FLOAT       | DEFAULT 0.0                         |
| created_at         | TIMESTAMPTZ | DEFAULT NOW()                       |

#### window_results

| Column              | Type        | Constraints                        |
|---------------------|-------------|------------------------------------|
| id                  | SERIAL      | PRIMARY KEY                        |
| barter_session_id   | INTEGER     | NOT NULL, FK -> barter_sessions    |
| window_number       | INTEGER     | NOT NULL                           |
| classification      | VARCHAR(20) | NOT NULL                           |
| cosine_similarity   | FLOAT       | NOT NULL                           |
| text_content        | TEXT        | NULLABLE                           |
| created_at          | TIMESTAMPTZ | NOT NULL, DEFAULT NOW()            |

Classification values: `correct`, `weakly_correct`, `incorrect`, `out_of_scope`.

#### warnings

| Column             | Type        | Constraints                         |
|--------------------|-------------|-------------------------------------|
| id                 | SERIAL      | PRIMARY KEY                         |
| barter_session_id  | INTEGER     | NOT NULL, FK -> barter_sessions     |
| severity           | VARCHAR(20) | NOT NULL                            |
| message            | TEXT        | NOT NULL                            |
| window_ids         | TEXT        | NULLABLE                            |
| created_at         | TIMESTAMPTZ | NOT NULL, DEFAULT NOW()             |

Severity values: `mild`, `strong`, `severe`.

#### verdicts

| Column              | Type        | Constraints                         |
|---------------------|-------------|-------------------------------------|
| id                  | SERIAL      | PRIMARY KEY                         |
| barter_session_id   | INTEGER     | NOT NULL, UNIQUE, FK -> barter_sessions |
| verdict_type        | VARCHAR(20) | NOT NULL                            |
| on_topic_percentage | FLOAT       | NOT NULL                            |
| warning_count       | INTEGER     | NOT NULL                            |
| duration_check      | VARCHAR(10) | NOT NULL                            |
| confirmation_check  | VARCHAR(10) | NOT NULL                            |
| trust_delta_user1   | FLOAT       | NOT NULL                            |
| trust_delta_user2   | FLOAT       | NOT NULL                            |
| drift_summary       | TEXT        | NULLABLE (JSON string)              |
| created_at          | TIMESTAMPTZ | NOT NULL, DEFAULT NOW()             |

Verdict types: `SUCCESSFUL`, `PARTIAL`, `DISPUTE`, `PENDING`.

The `drift_summary` column stores a JSON string containing the full drift analysis
from the Warning Engine and the engagement summary from Semantic Analysis.

#### confirmations

| Column             | Type        | Constraints                         |
|--------------------|-------------|-------------------------------------|
| id                 | SERIAL      | PRIMARY KEY                         |
| barter_session_id  | INTEGER     | NOT NULL, FK -> barter_sessions     |
| user_id            | INTEGER     | NOT NULL, FK -> users(id)           |
| confirmed_at       | TIMESTAMPTZ | NOT NULL, DEFAULT NOW()             |

Unique constraint on `(barter_session_id, user_id)` prevents double-confirmation.

### Migration History

| File                       | Purpose                                           |
|----------------------------|---------------------------------------------------|
| `001_create_tables.sql`    | Core tables: users, sessions, contracts, windows, warnings, verdicts |
| `002_seed_data.sql`        | Seed Alice and Bob with trust_score 0.30          |
| `003_stage2_updates.sql`   | Add confirmations table, drift_summary to verdicts, window_ids to warnings |
| `004_teacher_learner.sql`  | Add teacher/learner role columns to session_contracts |
| `005_transcript_segments.sql` | Add transcript_segments table                   |

---

## 10. Security Features

### 10.1 Toxicity Detection

Every transcript segment produced by the audio pipeline is checked against the
Mistral Moderation API before being forwarded for topic analysis.

**Detection thresholds**:
- Category score >= 0.7: flagged as toxic, `strong` warning issued.
- Category score >= 0.9: flagged as hard block, `severe` warning issued.

**Fail-open design**: If the Mistral API is unavailable, the transcript is
processed normally. This prevents API outages from blocking sessions.

**Coverage**: All text from both teacher and learner is checked. Toxicity and
topic analysis run in parallel (toxicity does not block or replace semantic
analysis).

### 10.2 NSFW Video Frame Detection

The frontend captures a frame from the local video element every 10 seconds and
sends it to the backend for NudeNet analysis.

**Detection pipeline**:
1. Canvas API captures the video frame.
2. JPEG export at 50% quality reduces payload size.
3. Base64-encoded image sent to `POST /safety/check-frame`.
4. NudeNet runs object detection on the decoded JPEG.
5. Detections filtered by class (5 NSFW classes) and confidence threshold.

**Confidence thresholds**:
- Score >= 0.6 (`NSFW_CONFIDENCE_THRESHOLD`): detection flagged.
- Score >= 0.9 (`NSFW_HARD_BLOCK_THRESHOLD`): hard block, `severe` warning.

**Fail-open design**: If NudeNet fails to initialize at startup, NSFW checks are
silently disabled. If an individual check errors, the frame passes unflagged.

**Frame rate**: 0.1 FPS (one frame every 10 seconds). This balances detection
coverage against server load.

### 10.3 Scope Violation Detection

Disallowed concepts are checked via case-insensitive substring matching against
the raw transcript text of each window:

```python
def check_scope_violation(text: str, disallowed_concepts: list[str]) -> bool:
    text_lower = text.lower()
    for concept in disallowed_concepts:
        if concept.lower() in text_lower:
            return True
    return False
```

A scope violation immediately triggers a `strong` warning regardless of the
cosine similarity score, and the window is classified as `out_of_scope`.

### 10.4 Escalation as Safety Net

The warning escalation system itself acts as a safety mechanism:

- **Silent monitoring** (1 off-topic window): no disruption, but the system is
  watching.
- **Active warning** (2+ consecutive off-topic windows): participants are alerted.
- **Auto-termination** (severe warnings in the UI): the frontend stops recording
  and closes connections when a severe warning is received.

### 10.5 Configuration Security

Sensitive configuration (database URL, Mistral API key) is loaded from environment
variables via `.env` files and `pydantic-settings`:

```python
class Settings(BaseSettings):
    DATABASE_URL: str
    DATABASE_URL_SYNC: str
    MISTRAL_API_KEY: str = ""
    model_config = {"env_file": ".env"}
```

CORS is configured with `allow_origins=["*"]` across all services. This is
appropriate for the LAN development deployment but should be restricted for
production use.
