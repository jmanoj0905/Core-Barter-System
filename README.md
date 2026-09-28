# Core Barter System

A real-time audio and video conversation monitoring platform that enforces topic adherence during
barter sessions. Two users agree a topic and a time limit; the system transcribes their speech,
measures how close each window of conversation stays to the agreed topic, scores webcam attention,
issues escalating warnings when either drifts, and at the end produces a verdict that settles the
escrowed credits and updates both trust scores.

Built as a capstone prototype. It is a working two-person demo, not a production system — see
[docs/issues.md](docs/issues.md) for the tracked gaps.

## How it works

The browser streams audio and video through nginx to the capture services. Each stage calls the
next directly, and evidence lands in the backend's SQLite database:

```
browser ──audio──> audio_pipeline ──segments──> semantic_analysis ──windows──┐
        │                     └──transcripts──> backend                      ├─> warning_engine ─> backend ─> browser
        └──video──> video_engagement ──attention scores─────────────────────┘                         (warnings over WS)
```

- **Audio** is buffered into 5-second segments and transcribed (AWS Transcribe by default).
- **Semantic analysis** accumulates segments into ~25-second windows, embeds each with
  Sentence-BERT, and classifies it `correct` / `weakly_correct` / `incorrect` against the topic.
- **Video engagement** scores webcam attention per window from eye, head-pose and gaze sub-signals.
- **The warning engine** fuses speech and video engagement and escalates on consecutive off-topic
  windows: 1 → silent, 2 → strong, 3+ → severe.
- **The backend** owns the session lifecycle (`create → start → confirm → verdict → trust update`),
  the escrow, and the WebSocket fan-out to both participants.

The thresholds and weights above are fitted, not picked by hand —
[docs/threshold-calibration.md](docs/threshold-calibration.md) lists each one and the experiment
behind it.

## Services

| Service | Port | Stack |
|---|---|---|
| `backend` | 8000 | FastAPI, SQLAlchemy, SQLite (aiosqlite) |
| `audio_pipeline` | 8001 | FastAPI, FFmpeg, AWS Transcribe |
| `semantic_analysis` | 8002 | FastAPI, Sentence-BERT (`all-MiniLM-L6-v2`) |
| `warning_engine` | 8003 | FastAPI |
| `video_engagement` | 8004 | FastAPI, face-landmark sub-signals |
| `frontend` | 80 / 443 | React + Vite, served by nginx (SSL + WebSocket proxies) |

## Running it

```bash
docker compose up --build
```

Then open `https://localhost/`. Full setup, configuration and test instructions are in
[docs/development.md](docs/development.md).

## Repository layout

```
apps/            the five services plus the React frontend
docs/            all documentation — start at docs/README.md
eval_dataset/    evaluation corpus, curation tools and topic pool
latency_bench/   latency benchmarking against the override compose file
tests/           cross-service end-to-end suites
```

## Documentation

[docs/README.md](docs/README.md) is the index. The most-used entries:

- [docs/architecture.md](docs/architecture.md) — services, data flow, lifecycle diagrams
- [docs/development.md](docs/development.md) — build, run, configure, test
- [docs/issues.md](docs/issues.md) — known bugs and gaps, with a suggested repair order
