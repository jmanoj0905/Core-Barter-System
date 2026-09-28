# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Core Barter System — a real-time audio and video conversation monitoring platform that enforces topic adherence during barter/negotiation sessions. Users have time-limited conversations on an agreed topic; the system transcribes audio, analyzes semantic relevance, scores webcam attention, issues escalating warnings for off-topic drift, and produces post-session verdicts with trust score updates.

Documentation lives under `docs/` — [`docs/README.md`](docs/README.md) is the index. `docs/archive/` holds superseded documents; do not treat them as current.

## Architecture

Monorepo with 5 FastAPI services + React frontend (nginx), all in Docker Compose:

```
apps/
├── backend/            # Port 8000 - FastAPI + SQLite, session lifecycle + escrow
├── audio_pipeline/     # Port 8001 - AWS Transcribe STT
├── semantic_analysis/  # Port 8002 - Sentence-BERT topic relevance
├── warning_engine/     # Port 8003 - Escalation + speech/video engagement fusion
├── video_engagement/   # Port 8004 - Webcam attention scoring
├── resource_agent/     # Scaffold only — not in docker-compose.yml, no implementation yet
└── frontend/           # nginx (ports 80→443 redirect, 443 for SPA+proxies)
```

The services are a pipeline, not hub-and-spoke: the browser streams media straight to
`audio_pipeline` and `video_engagement`; `audio_pipeline → semantic_analysis → warning_engine`;
`video_engagement → warning_engine`; and those services POST evidence back to `backend`, which
owns the database and the warning fan-out to the browser.

## Tech Stack

- **Backend**: Python, FastAPI, SQLite
- **Frontend**: React, nginx (SSL termination + WebSocket proxies)
- **ML/Audio**: Sentence-BERT (sentence-transformers), AWS Transcribe, FFmpeg
- **ML/Video**: face-landmark sub-signals (eyes / head pose / gaze)
- **Deployment**: Docker Compose

## Key Domain Concepts

- **Barter Session**: lifecycle `create → start → confirm (both users) → verdict → trust update`
- **Segment**: 5 seconds of buffered audio (`BUFFER_THRESHOLD_SECONDS`), transcribed as a unit
- **Window**: ~25 seconds of accumulated transcript (`WINDOW_DURATION_THRESHOLD`), classified as
  correct / weakly_correct / incorrect against cosine thresholds `UPPER = 0.36` / `LOWER = 0.14`
  (or the relative `thr = RHO · R`) — see `docs/threshold-calibration.md`
- **Warning escalation**: 1 consecutive off-topic → silent, 2 → strong, 3+ → severe
- **Trust scores**: updated post-session based on verdict
- **Escrow**: credits locked on session start, released based on QA score
- **Engagement fusion**: speech engagement (from `semantic_analysis`) blended with video attention (from `video_engagement`) in the warning engine, default 0.7 / 0.3

## Database

SQLite with 9 tables: Users, Barter Sessions, Session Contracts, Window Results, Warnings Log, Verdicts, Confirmations, Wallets, Escrows, Credit Transactions.

## Running the Services

All services run via Docker Compose. From project root:

```bash
# Build and start all services
docker compose up --build

# Start specific service
docker compose up --build audio_pipeline

# View logs
docker compose logs -f audio_pipeline
```

## Service Communication

Docker Compose DNS resolves service names:
- `backend` → port 8000
- `audio_pipeline` → port 8001
- `semantic_analysis` → port 8002
- `warning_engine` → port 8003
- `video_engagement` → port 8004

## Frontend Routing (nginx.conf)

Frontend nginx proxies:

| Path | → Service |
|------|----------|
| `/audio/` | audio_pipeline:8001 (WebSocket) |
| `/video/` | video_engagement:8004 (WebSocket) |
| `/ws/` | backend:8000 (WebSocket) |
| `/stt/` | audio_pipeline:8001 |
| `/session/*`, `/verdict/*`, `/trust/*`, `/wallet/*`, `/escrow/*`, `/warnings/*`, `/safety/*`, etc. | backend:8000 |
| `/` | React SPA |

Frontend connects via `location.host` — WebSockets go to nginx, which proxies them on.

## STT Configuration

Audio pipeline uses AWS Transcribe by default (`STT_BACKEND=aws`). Env vars:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (default: ap-south-1)
- `AWS_S3_BUCKET` (default: core-barter-audio-tmp)

Set via `.env` file or docker-compose.yml environment section. The full env-var table is in
`docs/development.md`.

Note: `.env.example` and `start.sh` are stale — both still assume PostgreSQL and `OPENAI_API_KEY`,
neither of which the services use. Use Docker Compose, not `start.sh`.

## Service Layout

Each service under `apps/` is self-contained with `main.py`, `requirements.txt`, and `Dockerfile`.
- `backend/` entry point: `app.main:app`
- Other services: `main:app`

Root `pyproject.toml` is repo-wide metadata and tool config (ruff line-length 100, pytest) only —
there is no uv workspace and dependencies are not shared across services.

Cross-service end-to-end tests are in `tests/`; per-service unit tests in `apps/<service>/tests/`.