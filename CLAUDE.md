# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Core Barter System — a real-time audio conversation monitoring platform that enforces topic adherence during barter/negotiation sessions. Users have time-limited audio conversations on an agreed topic; the system transcribes audio, analyzes semantic relevance, issues escalating warnings for off-topic drift, and produces post-session verdicts with trust score updates.

## Architecture

Monorepo with 5 Docker services + React frontend (nginx):

```
apps/
├── backend/           # Port 8000 - FastAPI + SQLite
├── audio_pipeline/    # Port 8001 - AWS Transcribe STT
├── semantic_analysis/  # Port 8002 - Sentence-BERT
├── warning_engine/    # Port 8003 - Escalation logic
└── frontend/          # nginx (ports 80→443 redirect, 443 for SPA+proxies)
```

## Tech Stack

- **Backend**: Python, FastAPI, SQLite
- **Frontend**: React, nginx (SSL termination + WebSocket proxies)
- **ML/Audio**: Sentence-BERT (sentence-transformers), AWS Transcribe, FFmpeg
- **Deployment**: Docker Compose

## Key Domain Concepts

- **Barter Session**: lifecycle `create → start → confirm (both users) → verdict → trust update`
- **Window**: ~5 seconds of accumulated audio, classified as correct / weakly_correct / incorrect
- **Warning escalation**: 1 consecutive off-topic → silent, 2 → strong, 3+ → severe
- **Trust scores**: updated post-session based on verdict
- **Escrow**: credits locked on session start, released based on QA score

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

## Frontend Routing (nginx.conf)

Frontend nginx proxies:

| Path | → Service |
|------|----------|
| `/audio/` | audio_pipeline:8001 (WebSocket) |
| `/ws/` | backend:8000 (WebSocket) |
| `/stt/` | audio_pipeline:8001 |
| `/session/*`, `/trust/*`, `/wallet/*`, etc. | backend:8000 |
| `/` | React SPA |

Frontend connects via `location.host` — WebSocket goes to nginx which proxies to audio_pipeline.

## STT Configuration

Audio pipeline uses AWS Transcribe by default (`STT_BACKEND=aws`). Env vars:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (default: ap-south-1)
- `AWS_S3_BUCKET` (default: core-barter-audio-tmp)

Set via `.env` file or docker-compose.yml environment section.

## Service Layout

Each service under `apps/` is self-contained with `main.py`, `requirements.txt`, and `Dockerfile`.
- `backend/` entry point: `app.main:app`
- Other services: `main:app`