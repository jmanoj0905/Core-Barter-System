# Development

How to build, run, configure and test the system locally.

## Running with Docker Compose (the supported path)

From the repo root:

```bash
docker compose up --build          # build and start everything
docker compose up --build backend  # one service
docker compose logs -f audio_pipeline
docker compose down
```

The frontend container terminates SSL and proxies everything else, so the only ports published by
`docker-compose.yml` are `80` (redirects) and `443` (SPA + proxies). Open `https://localhost/`.

`docker-compose.override.yml` additionally publishes `8000`–`8003` straight to the host. It is
local-only, for latency benchmarking (`latency_bench/`) — Compose picks it up automatically, so
delete or rename it if you don't want those ports exposed.

`get-urls.sh` prints the LAN URLs to hand to the two participants on other machines.

### Build speed

Docker BuildKit is on by default in current Docker releases and gives parallel stage builds plus
pip cache mounts. Useful variants:

```bash
docker compose build --parallel    # build service images concurrently
docker compose build --no-cache    # force a clean rebuild
docker compose up -d --build       # rebuild and start detached
```

If you are on an older Docker that still defaults to the legacy builder, export
`DOCKER_BUILDKIT=1` and `COMPOSE_DOCKER_CLI_BUILD=1` first.

## Services and ports

| Service | Port | Entry point | In Compose? |
|---|---|---|---|
| `backend` | 8000 | `app.main:app` | yes |
| `audio_pipeline` | 8001 | `main:app` | yes |
| `semantic_analysis` | 8002 | `main:app` | yes |
| `warning_engine` | 8003 | `main:app` | yes |
| `video_engagement` | 8004 | `main:app` | yes |
| `frontend` | 80 / 443 | nginx | yes |
| `resource_agent` | — | — | no — scaffold only, not wired up |

Each service under `apps/` is self-contained: its own `requirements.txt` and `Dockerfile`.
`pyproject.toml` at the root is repo-wide metadata and tool config (ruff, pytest) only — there is
no uv workspace and dependencies are not resolved across services.

## Configuration

Set these in a root `.env` file or in the `environment:` block of `docker-compose.yml`.

| Variable | Default | Used by |
|---|---|---|
| `STT_BACKEND` | `aws` | `audio_pipeline` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | — | `audio_pipeline`, `video_engagement` |
| `AWS_REGION` | `ap-south-1` | `audio_pipeline`, `video_engagement` |
| `AWS_S3_BUCKET` | `core-barter-audio-tmp` | `audio_pipeline` |
| `DEEPGRAM_API_KEY` | — | `audio_pipeline` (alternate STT backend) |
| `MISTRAL_API_KEY` | — | `backend`, `audio_pipeline` |
| `VIDEO_BACKEND` | `local` | `video_engagement` |
| `VIDEO_WEIGHT_EYES` / `_HEAD` / `_GAZE` | `0.55` / `0.4` / `0.05` | `video_engagement` |
| `ENGAGEMENT_FUSION_W_SPEECH` / `_W_VIDEO` | `0.7` / `0.3` | `warning_engine` |
| `DATABASE_URL` | `sqlite+aiosqlite:////data/barter.db` | `backend` |

The weight defaults are fitted values, not guesses — see
[video_engagement/design-choices.md](video_engagement/design-choices.md) and
[threshold-calibration.md](threshold-calibration.md).

AWS credential and IAM setup for the eval-dataset tooling is covered separately in
[eval_dataset/aws-setup.md](eval_dataset/aws-setup.md).

## Tests

```bash
pytest tests                          # end-to-end suites
pytest apps/backend/tests             # per-service suites
pytest apps/semantic_analysis/tests
pytest apps/video_engagement/tests
pytest eval_dataset/tools/tests
```

`tests/` holds the cross-service end-to-end suites (session lifecycle, verdict and trust, warning
engine, finalization regressions). Each service keeps its own unit tests under `apps/<service>/tests/`.

## Linting

```bash
ruff check .        # line-length 100, configured in pyproject.toml
```

## Known stale tooling

- `start.sh` is a non-Docker launcher that still provisions **PostgreSQL** and runs
  `apps/backend/migrations/*.sql` against it. The system runs on SQLite; this script has not been
  updated and will not produce a working stack. Use Docker Compose.
- `.env.example` likewise still lists PostgreSQL `DATABASE_URL`s and `OPENAI_API_KEY`, none of
  which match what the services actually read. The table above is the accurate list.
