# Archive

Historical documents. They record decisions and plans as they stood at the time and are kept for
that context, but they are **not** a description of the system as it is today — each has been
overtaken by the implementation in at least one respect.

| Document | What it is | Superseded by |
|---|---|---|
| [design-decisions-2026-03.md](design-decisions-2026-03.md) | The original 15-section decision log (scope, STT choice, thresholds, escrow, demo plan) | [architecture.md](../architecture.md), [threshold-calibration.md](../threshold-calibration.md) |
| [implementation-plan-2026-03.md](implementation-plan-2026-03.md) | File-level plan mapping those decisions to code, phases A–J | [issues.md](../issues.md), [superpowers/plans/](../superpowers/plans/) |
| [build-roadmap.md](build-roadmap.md) | The 8-stage build order used to bring the POC up from nothing | [architecture.md](../architecture.md) |
| [technical-details-2026-08.md](technical-details-2026-08.md) | The long-form internals reference: per-service internals, data flows, ML models, DB schema, deployment, security | [architecture.md](../architecture.md), [development.md](../development.md) — partially, see below |

Known ways these disagree with the code: they describe PostgreSQL rather than SQLite,
`faster-whisper` rather than AWS Transcribe, four services rather than five (no
`video_engagement`), deployment via the Vite dev server rather than the nginx container, and
semantic thresholds of `0.55` / `0.35` rather than the calibrated `0.36` / `0.14`.

`technical-details-2026-08.md` is the one worth reading anyway: nothing has replaced its
depth on per-service internals, the key data flows, and the trust/QA formulas. Its per-file
banner lists exactly which sections have gone stale. Rewriting it against the current code is
outstanding work.
