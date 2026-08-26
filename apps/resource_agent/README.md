# Resource Agent

Owns every credit in the system: accounts, escrow, and the double-entry ledger.

Design: `../../docs/superpowers/specs/2026-08-26-resource-escrow-design.md`

## Model

Balances are derived from `ledger_lines`. The `accounts.balance` column is a
cache that the reconciler verifies every 5 minutes. Every journal entry must sum
to zero, and every credit amount is an integer.

Locked balance is a separate account (`user_locked`), not a column — so locking
escrow is a transfer, which is auditable and cannot be recorded lopsidedly.

## Policy

| Mechanism | Rule |
|---|---|
| Escrow sizing | `clamp(round(40 * (1 - trust)), 5, 40)` |
| Regeneration | 5/10/20 credits per day by trust band, capped at 100, applied lazily on read |
| Participation floor | Top up to 5 credits when stranded — 24h cooldown, blocked during an active reservation |
| Settlement | `SUCCESSFUL` + qa ≥ 0.85 → full release; qa ≥ 0.5 → proportional; otherwise penalty |

Penalised stakes go to the counterparty, not the platform.

All constants are environment-overridable. See `app/policy.py`.

## Idempotency

Every mutating operation derives a key (`reserve:{session_id}`,
`settle:{session_id}`, …) written to a unique column. A retry hits the
constraint and returns the original result — no double payout, and no separate
idempotency table.

## Running the tests

Postgres must be up:

```bash
docker compose up -d resource_db
```

The host `python3` is not guaranteed to match this service's supported Python
version (3.11) — running the suite under whatever `python3` resolves to
system-wide is unreliable. Use a matching virtualenv (this repo keeps one at
`.venv-resource/` at the project root, gitignored):

```bash
cd apps/resource_agent && /path/to/.venv-resource/bin/python -m pytest tests/ -v
```

Tests reach Postgres on `localhost:5433`. Override with `DATABASE_URL`.
