# Resource Agent & Escrow Subsystem — Design

**Date:** 2026-08-26
**Status:** Approved design, not yet implemented
**Supersedes:** `DESIGN_DECISIONS.md` §12 ("Credits & Escrow (Minimal)")

---

## 1. Purpose

Build the Resource Agent described in the capstone architecture notes as a real
subsystem: a service that owns credits, sizes and locks escrow before a session,
settles it after the verdict, and keeps an auditable ledger of every movement.

Today `apps/backend/app/escrow.py` sketches this — trust-scaled escrow, lock and
release, three tables — but `get_or_create_wallet` seeds every wallet with
999,999 credits. Nothing is ever scarce, so nothing the escrow logic computes has
consequences. This design makes the economy real.

It also resolves the credit death spiral identified in
`Notes/Capstone/Credit system solutions.md`. That document lists Solutions A–G in
its table of contents but its text ends partway through Solution B; C through G
were never written. The policy below is the decision that document deferred.

### In scope

- `resource_agent` service owning all credit state
- Double-entry ledger with idempotent, auditable entries
- Asymmetric escrow sizing, reservation, and settlement
- Credit regeneration and a participation floor
- Dispute holds and admin resolution
- Reconciliation of stale reservations and balance drift
- Frontend wallet, escrow, settlement, and ledger views

### Out of scope

- Skill matching and identity verification (separate repo)
- Fraud detection scoring. `settle` accepts a `qa_score`; computing it belongs to
  the QA and fraud agents.
- Real money, payments, multi-currency
- Migration of existing `barter.db` credit rows (fresh start, see §8)

---

## 2. Decisions

| # | Decision | Alternatives rejected |
|---|---|---|
| D1 | Resource Agent is its own service, `apps/resource_agent`, port 8004 | Expanding `backend/app/escrow.py` into a package; a module behind an extractable interface |
| D2 | `resource_agent` owns wallet/escrow/ledger tables in its own **PostgreSQL** database | Shared SQLite file with WAL; owned tables still on SQLite |
| D3 | Hybrid credit policy: asymmetric escrow + regeneration + participation floor | Asymmetric escrow alone; reputation-only with behavioral penalties |
| D4 | Consistency via idempotent single-call operations plus a background reconciler | Best-effort with compensating release; transactional outbox with async settlement |
| D5 | Fresh start — no migration of existing synthetic wallet data | Migrate and rewrite balances; feature-flagged enforcement |
| D6 | Frontend gets full wallet, escrow, settlement, and ledger views | API-only; balance and settlement summary only |

### D2 rationale

Escrow reservation requires locking two accounts and writing a multi-row ledger
entry atomically, under concurrency. `SELECT ... FOR UPDATE` on both accounts is
the natural expression of that. SQLite offers a single writer for the whole
database, so the same guarantee costs full serialization of the application.
Postgres is the correct tool, and the requirement is genuine rather than
decorative.

### D3 rationale

Each mechanism handles a distinct failure mode, and each is independently
testable:

- Asymmetric escrow makes new and untrusted users stake more, so risk tracks
  reputation.
- Regeneration guarantees recovery from a bad run without manual intervention.
- The floor guarantees a user can always afford at least a minimum session, which
  is what actually kills the death spiral.

---

## 3. Architecture

```
┌───────────┐   reserve / settle / void   ┌────────────────┐      ┌─────────────┐
│  backend  │ ──────────────────────────> │ resource_agent │ ───> │ resource_db │
│   :8000   │ <────── balances, 409 ───── │     :8004      │      │  Postgres   │
└───────────┘                             └────────────────┘      └─────────────┘
      ▲                                          ▲
      │ /session/*, /ws/*                        │ /resource/*
      └──────────────── nginx (frontend) ────────┘
```

`resource_agent` is a FastAPI service following the existing per-service layout:
`main.py`, `requirements.txt`, `Dockerfile`. A new compose service `resource_db`
runs Postgres 16 with its own volume.

**Backend loses** the `Wallet`, `Escrow`, and `CreditTransaction` models and
`app/escrow.py`. It **gains** `app/clients/resource.py`, a thin `httpx` client
with one method per operation, retries, and idempotency-key handling.
`RESOURCE_URL=http://resource_agent:8004`.

Backend retains `trust_score` — that is reputation, not money — and passes it in
on reserve. `resource_agent` never calls back into backend. The dependency is
one-directional and acyclic.

nginx proxies `/resource/` to `resource_agent:8004` so the frontend reads wallet
and ledger data directly rather than tunnelling through backend.

### Integration surface

Exactly two touchpoints:

1. `POST /session/{id}/start` calls `reserve(session_id, [both participants])`.
   Insufficient credits returns 409, and the session does not start.
2. The confirm → verdict path calls `settle(session_id, verdict, qa_score)`.

Regeneration, floor top-ups, dispute resolution, and reconciliation are internal
to `resource_agent`. Backend never triggers them.

---

## 4. Data model

Double-entry. Balances are derived from the ledger; the cached column is an
optimization that the reconciler verifies. All amounts are **integer credits** —
no floating point anywhere in the money path.

### `accounts`

Every balance-bearing entity, user or system.

| Column | Notes |
|---|---|
| `id` | PK |
| `kind` | `user_available` \| `user_locked` \| `platform_mint` \| `platform_revenue` |
| `user_id` | null for system accounts |
| `balance` | cached, integer |
| `floor` | participation floor |
| `regen_rate` | credits/day |
| `last_regen_at` | drives lazy regeneration |
| `last_topup_at` | drives floor cooldown |
| `created_at` | |

Unique on `(kind, user_id)`.

Each user has two accounts: `user_available` and `user_locked`. Locked balance is
an account rather than a column, so a lock is a *transfer* — auditable, balanced,
and impossible to express as a lopsided mutation.

### `journal_entries`

One row per business event.

| Column | Notes |
|---|---|
| `id` | PK |
| `idempotency_key` | **UNIQUE** |
| `entry_type` | `grant` \| `escrow_reserve` \| `escrow_settle` \| `escrow_void` \| `regen` \| `floor_topup` \| `fee` \| `penalty` \| `bonus` \| `dispute_hold` \| `dispute_resolve` |
| `session_id` | nullable |
| `payload` | JSONB — the policy inputs that produced this entry |
| `created_at` | |

### `ledger_lines`

| Column | Notes |
|---|---|
| `entry_id` | FK |
| `account_id` | FK |
| `amount` | signed integer |

**Invariant: `SUM(amount) = 0` for every `entry_id`.** Enforced in code on write
and verified by the reconciler.

`accounts.balance` is updated in the same transaction as the lines.

### `escrows`

| Column | Notes |
|---|---|
| `session_id`, `user_id` | unique together |
| `amount` | integer |
| `state` | `RESERVED` \| `SETTLED` \| `VOIDED` \| `HELD` |
| `reserve_entry_id`, `settle_entry_id` | FK to `journal_entries` |
| `reserved_at`, `settled_at` | |

### `disputes`

`session_id`, `reason`, `state` (`OPEN` \| `RESOLVED`), `resolution`,
`resolved_by`, `resolved_at`.

### Idempotency

Idempotency lives entirely in the unique index on
`journal_entries.idempotency_key`. Keys are `reserve:{session_id}`,
`settle:{session_id}`, `void:{session_id}`. A retried call violates the
constraint; the handler catches it and returns the original entry's result. No
second payout, and no separate idempotency table to keep in sync.

### Escrow state machine

```
        reserve
  (none) ───────> RESERVED ──settle──> SETTLED
                     │  │
              void   │  └──hold──> HELD ──resolve──> SETTLED | VOIDED
                     ▼
                  VOIDED
```

`SETTLED` and `VOIDED` are terminal. Every edge writes a journal entry.

---

## 5. Policy engine

`apps/resource_agent/app/policy.py`. Pure functions, no I/O. Constants live in a
`PolicyConfig` dataclass loaded from environment variables, so tuning for a demo
never means editing logic.

### Escrow sizing

```
escrow(trust) = clamp(round(BASE * (1 - trust)), MIN, MAX)
BASE = 40, MIN = 5, MAX = 40
```

Trust 0.30 → 28 credits. Trust 0.72 → 11 credits. Carried over from the existing
`calculate_escrow_amount`, with an added upper clamp and integer discipline.

### Regeneration

Lazy — computed on account read, never scheduled.

```
rate(trust)  = 20/day if trust > 0.8
             = 10/day if trust > 0.5
             =  5/day otherwise

regen        = floor(days_since(last_regen_at) * rate)
available    = min(available + regen, REGEN_CAP = 100)
```

Any read of an account first materializes owed regeneration as a real `regen`
journal entry funded from `platform_mint`, then answers. This avoids both a
scheduler and the drift a scheduler would introduce. The cap keeps regeneration a
safety net rather than an income stream.

### Participation floor

```
if available < FLOOR (= MIN escrow = 5)
   and the user has no RESERVED escrow
   and now - last_topup_at > COOLDOWN (24h):
       top up to FLOOR from platform_mint
```

This is the death-spiral fix: a user can always afford some session. The cooldown
and the active-reservation guard prevent it becoming farmable.

### Settlement

```
SUCCESSFUL and qa >= 0.85       -> FULL_RELEASE
PARTIAL, or 0.5 <= qa < 0.85    -> PARTIAL_RELEASE, release = round(stake * qa)
DISPUTE, or qa < 0.5            -> PENALTY
no_show                         -> PENALTY, NO_SHOW_PENALTY = 20 to counterparty
```

**FULL_RELEASE**, per user: stake returned, plus
`teaching_bonus = round(5 * quality)`, plus `engagement_bonus = 2` when
`engagement >= 0.6`. Platform fee is 5% of the pool with a floor of 1 credit, to
`platform_revenue`.

**PARTIAL_RELEASE**: the released portion returns to the user, the remainder goes
to `platform_revenue`, and bonuses scale by `qa_score`.

**PENALTY**: forfeiture is conditional on `no_show`, since that is the only
signal identifying which party is at fault:

- One party no-showed — their stake goes **to the counterparty**, not the
  platform. The wronged party is made whole, which is where the deterrent
  actually lives.
- Both no-showed — the whole pool goes to `platform_revenue`.
- Neither no-showed (a mutual dispute) — stakes are returned, no bonuses are
  paid, and no fee is charged. Net zero for both parties.

Every branch returns a balanced set of ledger lines.

---

## 6. API

All under `/resource/`. Mutating endpoints accept an `Idempotency-Key` header.

| Method | Path | Purpose |
|---|---|---|
| POST | `/resource/accounts` | Create user accounts with `INITIAL_GRANT = 100`; idempotent on `user_id` |
| GET | `/resource/accounts/{user_id}` | Balances; materializes regeneration and floor first |
| POST | `/resource/escrow/reserve` | `{session_id, participants: [{user_id, trust_score}]}` — locks all in one transaction |
| POST | `/resource/escrow/settle` | `{session_id, verdict_type, qa_score, per_user: {quality, engagement, no_show}}` |
| POST | `/resource/escrow/void` | `{session_id, reason}` |
| POST | `/resource/escrow/{session_id}/hold` | `{reason}` → `HELD` |
| GET | `/resource/escrow/{session_id}` | Escrow rows and states |
| GET | `/resource/ledger/{user_id}` | Paginated entries with running balance |
| POST | `/resource/disputes/{session_id}/resolve` | Admin: `{resolution: settle\|void, note}` |
| GET | `/resource/health` | Liveness plus last reconcile status |

### Reserve is all-or-nothing by construction

A single call carries every participant. Inside one transaction the handler locks
the relevant accounts with `SELECT ... FOR UPDATE`, ordered by `user_id` for a
deterministic lock order that cannot deadlock, checks sufficiency for all
participants, then writes one journal entry containing every line (for two users:
A available → A locked, B available → B locked).

Either everyone is locked or nobody is. This removes the partial-lock case the
current in-process code hand-rolls with a manual rollback.

### Backend changes

In `routes.py`, `start_session` replaces its two `lock_escrow` calls with one
client call. A 409 `INSUFFICIENT_CREDITS` becomes an HTTP 400 to the user naming
which participant is short and how long until regeneration covers the gap. The
confirm and verdict path replaces `apply_settlement` with one `settle` call.

---

## 7. Failure handling

| Failure | Behavior |
|---|---|
| `resource_agent` unreachable at reserve | Session does not start, 503, nothing locked |
| Reserve times out but committed | Retry with the same key hits the unique constraint and returns the original result — no double lock |
| Settle times out but committed | Same — no double payout |
| Session starts, then crashes before verdict | Escrow stays `RESERVED`; the reconciler voids it after TTL |
| Backend never calls settle | Same — reconciler |
| Toxicity or fraud flag raised | `hold` moves escrow to `HELD`, excluded from reconciliation, awaits admin |

### Reconciler

A background task inside `resource_agent`, every 5 minutes:

1. `RESERVED` escrows older than `RESERVE_TTL` (24h) whose session is not live are
   voided and the credits returned.
2. Every `accounts.balance` is re-derived from `ledger_lines`. Any mismatch is
   logged at ERROR and surfaced on `/resource/health`. It is **never** silently
   corrected — drift means a bug, and it must stay visible.
3. Every journal entry is asserted to sum to zero.

---

## 8. Cutover

Fresh start. `resource_agent` begins with an empty Postgres database. The old
SQLite credit tables (`wallets`, `escrows`, `credit_transactions`) are dropped
from backend in the same change that removes the models. Existing wallet history
is synthetic (every balance is 999,999), so nothing of value is lost.

Users get `INITIAL_GRANT = 100` credits on account creation. Scarcity is live
from day one: a session can genuinely fail to start for insufficient credits,
which is the point of the subsystem.

---

## 9. Testing

The existing `tests/` directory holds end-to-end tests against running services.
This work adds a unit layer, because the policy math is the part that must be
provably correct.

1. **Policy unit tests** — `apps/resource_agent/tests/test_policy.py`. No DB, no
   HTTP. Escrow sizing across the trust range including both clamps;
   regeneration including cap and zero-elapsed; floor including cooldown and the
   active-reservation guard; every settlement branch. Plus a property test: for
   randomized verdicts, scores, and trust values, generated lines always sum to
   zero and no balance goes negative. This is the test that backs the ledger
   claim.
2. **Ledger and repository tests** — against a real Postgres. Reserve locks all
   participants or none; a retry with the same key produces exactly one entry;
   concurrent reserves on one session leave exactly one winner and one no-op;
   lock ordering does not deadlock under parallel sessions.
3. **API tests** — FastAPI `TestClient`. Idempotency replay on every mutating
   endpoint, 409 on insufficient credits, state-machine rejections (settling a
   `VOIDED` escrow returns 409).
4. **Reconciler tests** — stale `RESERVED` is voided; `HELD` is skipped; injected
   balance drift is detected and reported rather than silently repaired.
5. **End-to-end** — extend `tests/test_e2e_session_lifecycle.py`: start, verify
   locked balances, produce a verdict, verify settled balances, and confirm the
   ledger reconstructs those balances exactly.

---

## 10. Build order

Each step ends with a green build; nothing is left half-wired.

1. Scaffold the service, add the `resource_db` compose service, expose health.
   Nothing calls it yet.
2. Accounts, journal entries, ledger lines, the balanced-entry helper, and test
   layers 1–2.
3. `policy.py` and its full unit suite. Still nothing calls it.
4. Escrow state machine and the reserve, settle, void, and hold endpoints, with
   idempotency.
5. Regeneration and floor, materialized lazily on account read.
6. Reconciler and its `/resource/health` reporting.
7. **Cutover** — delete `app/escrow.py` and the three models from backend, add
   `clients/resource.py`, rewire the two touchpoints, drop the old tables. One
   commit; end-to-end tests green afterwards.
8. Disputes and admin resolution.
9. Frontend: balance widget, then escrow panel, then settlement breakdown, then
   ledger page. New `src/api/resource.js` hitting `/resource/` via nginx.
10. Documentation: this spec plus decision entries in
    `Notes/Capstone/DESIGN_DECISIONS.md`.

Step 7 is the only risky step — it is the moment the old path dies. Everything
before it is additive and can land incrementally without breaking demos.

---

## 11. Frontend

- **Balance widget** in the header: available and locked, side by side.
- **Escrow panel** on the session-start screen: each participant's stake and the
  trust score that produced it, so the asymmetry is visible rather than implied.
- **Settlement breakdown** after the verdict: stake returned, plus bonuses, minus
  fee, equals net change.
- **Ledger page**: paginated history with running balance.

All served by `src/api/resource.js` against `/resource/` through nginx.
