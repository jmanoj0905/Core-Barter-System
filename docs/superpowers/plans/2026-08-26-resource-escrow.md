# Resource Agent & Escrow Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `resource_agent`, a fifth service owning all credit state on PostgreSQL, with a double-entry ledger, asymmetric escrow, regeneration, a participation floor, dispute holds, and reconciliation — replacing the decorative escrow code inside `apps/backend`.

**Architecture:** New FastAPI service on port 8004 with its own Postgres (`resource_db`). Backend keeps `trust_score` and calls `resource_agent` over HTTP at exactly two points — session start (reserve) and session confirm (settle). All money movement is a balanced journal entry; balances are derived from the ledger and the cached column is verified by a background reconciler. Policy math lives in pure functions with no I/O.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.x async, asyncpg, PostgreSQL 16, pydantic-settings, httpx, pytest + pytest-asyncio + hypothesis, React 19 + Vite + Tailwind 4, Docker Compose, nginx.

**Spec:** `docs/superpowers/specs/2026-08-26-resource-escrow-design.md`

## Global Constraints

- **All credit amounts are integers.** No float ever touches a balance, a ledger line, or an escrow amount. Floats appear only as inputs (`trust_score`, `qa_score`, `quality`, `engagement`) and are converted with `round()` at the policy boundary.
- **Every journal entry must sum to zero.** `post_entry` raises `UnbalancedEntry` otherwise. This is enforced in code, asserted in a property test, and re-verified by the reconciler.
- **Idempotency lives in one place:** the `UNIQUE` index on `journal_entries.idempotency_key`. No separate idempotency table.
- **Key formats (verbatim):** `reserve:{session_id}`, `settle:{session_id}`, `void:{session_id}`, `hold:{session_id}`, `grant:{user_id}`, `regen:{user_id}:{iso_day}`, `topup:{user_id}:{iso_day}`, `dispute_resolve:{session_id}`.
- **Account kinds (verbatim):** `user_available`, `user_locked`, `platform_mint`, `platform_revenue`. No other kinds.
- **Escrow states (verbatim):** `RESERVED`, `SETTLED`, `VOIDED`, `HELD`. `SETTLED` and `VOIDED` are terminal.
- **Settlement modes (verbatim):** `FULL_RELEASE`, `PARTIAL_RELEASE`, `PENALTY`.
- **Policy defaults:** `BASE_ESCROW=40`, `MIN_ESCROW=5`, `MAX_ESCROW=40`, `INITIAL_GRANT=100`, `REGEN_CAP=100`, `FLOOR=5`, `FLOOR_COOLDOWN_HOURS=24`, `TEACHING_BONUS_BASE=5`, `ENGAGEMENT_BONUS=2`, `ENGAGEMENT_THRESHOLD=0.6`, `PLATFORM_FEE_PCT=0.05`, `PLATFORM_FEE_MIN=1`, `NO_SHOW_PENALTY=20`, `RESERVE_TTL_HOURS=24`, `FULL_RELEASE_QA=0.85`, `PARTIAL_RELEASE_QA=0.5`. All overridable by env var of the same name.
- **Service port:** 8004. **Route prefix:** `/resource`. **DB name:** `resource`.
- **Follow existing service layout:** `apps/<service>/{main.py|app/, requirements.txt, Dockerfile}`. Backend is the only service with an `app/` package; `resource_agent` also uses `app/` since it has many modules.
- **No `git commit` unless the step says to.** Steps that commit say so explicitly.

---

## File Structure

**New service — `apps/resource_agent/`**

| File | Responsibility |
|---|---|
| `Dockerfile` | Python 3.11-slim image, uvicorn on 8004 |
| `requirements.txt` | Service dependencies |
| `app/config.py` | `Settings` (DB URL, log level) |
| `app/policy.py` | Pure policy math. No imports from `app.models` or `app.database`. |
| `app/models.py` | `Account`, `JournalEntry`, `LedgerLine`, `Escrow`, `Dispute` |
| `app/database.py` | Engine, session factory, `init_db`, `get_db` |
| `app/ledger.py` | `post_entry`, `get_or_create_account`, `balance_of`, `resolve_movements` |
| `app/accounts.py` | Account creation, grant, lazy regen + floor materialization |
| `app/escrow.py` | Escrow state machine: reserve, settle, void, hold, dispute resolve |
| `app/reconciler.py` | Background loop: stale reservations, balance drift, entry balance |
| `app/schemas.py` | Request/response Pydantic models |
| `app/routes.py` | `/resource/*` endpoints |
| `app/main.py` | App factory, lifespan (init_db, seed system accounts, start reconciler) |
| `tests/conftest.py` | Postgres test engine, per-test truncation |
| `tests/test_policy.py` | Layer 1 — pure policy unit + property tests |
| `tests/test_ledger.py` | Layer 2 — ledger, idempotency, concurrency |
| `tests/test_escrow_api.py` | Layer 3 — API, state machine, 409s |
| `tests/test_reconciler.py` | Layer 4 — stale voids, drift detection |

**Modified — `apps/backend/`**

| File | Change |
|---|---|
| `app/clients/resource.py` | **New.** httpx client, one method per operation |
| `app/config.py` | Add `RESOURCE_URL` |
| `app/models.py` | Delete `Wallet`, `Escrow`, `CreditTransaction` |
| `app/escrow.py` | **Delete** |
| `app/schemas.py` | Delete wallet/escrow/settlement schemas |
| `app/routes.py` | Rewire `start_session` and `confirm_session`; delete `/wallet`, `/escrow/*`, `/settlement`, `/transactions` routes |
| `app/main.py` | Ensure resource accounts for seeded users |
| `requirements.txt` | No change (httpx already present) |

**Modified — other**

| File | Change |
|---|---|
| `docker-compose.yml` | Add `resource_db` and `resource_agent` services |
| `apps/frontend/nginx.conf` | Add `/resource/` proxy; drop `wallet\|escrow\|transactions\|settlement` from the backend regex |
| `apps/frontend/src/api/resource.js` | **New.** Resource API helpers |
| `apps/frontend/src/components/BalanceWidget.jsx` | **New** |
| `apps/frontend/src/components/EscrowPanel.jsx` | **New** |
| `apps/frontend/src/components/SettlementBreakdown.jsx` | **New** |
| `apps/frontend/src/screens/Ledger.jsx` | **New** |
| `apps/frontend/src/screens/Setup.jsx` | Wallet fetch → resource API |
| `apps/frontend/src/screens/LiveSession.jsx` | Show `EscrowPanel` on start |
| `apps/frontend/src/screens/PostSession.jsx` | Escrow fetch → resource API; show `SettlementBreakdown` |
| `apps/frontend/src/App.jsx` | Wallet fetches → resource API; ledger route |
| `tests/test_e2e_session_lifecycle.py` | Assert real balances across the lifecycle |

---

## Task 1: Service scaffold, Postgres, health

**Files:**
- Create: `apps/resource_agent/Dockerfile`
- Create: `apps/resource_agent/requirements.txt`
- Create: `apps/resource_agent/app/__init__.py`
- Create: `apps/resource_agent/app/config.py`
- Create: `apps/resource_agent/app/main.py`
- Create: `apps/resource_agent/tests/__init__.py`
- Create: `apps/resource_agent/pytest.ini`
- Create: `apps/resource_agent/tests/test_health.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: `app.config.settings` with `DATABASE_URL: str`; `app.main.app` (FastAPI instance); compose services `resource_db` (Postgres 16, host port 5433) and `resource_agent` (port 8004).

- [ ] **Step 1: Write the failing test**

Create `apps/resource_agent/tests/test_health.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_returns_ok():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/resource/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["service"] == "resource-agent"
```

Create `apps/resource_agent/tests/__init__.py` (empty file).

Create `apps/resource_agent/pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_functions = test_*
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/resource_agent && python -m pytest tests/test_health.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Create requirements and config**

Create `apps/resource_agent/requirements.txt`:

```
fastapi
uvicorn[standard]
sqlalchemy[asyncio]
asyncpg
pydantic-settings
python-dotenv
httpx
pytest
pytest-asyncio
hypothesis
```

Create `apps/resource_agent/app/__init__.py` (empty file).

Create `apps/resource_agent/app/config.py`:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://resource:resource@resource_db:5432/resource"
    LOG_LEVEL: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
```

- [ ] **Step 4: Create the app**

Create `apps/resource_agent/app/main.py`:

```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("resource-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Resource Agent · Port 8004 · PostgreSQL")
    yield


app = FastAPI(title="Resource Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/resource/health")
async def health():
    return {"service": "resource-agent", "status": "ok"}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd apps/resource_agent && python -m pytest tests/test_health.py -v`
Expected: PASS

- [ ] **Step 6: Create the Dockerfile**

Create `apps/resource_agent/Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
```

- [ ] **Step 7: Add compose services**

In `docker-compose.yml`, add these two services after the `warning_engine` block:

```yaml
  resource_db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: resource
      POSTGRES_PASSWORD: resource
      POSTGRES_DB: resource
    ports:
      - "5433:5432"
    volumes:
      - resource_db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U resource"]
      interval: 5s
      timeout: 5s
      retries: 10
    restart: on-failure

  resource_agent:
    build:
      context: ./apps/resource_agent
    environment:
      DATABASE_URL: postgresql+asyncpg://resource:resource@resource_db:5432/resource
    depends_on:
      resource_db:
        condition: service_healthy
    restart: on-failure
```

In the same file, add `resource_db_data:` under the existing `volumes:` block:

```yaml
volumes:
  db_data:
  model_cache:
  resource_db_data:
```

Port 5433 is published so the test suite can reach Postgres from the host.

- [ ] **Step 8: Verify the service builds and runs**

Run: `docker compose up --build -d resource_db resource_agent && sleep 15 && docker compose exec -T resource_agent curl -sf http://localhost:8004/resource/health`
Expected: `{"service":"resource-agent","status":"ok"}`

- [ ] **Step 9: Commit**

```bash
git add apps/resource_agent docker-compose.yml
git commit -m "feat(resource): scaffold resource_agent service with Postgres"
```

---

## Task 2: Ledger core — models, balanced entries, idempotency

**Files:**
- Create: `apps/resource_agent/app/models.py`
- Create: `apps/resource_agent/app/database.py`
- Create: `apps/resource_agent/app/ledger.py`
- Create: `apps/resource_agent/tests/conftest.py`
- Create: `apps/resource_agent/tests/test_ledger.py`
- Modify: `apps/resource_agent/app/main.py`

**Interfaces:**
- Consumes: `app.config.settings.DATABASE_URL` (Task 1).
- Produces:
  - `app.models`: `Base`, `Account`, `JournalEntry`, `LedgerLine`, `Escrow`, `Dispute`
  - `app.database`: `engine`, `async_session`, `init_db() -> None`, `get_db()` async generator
  - `app.ledger`: `UnbalancedEntry(Exception)`, `async get_or_create_account(db, kind: str, user_id: int | None) -> Account`, `async resolve_movements(db, movements: Sequence[Movement]) -> list[tuple[int, int]]`, `async post_entry(db, *, idempotency_key: str, entry_type: str, session_id: int | None, payload: dict, lines: Sequence[tuple[int, int]]) -> tuple[JournalEntry, bool]`, `async balance_of(db, account_id: int) -> int`
  - `post_entry` returns `(entry, created)`; `created` is `False` on idempotent replay.

- [ ] **Step 1: Write the failing tests**

Create `apps/resource_agent/tests/conftest.py`:

```python
import os

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://resource:resource@localhost:5433/resource",
)

from app.models import Base  # noqa: E402

TEST_DB_URL = os.environ["DATABASE_URL"]

test_engine = create_async_engine(TEST_DB_URL, echo=False)
test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)

TABLES = ["ledger_lines", "journal_entries", "escrows", "disputes", "accounts"]


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _create_schema():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await test_engine.dispose()


@pytest_asyncio.fixture
async def db():
    async with test_session_factory() as session:
        yield session
        await session.rollback()

    async with test_engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"))


@pytest.fixture
def session_factory():
    """For tests that need two independent concurrent sessions."""
    return test_session_factory
```

Create `apps/resource_agent/tests/test_ledger.py`:

```python
import asyncio

import pytest

from app.ledger import (
    UnbalancedEntry,
    balance_of,
    get_or_create_account,
    post_entry,
)


async def _two_accounts(db):
    src = await get_or_create_account(db, "platform_mint", None)
    dst = await get_or_create_account(db, "user_available", 1)
    await db.flush()
    return src, dst


@pytest.mark.asyncio
async def test_get_or_create_account_is_idempotent(db):
    first = await get_or_create_account(db, "user_available", 7)
    await db.commit()
    second = await get_or_create_account(db, "user_available", 7)
    assert first.id == second.id


@pytest.mark.asyncio
async def test_post_entry_moves_credits_and_balances(db):
    src, dst = await _two_accounts(db)
    entry, created = await post_entry(
        db,
        idempotency_key="grant:1",
        entry_type="grant",
        session_id=None,
        payload={"user_id": 1},
        lines=[(src.id, -100), (dst.id, 100)],
    )
    await db.commit()

    assert created is True
    assert entry.idempotency_key == "grant:1"
    assert await balance_of(db, dst.id) == 100
    assert await balance_of(db, src.id) == -100


@pytest.mark.asyncio
async def test_post_entry_rejects_unbalanced_lines(db):
    src, dst = await _two_accounts(db)
    with pytest.raises(UnbalancedEntry):
        await post_entry(
            db,
            idempotency_key="grant:2",
            entry_type="grant",
            session_id=None,
            payload={},
            lines=[(src.id, -100), (dst.id, 99)],
        )


@pytest.mark.asyncio
async def test_post_entry_replay_returns_original_without_double_posting(db):
    src, dst = await _two_accounts(db)
    lines = [(src.id, -100), (dst.id, 100)]

    first, created_first = await post_entry(
        db, idempotency_key="grant:3", entry_type="grant",
        session_id=None, payload={}, lines=lines,
    )
    await db.commit()

    second, created_second = await post_entry(
        db, idempotency_key="grant:3", entry_type="grant",
        session_id=None, payload={}, lines=lines,
    )
    await db.commit()

    assert created_first is True
    assert created_second is False
    assert first.id == second.id
    assert await balance_of(db, dst.id) == 100


@pytest.mark.asyncio
async def test_cached_balance_matches_derived_balance(db):
    src, dst = await _two_accounts(db)
    await post_entry(
        db, idempotency_key="grant:4", entry_type="grant",
        session_id=None, payload={}, lines=[(src.id, -40), (dst.id, 40)],
    )
    await db.commit()
    await db.refresh(dst)

    assert dst.balance == await balance_of(db, dst.id)


@pytest.mark.asyncio
async def test_concurrent_replay_posts_exactly_one_entry(db, session_factory):
    src, dst = await _two_accounts(db)
    await db.commit()
    src_id, dst_id = src.id, dst.id

    async def attempt():
        async with session_factory() as session:
            _, created = await post_entry(
                session, idempotency_key="grant:5", entry_type="grant",
                session_id=None, payload={}, lines=[(src_id, -10), (dst_id, 10)],
            )
            await session.commit()
            return created

    results = await asyncio.gather(attempt(), attempt())

    assert sorted(results) == [False, True]
    assert await balance_of(db, dst_id) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose up -d resource_db && cd apps/resource_agent && python -m pytest tests/test_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.models'`

- [ ] **Step 3: Write the models**

Create `apps/resource_agent/app/models.py`:

```python
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"
    __table_args__ = (UniqueConstraint("kind", "user_id", name="uq_account_kind_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    balance: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    floor: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    regen_rate: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_regen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_topup_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    entry_type: Mapped[str] = mapped_column(String(32), nullable=False)
    session_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB().with_variant(JSON, "sqlite"), default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class LedgerLine(Base):
    __tablename__ = "ledger_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entry_id: Mapped[int] = mapped_column(
        ForeignKey("journal_entries.id"), nullable=False, index=True
    )
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id"), nullable=False, index=True
    )
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)


class Escrow(Base):
    __tablename__ = "escrows"
    __table_args__ = (UniqueConstraint("session_id", "user_id", name="uq_escrow_session_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    reserve_entry_id: Mapped[int | None] = mapped_column(
        ForeignKey("journal_entries.id"), nullable=True
    )
    settle_entry_id: Mapped[int | None] = mapped_column(
        ForeignKey("journal_entries.id"), nullable=True
    )
    reserved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    settled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Dispute(Base):
    __tablename__ = "disputes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(16), default="OPEN", nullable=False)
    resolution: Mapped[str | None] = mapped_column(String(16), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
```

- [ ] **Step 4: Write the database module**

Create `apps/resource_agent/app/database.py`:

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base

engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with async_session() as session:
        yield session
```

- [ ] **Step 5: Write the ledger module**

Create `apps/resource_agent/app/ledger.py`:

```python
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Account, JournalEntry, LedgerLine

SYSTEM_KINDS = ("platform_mint", "platform_revenue")


class UnbalancedEntry(Exception):
    """Raised when the ledger lines of a journal entry do not sum to zero."""


@dataclass(frozen=True)
class Movement:
    """A policy-level instruction: move `amount` into the named account.

    `account` is one of: "user_available:{id}", "user_locked:{id}",
    "platform_mint", "platform_revenue".
    """

    account: str
    amount: int


async def get_or_create_account(db: AsyncSession, kind: str, user_id: int | None) -> Account:
    stmt = select(Account).where(Account.kind == kind, Account.user_id == user_id)
    account = (await db.execute(stmt)).scalar_one_or_none()
    if account:
        return account

    account = Account(kind=kind, user_id=user_id, balance=0)
    db.add(account)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        account = (await db.execute(stmt)).scalar_one()
    return account


def _parse_account_key(key: str) -> tuple[str, int | None]:
    if ":" in key:
        kind, raw_user_id = key.split(":", 1)
        return kind, int(raw_user_id)
    return key, None


async def resolve_movements(
    db: AsyncSession, movements: Sequence[Movement]
) -> list[tuple[int, int]]:
    """Turn policy Movements into (account_id, amount) ledger lines."""
    lines: list[tuple[int, int]] = []
    for movement in movements:
        kind, user_id = _parse_account_key(movement.account)
        account = await get_or_create_account(db, kind, user_id)
        lines.append((account.id, movement.amount))
    return lines


async def post_entry(
    db: AsyncSession,
    *,
    idempotency_key: str,
    entry_type: str,
    session_id: int | None,
    payload: dict,
    lines: Sequence[tuple[int, int]],
) -> tuple[JournalEntry, bool]:
    """Post a balanced journal entry. Returns (entry, created).

    On idempotent replay the existing entry is returned with created=False and
    no new ledger lines are written.
    """
    if sum(amount for _, amount in lines) != 0:
        raise UnbalancedEntry(
            f"lines for {idempotency_key} sum to {sum(a for _, a in lines)}, expected 0"
        )

    existing = (
        await db.execute(
            select(JournalEntry).where(JournalEntry.idempotency_key == idempotency_key)
        )
    ).scalar_one_or_none()
    if existing:
        return existing, False

    savepoint = await db.begin_nested()
    entry = JournalEntry(
        idempotency_key=idempotency_key,
        entry_type=entry_type,
        session_id=session_id,
        payload=payload,
    )
    db.add(entry)
    try:
        await db.flush()
    except IntegrityError:
        await savepoint.rollback()
        existing = (
            await db.execute(
                select(JournalEntry).where(
                    JournalEntry.idempotency_key == idempotency_key
                )
            )
        ).scalar_one()
        return existing, False

    for account_id, amount in lines:
        db.add(LedgerLine(entry_id=entry.id, account_id=account_id, amount=amount))
        account = await db.get(Account, account_id, with_for_update=True)
        account.balance += amount

    await db.flush()
    return entry, True


async def balance_of(db: AsyncSession, account_id: int) -> int:
    """Derive an account balance from ledger lines (ignores the cached column)."""
    stmt = select(func.coalesce(func.sum(LedgerLine.amount), 0)).where(
        LedgerLine.account_id == account_id
    )
    return int((await db.execute(stmt)).scalar_one())
```

- [ ] **Step 6: Initialize the schema on startup**

In `apps/resource_agent/app/main.py`, replace the `lifespan` function with:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Resource Agent · Port 8004 · PostgreSQL")

    from app.database import async_session, init_db
    from app.ledger import SYSTEM_KINDS, get_or_create_account

    await init_db()

    async with async_session() as db:
        for kind in SYSTEM_KINDS:
            await get_or_create_account(db, kind, None)
        await db.commit()

    logger.info("Database ready — tables created, system accounts seeded")
    yield
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd apps/resource_agent && python -m pytest tests/test_ledger.py -v`
Expected: PASS — 6 tests

- [ ] **Step 8: Commit**

```bash
git add apps/resource_agent
git commit -m "feat(resource): double-entry ledger with idempotent balanced entries"
```

---

## Task 3: Policy engine

**Files:**
- Create: `apps/resource_agent/app/policy.py`
- Create: `apps/resource_agent/tests/test_policy.py`

**Interfaces:**
- Consumes: `app.ledger.Movement` (Task 2) — imported for the return type only; `policy.py` performs no I/O.
- Produces:
  - `PolicyConfig` frozen dataclass with `from_env() -> PolicyConfig`
  - `calculate_escrow(trust_score: float, cfg: PolicyConfig) -> int`
  - `regen_rate(trust_score: float, cfg: PolicyConfig) -> int`
  - `regen_amount(last_regen_at, now, trust_score, current_balance, cfg) -> int`
  - `floor_topup_amount(available, has_active_reservation, last_topup_at, now, cfg) -> int`
  - `ParticipantOutcome` frozen dataclass
  - `SettlementPlan` frozen dataclass with `mode`, `movements`, `breakdown`
  - `plan_settlement(verdict_type, qa_score, participants, cfg) -> SettlementPlan`

- [ ] **Step 1: Write the failing tests**

Create `apps/resource_agent/tests/test_policy.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.policy import (
    ParticipantOutcome,
    PolicyConfig,
    calculate_escrow,
    floor_topup_amount,
    plan_settlement,
    regen_amount,
    regen_rate,
)

CFG = PolicyConfig()
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


# ── escrow sizing ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "trust,expected",
    [
        (0.0, 40),    # no trust -> full base stake
        (0.30, 28),   # spec example: new user
        (0.72, 11),   # spec example: experienced user
        (0.95, 5),    # clamped up to MIN
        (1.0, 5),     # clamped up to MIN
    ],
)
def test_calculate_escrow_matches_spec_examples(trust, expected):
    assert calculate_escrow(trust, CFG) == expected


def test_calculate_escrow_never_exceeds_max():
    assert calculate_escrow(-5.0, CFG) == CFG.max_escrow


def test_calculate_escrow_returns_int():
    assert isinstance(calculate_escrow(0.37, CFG), int)


# ── regeneration ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "trust,expected", [(0.9, 20), (0.81, 20), (0.6, 10), (0.51, 10), (0.5, 5), (0.1, 5)]
)
def test_regen_rate_bands(trust, expected):
    assert regen_rate(trust, CFG) == expected


def test_regen_amount_is_zero_for_fresh_account():
    assert regen_amount(NOW, NOW, 0.5, 10, CFG) == 0


def test_regen_amount_accrues_per_whole_day():
    last = NOW - timedelta(days=3)
    assert regen_amount(last, NOW, 0.5, 0, CFG) == 15  # 3 days * 5/day


def test_regen_amount_ignores_partial_days():
    last = NOW - timedelta(hours=47)
    assert regen_amount(last, NOW, 0.5, 0, CFG) == 5  # 1 whole day only


def test_regen_amount_is_capped_at_regen_cap():
    last = NOW - timedelta(days=100)
    assert regen_amount(last, NOW, 0.9, 0, CFG) == CFG.regen_cap


def test_regen_amount_is_zero_when_already_at_cap():
    last = NOW - timedelta(days=10)
    assert regen_amount(last, NOW, 0.9, CFG.regen_cap, CFG) == 0


def test_regen_amount_is_zero_when_never_regenerated():
    assert regen_amount(None, NOW, 0.5, 0, CFG) == 0


# ── participation floor ─────────────────────────────────────────────────────

def test_floor_topup_lifts_a_stranded_user_to_the_floor():
    assert floor_topup_amount(1, False, None, NOW, CFG) == 4  # 1 -> 5


def test_floor_topup_is_zero_above_the_floor():
    assert floor_topup_amount(CFG.floor, False, None, NOW, CFG) == 0


def test_floor_topup_is_blocked_by_an_active_reservation():
    assert floor_topup_amount(0, True, None, NOW, CFG) == 0


def test_floor_topup_is_blocked_inside_the_cooldown():
    last = NOW - timedelta(hours=CFG.floor_cooldown_hours - 1)
    assert floor_topup_amount(0, False, last, NOW, CFG) == 0


def test_floor_topup_is_allowed_after_the_cooldown():
    last = NOW - timedelta(hours=CFG.floor_cooldown_hours + 1)
    assert floor_topup_amount(0, False, last, NOW, CFG) == CFG.floor


# ── settlement ──────────────────────────────────────────────────────────────

def _participants(**overrides):
    base = dict(quality=0.9, engagement=0.8, no_show=False)
    base.update(overrides)
    return [
        ParticipantOutcome(user_id=1, stake=28, **base),
        ParticipantOutcome(user_id=2, stake=11, **base),
    ]


def test_full_release_returns_stakes_and_pays_bonuses():
    plan = plan_settlement("SUCCESSFUL", 0.92, _participants(), CFG)

    assert plan.mode == "FULL_RELEASE"
    user_1 = plan.breakdown[1]
    assert user_1["stake_returned"] == 28
    assert user_1["teaching_bonus"] == 5   # round(5 * 0.9)
    assert user_1["engagement_bonus"] == 2  # engagement 0.8 >= 0.6


def test_full_release_withholds_engagement_bonus_below_threshold():
    plan = plan_settlement("SUCCESSFUL", 0.92, _participants(engagement=0.4), CFG)
    assert plan.breakdown[1]["engagement_bonus"] == 0


def test_full_release_charges_platform_fee_on_the_pool():
    plan = plan_settlement("SUCCESSFUL", 0.92, _participants(), CFG)
    # pool = 28 + 11 = 39; 5% = 1.95 -> round -> 2
    assert sum(b["fee_share"] for b in plan.breakdown.values()) == 2


def test_partial_release_scales_with_qa_score():
    plan = plan_settlement("PARTIAL", 0.60, _participants(), CFG)

    assert plan.mode == "PARTIAL_RELEASE"
    assert plan.breakdown[1]["stake_returned"] == 17  # round(28 * 0.60)
    assert plan.breakdown[2]["stake_returned"] == 7   # round(11 * 0.60)


def test_dispute_is_a_penalty():
    plan = plan_settlement("DISPUTE", 0.2, _participants(), CFG)
    assert plan.mode == "PENALTY"


def test_low_qa_score_overrides_a_successful_verdict():
    plan = plan_settlement("SUCCESSFUL", 0.10, _participants(), CFG)
    assert plan.mode == "PENALTY"


def test_no_show_forfeits_the_stake_to_the_counterparty():
    participants = [
        ParticipantOutcome(user_id=1, stake=28, quality=0.0, engagement=0.0, no_show=True),
        ParticipantOutcome(user_id=2, stake=11, quality=0.9, engagement=0.8, no_show=False),
    ]
    plan = plan_settlement("DISPUTE", 0.0, participants, CFG)

    assert plan.mode == "PENALTY"
    assert plan.breakdown[1]["penalty"] == 28
    # the wronged party is made whole: own stake back + forfeited stake
    assert plan.breakdown[2]["net"] > 0


def test_mutual_dispute_returns_stakes_but_pays_no_bonuses():
    """No party is identifiably at fault, so nobody is penalised and nobody is
    rewarded. Forfeiture requires a no_show."""
    plan = plan_settlement("DISPUTE", 0.2, _participants(), CFG)

    assert plan.mode == "PENALTY"
    for user_id in (1, 2):
        row = plan.breakdown[user_id]
        assert row["stake_returned"] == row["stake"]
        assert row["penalty"] == 0
        assert row["teaching_bonus"] == 0
        assert row["engagement_bonus"] == 0
        assert row["net"] == 0


def test_mutual_no_show_forfeits_the_whole_pool_to_the_platform():
    participants = [
        ParticipantOutcome(user_id=1, stake=28, quality=0.0, engagement=0.0, no_show=True),
        ParticipantOutcome(user_id=2, stake=11, quality=0.0, engagement=0.0, no_show=True),
    ]
    plan = plan_settlement("DISPUTE", 0.0, participants, CFG)

    platform = sum(m.amount for m in plan.movements if m.account == "platform_revenue")
    assert platform == 39
    assert plan.breakdown[1]["net"] == -28
    assert plan.breakdown[2]["net"] == -11


def test_settlement_movements_always_balance():
    for verdict, qa in [("SUCCESSFUL", 0.95), ("PARTIAL", 0.6), ("DISPUTE", 0.1)]:
        plan = plan_settlement(verdict, qa, _participants(), CFG)
        assert sum(m.amount for m in plan.movements) == 0, verdict


@hyp_settings(max_examples=300)
@given(
    verdict=st.sampled_from(["SUCCESSFUL", "PARTIAL", "DISPUTE"]),
    qa=st.floats(min_value=0.0, max_value=1.0),
    stake_a=st.integers(min_value=5, max_value=40),
    stake_b=st.integers(min_value=5, max_value=40),
    quality=st.floats(min_value=0.0, max_value=1.0),
    engagement=st.floats(min_value=0.0, max_value=1.0),
    no_show_a=st.booleans(),
)
def test_property_every_plan_balances_and_uses_integers(
    verdict, qa, stake_a, stake_b, quality, engagement, no_show_a
):
    participants = [
        ParticipantOutcome(1, stake_a, quality, engagement, no_show_a),
        ParticipantOutcome(2, stake_b, quality, engagement, False),
    ]
    plan = plan_settlement(verdict, qa, participants, CFG)

    assert sum(m.amount for m in plan.movements) == 0
    assert all(isinstance(m.amount, int) for m in plan.movements)
    for row in plan.breakdown.values():
        assert all(isinstance(value, int) for value in row.values())


@hyp_settings(max_examples=200)
@given(
    verdict=st.sampled_from(["SUCCESSFUL", "PARTIAL", "DISPUTE"]),
    qa=st.floats(min_value=0.0, max_value=1.0),
    stake_a=st.integers(min_value=5, max_value=40),
    stake_b=st.integers(min_value=5, max_value=40),
)
def test_property_locked_accounts_are_fully_drained(verdict, qa, stake_a, stake_b):
    """Settlement must empty both locked accounts — no credits stranded."""
    participants = [
        ParticipantOutcome(1, stake_a, 0.8, 0.7, False),
        ParticipantOutcome(2, stake_b, 0.8, 0.7, False),
    ]
    plan = plan_settlement(verdict, qa, participants, CFG)

    drained = {
        1: sum(m.amount for m in plan.movements if m.account == "user_locked:1"),
        2: sum(m.amount for m in plan.movements if m.account == "user_locked:2"),
    }
    assert drained[1] == -stake_a
    assert drained[2] == -stake_b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/resource_agent && python -m pytest tests/test_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.policy'`

- [ ] **Step 3: Write the policy module**

Create `apps/resource_agent/app/policy.py`:

```python
"""Pure credit policy. No I/O, no database, no imports from app.models.

Every function here is deterministic given its arguments, which is what makes
the property tests in tests/test_policy.py meaningful.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from app.ledger import Movement


@dataclass(frozen=True)
class PolicyConfig:
    base_escrow: int = 40
    min_escrow: int = 5
    max_escrow: int = 40
    initial_grant: int = 100
    regen_cap: int = 100
    floor: int = 5
    floor_cooldown_hours: int = 24
    teaching_bonus_base: int = 5
    engagement_bonus: int = 2
    engagement_threshold: float = 0.6
    platform_fee_pct: float = 0.05
    platform_fee_min: int = 1
    no_show_penalty: int = 20
    reserve_ttl_hours: int = 24
    full_release_qa: float = 0.85
    partial_release_qa: float = 0.5

    @classmethod
    def from_env(cls) -> "PolicyConfig":
        def _int(name: str, default: int) -> int:
            return int(os.environ.get(name, default))

        def _float(name: str, default: float) -> float:
            return float(os.environ.get(name, default))

        return cls(
            base_escrow=_int("BASE_ESCROW", 40),
            min_escrow=_int("MIN_ESCROW", 5),
            max_escrow=_int("MAX_ESCROW", 40),
            initial_grant=_int("INITIAL_GRANT", 100),
            regen_cap=_int("REGEN_CAP", 100),
            floor=_int("FLOOR", 5),
            floor_cooldown_hours=_int("FLOOR_COOLDOWN_HOURS", 24),
            teaching_bonus_base=_int("TEACHING_BONUS_BASE", 5),
            engagement_bonus=_int("ENGAGEMENT_BONUS", 2),
            engagement_threshold=_float("ENGAGEMENT_THRESHOLD", 0.6),
            platform_fee_pct=_float("PLATFORM_FEE_PCT", 0.05),
            platform_fee_min=_int("PLATFORM_FEE_MIN", 1),
            no_show_penalty=_int("NO_SHOW_PENALTY", 20),
            reserve_ttl_hours=_int("RESERVE_TTL_HOURS", 24),
            full_release_qa=_float("FULL_RELEASE_QA", 0.85),
            partial_release_qa=_float("PARTIAL_RELEASE_QA", 0.5),
        )


@dataclass(frozen=True)
class ParticipantOutcome:
    user_id: int
    stake: int
    quality: float
    engagement: float
    no_show: bool


@dataclass(frozen=True)
class SettlementPlan:
    mode: str
    movements: tuple[Movement, ...]
    breakdown: dict[int, dict[str, int]]


# ── escrow sizing ───────────────────────────────────────────────────────────

def calculate_escrow(trust_score: float, cfg: PolicyConfig) -> int:
    """Asymmetric stake: less trust means more skin in the game."""
    raw = round(cfg.base_escrow * (1 - trust_score))
    return int(max(cfg.min_escrow, min(cfg.max_escrow, raw)))


# ── regeneration ────────────────────────────────────────────────────────────

def regen_rate(trust_score: float, cfg: PolicyConfig) -> int:
    if trust_score > 0.8:
        return 20
    if trust_score > 0.5:
        return 10
    return 5


def regen_amount(
    last_regen_at: datetime | None,
    now: datetime,
    trust_score: float,
    current_balance: int,
    cfg: PolicyConfig,
) -> int:
    """Credits owed since last_regen_at, capped so the balance never exceeds
    regen_cap. Only whole elapsed days accrue."""
    if last_regen_at is None:
        return 0

    elapsed_days = int((now - last_regen_at).total_seconds() // 86400)
    if elapsed_days <= 0:
        return 0

    owed = elapsed_days * regen_rate(trust_score, cfg)
    headroom = cfg.regen_cap - current_balance
    return int(max(0, min(owed, headroom)))


# ── participation floor ─────────────────────────────────────────────────────

def floor_topup_amount(
    available: int,
    has_active_reservation: bool,
    last_topup_at: datetime | None,
    now: datetime,
    cfg: PolicyConfig,
) -> int:
    """The death-spiral guard: lift a stranded user back to the floor.

    Blocked while an escrow is reserved (so it cannot be farmed mid-session)
    and while inside the cooldown window.
    """
    if available >= cfg.floor:
        return 0
    if has_active_reservation:
        return 0
    if last_topup_at is not None:
        elapsed_hours = (now - last_topup_at).total_seconds() / 3600
        if elapsed_hours < cfg.floor_cooldown_hours:
            return 0
    return int(cfg.floor - available)


# ── settlement ──────────────────────────────────────────────────────────────

def _mode_for(verdict_type: str, qa_score: float, cfg: PolicyConfig) -> str:
    if verdict_type == "DISPUTE" or qa_score < cfg.partial_release_qa:
        return "PENALTY"
    if verdict_type == "SUCCESSFUL" and qa_score >= cfg.full_release_qa:
        return "FULL_RELEASE"
    return "PARTIAL_RELEASE"


def _platform_fee(pool: int, cfg: PolicyConfig) -> int:
    if pool <= 0:
        return 0
    return int(max(cfg.platform_fee_min, round(pool * cfg.platform_fee_pct)))


def plan_settlement(
    verdict_type: str,
    qa_score: float,
    participants: Sequence[ParticipantOutcome],
    cfg: PolicyConfig,
) -> SettlementPlan:
    """Turn a verdict into balanced credit movements.

    Invariant: the returned movements always sum to zero, and every locked
    account is drained by exactly the participant's stake.
    """
    mode = _mode_for(verdict_type, qa_score, cfg)
    movements: list[Movement] = []
    breakdown: dict[int, dict[str, int]] = {}

    pool = sum(p.stake for p in participants)
    fee = _platform_fee(pool, cfg) if mode in ("FULL_RELEASE", "PARTIAL_RELEASE") else 0
    fee_shares = _split_fee(fee, participants)

    forfeited_total = 0
    forfeited_by: dict[int, int] = {}

    # Pass 1: drain locked accounts, decide each participant's returned stake.
    for participant in participants:
        user_id = participant.user_id
        movements.append(Movement(f"user_locked:{user_id}", -participant.stake))

        if mode == "PENALTY" and participant.no_show:
            returned = 0
        elif mode == "PENALTY":
            returned = participant.stake
        elif mode == "FULL_RELEASE":
            returned = participant.stake
        else:
            returned = int(round(participant.stake * qa_score))

        forfeited = participant.stake - returned
        if mode == "PENALTY":
            forfeited_by[user_id] = forfeited
            forfeited_total += forfeited

        breakdown[user_id] = {
            "stake": participant.stake,
            "stake_returned": returned,
            "teaching_bonus": 0,
            "engagement_bonus": 0,
            "penalty": forfeited if mode == "PENALTY" else 0,
            "fee_share": fee_shares.get(user_id, 0),
            "compensation": 0,
            "net": 0,
        }

    # Pass 2: bonuses (release modes only).
    if mode in ("FULL_RELEASE", "PARTIAL_RELEASE"):
        scale = 1.0 if mode == "FULL_RELEASE" else qa_score
        for participant in participants:
            row = breakdown[participant.user_id]
            row["teaching_bonus"] = int(
                round(cfg.teaching_bonus_base * participant.quality * scale)
            )
            row["engagement_bonus"] = (
                cfg.engagement_bonus
                if participant.engagement >= cfg.engagement_threshold
                else 0
            )

    # Pass 3: penalty compensation — forfeited stakes go to the wronged parties.
    if mode == "PENALTY" and forfeited_total > 0:
        wronged = [p for p in participants if forfeited_by.get(p.user_id, 0) == 0]
        if wronged:
            shares = _split_evenly(forfeited_total, [p.user_id for p in wronged])
            for user_id, share in shares.items():
                breakdown[user_id]["compensation"] = share
        else:
            # Everyone forfeited: the pool goes to the platform.
            movements.append(Movement("platform_revenue", forfeited_total))

    # Pass 4: emit credit movements and compute net change per participant.
    minted = 0
    for participant in participants:
        user_id = participant.user_id
        row = breakdown[user_id]
        credited = (
            row["stake_returned"]
            + row["teaching_bonus"]
            + row["engagement_bonus"]
            + row["compensation"]
            - row["fee_share"]
        )
        if credited:
            movements.append(Movement(f"user_available:{user_id}", credited))
        row["net"] = credited - row["stake"]
        minted += row["teaching_bonus"] + row["engagement_bonus"]

    if fee:
        movements.append(Movement("platform_revenue", fee))
    if minted:
        movements.append(Movement("platform_mint", -minted))

    # Remainder of the pool not returned to anyone (partial release) goes to
    # platform revenue, which is also what balances the entry.
    residual = -sum(m.amount for m in movements)
    if residual:
        movements.append(Movement("platform_revenue", residual))

    return SettlementPlan(mode=mode, movements=tuple(movements), breakdown=breakdown)


def _split_evenly(total: int, user_ids: Sequence[int]) -> dict[int, int]:
    """Split `total` across user_ids, giving remainders to the earliest ids."""
    if not user_ids:
        return {}
    base, remainder = divmod(total, len(user_ids))
    return {
        user_id: base + (1 if index < remainder else 0)
        for index, user_id in enumerate(user_ids)
    }


def _split_fee(fee: int, participants: Sequence[ParticipantOutcome]) -> dict[int, int]:
    if fee <= 0:
        return {}
    return _split_evenly(fee, [p.user_id for p in participants])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/resource_agent && python -m pytest tests/test_policy.py -v`
Expected: PASS — all parametrized cases plus both property tests

- [ ] **Step 5: Commit**

```bash
git add apps/resource_agent/app/policy.py apps/resource_agent/tests/test_policy.py
git commit -m "feat(resource): policy engine for escrow, regen, floor, settlement"
```

---

## Task 4: Accounts — grants, lazy regeneration, floor

**Files:**
- Create: `apps/resource_agent/app/accounts.py`
- Create: `apps/resource_agent/tests/test_accounts.py`

**Interfaces:**
- Consumes: `app.ledger.{Movement, get_or_create_account, post_entry, resolve_movements, balance_of}`, `app.policy.{PolicyConfig, regen_amount, floor_topup_amount, regen_rate}`, `app.models.{Account, Escrow}`.
- Produces:
  - `async ensure_accounts(db, user_id: int, cfg: PolicyConfig) -> dict` — creates both user accounts, grants `initial_grant` once (key `grant:{user_id}`)
  - `async materialize(db, user_id: int, trust_score: float, cfg: PolicyConfig, now: datetime | None = None) -> None` — posts owed regen and floor top-up entries
  - `async account_summary(db, user_id: int, trust_score: float, cfg: PolicyConfig) -> dict` with keys `user_id`, `available`, `locked`, `regen_rate`, `next_regen_at`

- [ ] **Step 1: Write the failing tests**

Create `apps/resource_agent/tests/test_accounts.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from app.accounts import account_summary, ensure_accounts, materialize
from app.ledger import get_or_create_account, post_entry
from app.models import Escrow
from app.policy import PolicyConfig

CFG = PolicyConfig()
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_ensure_accounts_grants_initial_credits(db):
    summary = await ensure_accounts(db, 1, CFG)
    await db.commit()

    assert summary["available"] == CFG.initial_grant
    assert summary["locked"] == 0


@pytest.mark.asyncio
async def test_ensure_accounts_is_idempotent(db):
    await ensure_accounts(db, 1, CFG)
    await db.commit()
    summary = await ensure_accounts(db, 1, CFG)
    await db.commit()

    assert summary["available"] == CFG.initial_grant


@pytest.mark.asyncio
async def test_materialize_credits_owed_regeneration(db):
    await ensure_accounts(db, 1, CFG)
    account = await get_or_create_account(db, "user_available", 1)
    # Spend down so there is headroom under the regen cap.
    mint = await get_or_create_account(db, "platform_mint", None)
    await post_entry(
        db, idempotency_key="spend:1", entry_type="penalty", session_id=None,
        payload={}, lines=[(account.id, -90), (mint.id, 90)],
    )
    account.last_regen_at = NOW - timedelta(days=2)
    await db.commit()

    await materialize(db, 1, trust_score=0.5, cfg=CFG, now=NOW)
    await db.commit()
    await db.refresh(account)

    assert account.balance == 20  # 10 remaining + 2 days * 5/day
    assert account.last_regen_at == NOW


@pytest.mark.asyncio
async def test_materialize_tops_up_a_stranded_user_to_the_floor(db):
    await ensure_accounts(db, 1, CFG)
    account = await get_or_create_account(db, "user_available", 1)
    mint = await get_or_create_account(db, "platform_mint", None)
    await post_entry(
        db, idempotency_key="spend:2", entry_type="penalty", session_id=None,
        payload={}, lines=[(account.id, -100), (mint.id, 100)],
    )
    account.last_regen_at = NOW  # no regen owed; isolate the floor path
    await db.commit()

    await materialize(db, 1, trust_score=0.5, cfg=CFG, now=NOW)
    await db.commit()
    await db.refresh(account)

    assert account.balance == CFG.floor
    assert account.last_topup_at == NOW


@pytest.mark.asyncio
async def test_materialize_does_not_top_up_during_an_active_reservation(db):
    await ensure_accounts(db, 1, CFG)
    account = await get_or_create_account(db, "user_available", 1)
    mint = await get_or_create_account(db, "platform_mint", None)
    await post_entry(
        db, idempotency_key="spend:3", entry_type="penalty", session_id=None,
        payload={}, lines=[(account.id, -100), (mint.id, 100)],
    )
    account.last_regen_at = NOW
    db.add(Escrow(session_id=99, user_id=1, amount=5, state="RESERVED"))
    await db.commit()

    await materialize(db, 1, trust_score=0.5, cfg=CFG, now=NOW)
    await db.commit()
    await db.refresh(account)

    assert account.balance == 0


@pytest.mark.asyncio
async def test_account_summary_reports_available_and_locked(db):
    await ensure_accounts(db, 1, CFG)
    await db.commit()

    summary = await account_summary(db, 1, trust_score=0.9, cfg=CFG)

    assert summary["user_id"] == 1
    assert summary["available"] == CFG.initial_grant
    assert summary["locked"] == 0
    assert summary["regen_rate"] == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/resource_agent && python -m pytest tests/test_accounts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.accounts'`

- [ ] **Step 3: Write the accounts module**

Create `apps/resource_agent/app/accounts.py`:

```python
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ledger import get_or_create_account, post_entry
from app.models import Escrow
from app.policy import PolicyConfig, floor_topup_amount, regen_amount, regen_rate


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _has_active_reservation(db: AsyncSession, user_id: int) -> bool:
    stmt = select(Escrow.id).where(Escrow.user_id == user_id, Escrow.state == "RESERVED")
    return (await db.execute(stmt)).first() is not None


async def ensure_accounts(db: AsyncSession, user_id: int, cfg: PolicyConfig) -> dict:
    """Create the user's available and locked accounts and grant starting
    credits exactly once."""
    available = await get_or_create_account(db, "user_available", user_id)
    locked = await get_or_create_account(db, "user_locked", user_id)
    mint = await get_or_create_account(db, "platform_mint", None)

    if available.last_regen_at is None:
        available.last_regen_at = _utcnow()

    await post_entry(
        db,
        idempotency_key=f"grant:{user_id}",
        entry_type="grant",
        session_id=None,
        payload={"user_id": user_id, "amount": cfg.initial_grant},
        lines=[(mint.id, -cfg.initial_grant), (available.id, cfg.initial_grant)],
    )

    return {"user_id": user_id, "available": available.balance, "locked": locked.balance}


async def materialize(
    db: AsyncSession,
    user_id: int,
    trust_score: float,
    cfg: PolicyConfig,
    now: datetime | None = None,
) -> None:
    """Post any owed regeneration and floor top-up as real ledger entries.

    Called before every balance read, which is why no scheduler is needed.
    """
    now = now or _utcnow()
    available = await get_or_create_account(db, "user_available", user_id)
    mint = await get_or_create_account(db, "platform_mint", None)

    owed = regen_amount(
        available.last_regen_at, now, trust_score, available.balance, cfg
    )
    if owed > 0:
        await post_entry(
            db,
            idempotency_key=f"regen:{user_id}:{now.date().isoformat()}",
            entry_type="regen",
            session_id=None,
            payload={"user_id": user_id, "amount": owed, "trust_score": trust_score},
            lines=[(mint.id, -owed), (available.id, owed)],
        )
    if owed > 0 or available.last_regen_at is None:
        available.last_regen_at = now

    topup = floor_topup_amount(
        available.balance,
        await _has_active_reservation(db, user_id),
        available.last_topup_at,
        now,
        cfg,
    )
    if topup > 0:
        await post_entry(
            db,
            idempotency_key=f"topup:{user_id}:{now.date().isoformat()}",
            entry_type="floor_topup",
            session_id=None,
            payload={"user_id": user_id, "amount": topup},
            lines=[(mint.id, -topup), (available.id, topup)],
        )
        available.last_topup_at = now

    await db.flush()


async def account_summary(
    db: AsyncSession, user_id: int, trust_score: float, cfg: PolicyConfig
) -> dict:
    """Materialize owed credits, then report balances."""
    await materialize(db, user_id, trust_score, cfg)

    available = await get_or_create_account(db, "user_available", user_id)
    locked = await get_or_create_account(db, "user_locked", user_id)

    next_regen_at = (
        available.last_regen_at + timedelta(days=1) if available.last_regen_at else None
    )

    return {
        "user_id": user_id,
        "available": available.balance,
        "locked": locked.balance,
        "regen_rate": regen_rate(trust_score, cfg),
        "next_regen_at": next_regen_at.isoformat() if next_regen_at else None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/resource_agent && python -m pytest tests/test_accounts.py -v`
Expected: PASS — 6 tests

- [ ] **Step 5: Commit**

```bash
git add apps/resource_agent/app/accounts.py apps/resource_agent/tests/test_accounts.py
git commit -m "feat(resource): accounts with grants, lazy regen, participation floor"
```

---

## Task 5: Escrow state machine and API

**Files:**
- Create: `apps/resource_agent/app/escrow.py`
- Create: `apps/resource_agent/app/schemas.py`
- Create: `apps/resource_agent/app/routes.py`
- Create: `apps/resource_agent/tests/test_escrow_api.py`
- Modify: `apps/resource_agent/app/main.py`

**Interfaces:**
- Consumes: `app.accounts.{ensure_accounts, materialize, account_summary}`, `app.ledger.{Movement, post_entry, resolve_movements, get_or_create_account}`, `app.policy.*`, `app.models.{Escrow, Dispute, JournalEntry, LedgerLine, Account}`.
- Produces:
  - `app.escrow`: `InsufficientCredits(Exception)` with `.user_id` and `.shortfall`; `EscrowStateError(Exception)`; `async reserve(db, session_id, participants: list[dict], cfg) -> dict`; `async settle(db, session_id, verdict_type, qa_score, per_user: dict[int, dict], cfg) -> dict`; `async void(db, session_id, reason) -> dict`; `async hold(db, session_id, reason) -> dict`
  - `participants` items are `{"user_id": int, "trust_score": float}`.
  - `per_user` maps `user_id` to `{"quality": float, "engagement": float, "no_show": bool}`.
  - `app.routes.router` mounted at `/resource` in `main.py`.

- [ ] **Step 1: Write the failing tests**

Create `apps/resource_agent/tests/test_escrow_api.py`:

```python
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app
from app.policy import PolicyConfig

CFG = PolicyConfig()


@pytest_asyncio.fixture
async def client(db):
    async def _override():
        yield db

    app.dependency_overrides[get_db] = _override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def _open_accounts(client, user_ids=(1, 2)):
    for user_id in user_ids:
        await client.post("/resource/accounts", json={"user_id": user_id})


RESERVE_BODY = {
    "session_id": 100,
    "participants": [
        {"user_id": 1, "trust_score": 0.30},
        {"user_id": 2, "trust_score": 0.72},
    ],
}


@pytest.mark.asyncio
async def test_create_account_grants_initial_credits(client):
    res = await client.post("/resource/accounts", json={"user_id": 1})

    assert res.status_code == 200
    assert res.json()["available"] == CFG.initial_grant


@pytest.mark.asyncio
async def test_reserve_locks_asymmetric_stakes(client):
    await _open_accounts(client)

    res = await client.post("/resource/escrow/reserve", json=RESERVE_BODY)

    assert res.status_code == 200
    body = res.json()
    stakes = {e["user_id"]: e["amount"] for e in body["escrows"]}
    assert stakes == {1: 28, 2: 11}

    account_1 = (await client.get("/resource/accounts/1")).json()
    assert account_1["available"] == CFG.initial_grant - 28
    assert account_1["locked"] == 28


@pytest.mark.asyncio
async def test_reserve_replay_does_not_double_lock(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    res = await client.post("/resource/escrow/reserve", json=RESERVE_BODY)

    assert res.status_code == 200
    account_1 = (await client.get("/resource/accounts/1")).json()
    assert account_1["locked"] == 28


@pytest.mark.asyncio
async def test_reserve_is_all_or_nothing_when_one_user_is_short(client):
    """Lock up funds across two sessions, then verify a third reservation
    fails cleanly without moving anyone's credits."""
    await _open_accounts(client)

    # trust 0.0 -> maximum 40-credit stake each, against a 100-credit grant.
    def body(session_id):
        return {
            "session_id": session_id,
            "participants": [
                {"user_id": 1, "trust_score": 0.0},
                {"user_id": 2, "trust_score": 0.0},
            ],
        }

    assert (await client.post("/resource/escrow/reserve", json=body(900))).status_code == 200
    assert (await client.post("/resource/escrow/reserve", json=body(901))).status_code == 200
    # each user: available 100 - 80 = 20, locked 80

    res = await client.post("/resource/escrow/reserve", json=body(902))

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "INSUFFICIENT_CREDITS"
    assert res.json()["detail"]["required"] == 40
    assert res.json()["detail"]["available"] == 20

    # Neither user may be touched by the failed reservation.
    for user_id in (1, 2):
        account = (await client.get(f"/resource/accounts/{user_id}")).json()
        assert account["locked"] == 80
        assert account["available"] == 20


@pytest.mark.asyncio
async def test_settle_full_release_returns_stakes_and_pays_bonuses(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)

    res = await client.post(
        "/resource/escrow/settle",
        json={
            "session_id": 100,
            "verdict_type": "SUCCESSFUL",
            "qa_score": 0.92,
            "per_user": {
                "1": {"quality": 0.9, "engagement": 0.8, "no_show": False},
                "2": {"quality": 0.9, "engagement": 0.8, "no_show": False},
            },
        },
    )

    assert res.status_code == 200
    assert res.json()["mode"] == "FULL_RELEASE"

    account_1 = (await client.get("/resource/accounts/1")).json()
    assert account_1["locked"] == 0
    assert account_1["available"] > CFG.initial_grant - 28


@pytest.mark.asyncio
async def test_settle_replay_does_not_double_pay(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    body = {
        "session_id": 100,
        "verdict_type": "SUCCESSFUL",
        "qa_score": 0.92,
        "per_user": {
            "1": {"quality": 0.9, "engagement": 0.8, "no_show": False},
            "2": {"quality": 0.9, "engagement": 0.8, "no_show": False},
        },
    }
    await client.post("/resource/escrow/settle", json=body)
    first = (await client.get("/resource/accounts/1")).json()["available"]
    await client.post("/resource/escrow/settle", json=body)
    second = (await client.get("/resource/accounts/1")).json()["available"]

    assert first == second


@pytest.mark.asyncio
async def test_settling_a_voided_escrow_is_rejected(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    await client.post("/resource/escrow/void", json={"session_id": 100, "reason": "abandoned"})

    res = await client.post(
        "/resource/escrow/settle",
        json={
            "session_id": 100,
            "verdict_type": "SUCCESSFUL",
            "qa_score": 0.92,
            "per_user": {"1": {"quality": 0.9, "engagement": 0.8, "no_show": False}},
        },
    )

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "ESCROW_STATE"


@pytest.mark.asyncio
async def test_void_returns_locked_credits(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    await client.post("/resource/escrow/void", json={"session_id": 100, "reason": "crash"})

    account_1 = (await client.get("/resource/accounts/1")).json()
    assert account_1["available"] == CFG.initial_grant
    assert account_1["locked"] == 0


@pytest.mark.asyncio
async def test_hold_blocks_settlement_until_resolved(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    await client.post("/resource/escrow/100/hold", json={"reason": "toxicity flagged"})

    escrows = (await client.get("/resource/escrow/100")).json()
    assert {e["state"] for e in escrows["escrows"]} == {"HELD"}


@pytest.mark.asyncio
async def test_ledger_lists_entries_for_a_user(client):
    await _open_accounts(client, user_ids=(1,))

    res = await client.get("/resource/ledger/1")

    assert res.status_code == 200
    assert res.json()["entries"][0]["entry_type"] == "grant"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/resource_agent && python -m pytest tests/test_escrow_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.escrow'`

- [ ] **Step 3: Write the escrow state machine**

Create `apps/resource_agent/app/escrow.py`:

```python
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.accounts import ensure_accounts, materialize
from app.ledger import Movement, get_or_create_account, post_entry, resolve_movements
from app.models import Account, Dispute, Escrow
from app.policy import (
    ParticipantOutcome,
    PolicyConfig,
    calculate_escrow,
    plan_settlement,
)

ACTIVE_STATES = ("RESERVED", "HELD")


class InsufficientCredits(Exception):
    def __init__(self, user_id: int, required: int, available: int):
        self.user_id = user_id
        self.required = required
        self.available = available
        self.shortfall = required - available
        super().__init__(
            f"user {user_id} needs {required} credits, has {available}"
        )


class EscrowStateError(Exception):
    def __init__(self, session_id: int, state: str, action: str):
        self.session_id = session_id
        self.state = state
        self.action = action
        super().__init__(f"cannot {action} escrow for session {session_id} in state {state}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _escrows_for(db: AsyncSession, session_id: int) -> list[Escrow]:
    stmt = select(Escrow).where(Escrow.session_id == session_id).order_by(Escrow.user_id)
    return list((await db.execute(stmt)).scalars().all())


async def reserve(
    db: AsyncSession, session_id: int, participants: list[dict], cfg: PolicyConfig
) -> dict:
    """Lock every participant's stake in one transaction, or lock nothing.

    Accounts are locked in user_id order so concurrent sessions cannot deadlock.
    """
    existing = await _escrows_for(db, session_id)
    if existing:
        return {
            "session_id": session_id,
            "escrows": [_escrow_dict(e) for e in existing],
            "created": False,
        }

    ordered = sorted(participants, key=lambda p: p["user_id"])

    stakes: dict[int, int] = {}
    for participant in ordered:
        user_id = participant["user_id"]
        await ensure_accounts(db, user_id, cfg)
        await materialize(db, user_id, participant["trust_score"], cfg)
        stakes[user_id] = calculate_escrow(participant["trust_score"], cfg)

    # Check every participant before moving any credits. Accounts are fetched
    # FOR UPDATE in user_id order, which is what makes concurrent sessions
    # deadlock-free.
    for user_id, stake in stakes.items():
        account = await get_or_create_account(db, "user_available", user_id)
        locked_account = await db.get(Account, account.id, with_for_update=True)
        if locked_account.balance < stake:
            raise InsufficientCredits(user_id, stake, locked_account.balance)

    movements: list[Movement] = []
    for user_id, stake in stakes.items():
        movements.append(Movement(f"user_available:{user_id}", -stake))
        movements.append(Movement(f"user_locked:{user_id}", stake))

    lines = await resolve_movements(db, movements)
    entry, _ = await post_entry(
        db,
        idempotency_key=f"reserve:{session_id}",
        entry_type="escrow_reserve",
        session_id=session_id,
        payload={"stakes": {str(k): v for k, v in stakes.items()}},
        lines=lines,
    )

    escrows = []
    for user_id, stake in stakes.items():
        escrow = Escrow(
            session_id=session_id,
            user_id=user_id,
            amount=stake,
            state="RESERVED",
            reserve_entry_id=entry.id,
        )
        db.add(escrow)
        escrows.append(escrow)
    await db.flush()

    return {
        "session_id": session_id,
        "escrows": [_escrow_dict(e) for e in escrows],
        "created": True,
    }


async def settle(
    db: AsyncSession,
    session_id: int,
    verdict_type: str,
    qa_score: float,
    per_user: dict[int, dict],
    cfg: PolicyConfig,
) -> dict:
    escrows = await _escrows_for(db, session_id)
    if not escrows:
        raise EscrowStateError(session_id, "MISSING", "settle")

    states = {e.state for e in escrows}
    if states == {"SETTLED"}:
        entry_ids = [e.settle_entry_id for e in escrows if e.settle_entry_id]
        return {
            "session_id": session_id,
            "mode": "ALREADY_SETTLED",
            "breakdown": {},
            "entry_id": entry_ids[0] if entry_ids else None,
        }
    if not states <= {"RESERVED"}:
        raise EscrowStateError(session_id, ", ".join(sorted(states)), "settle")

    participants = [
        ParticipantOutcome(
            user_id=escrow.user_id,
            stake=escrow.amount,
            quality=float(per_user.get(escrow.user_id, {}).get("quality", 0.0)),
            engagement=float(per_user.get(escrow.user_id, {}).get("engagement", 0.0)),
            no_show=bool(per_user.get(escrow.user_id, {}).get("no_show", False)),
        )
        for escrow in escrows
    ]

    plan = plan_settlement(verdict_type, qa_score, participants, cfg)
    lines = await resolve_movements(db, plan.movements)
    entry, _ = await post_entry(
        db,
        idempotency_key=f"settle:{session_id}",
        entry_type="escrow_settle",
        session_id=session_id,
        payload={
            "verdict_type": verdict_type,
            "qa_score": qa_score,
            "mode": plan.mode,
            "breakdown": {str(k): v for k, v in plan.breakdown.items()},
        },
        lines=lines,
    )

    now = _utcnow()
    for escrow in escrows:
        escrow.state = "SETTLED"
        escrow.settle_entry_id = entry.id
        escrow.settled_at = now
    await db.flush()

    return {
        "session_id": session_id,
        "mode": plan.mode,
        "breakdown": plan.breakdown,
        "entry_id": entry.id,
    }


async def void(db: AsyncSession, session_id: int, reason: str) -> dict:
    escrows = await _escrows_for(db, session_id)
    if not escrows:
        raise EscrowStateError(session_id, "MISSING", "void")

    states = {e.state for e in escrows}
    if states == {"VOIDED"}:
        return {"session_id": session_id, "voided": False, "reason": reason}
    if not states <= {"RESERVED", "HELD"}:
        raise EscrowStateError(session_id, ", ".join(sorted(states)), "void")

    movements: list[Movement] = []
    for escrow in escrows:
        movements.append(Movement(f"user_locked:{escrow.user_id}", -escrow.amount))
        movements.append(Movement(f"user_available:{escrow.user_id}", escrow.amount))

    lines = await resolve_movements(db, movements)
    entry, _ = await post_entry(
        db,
        idempotency_key=f"void:{session_id}",
        entry_type="escrow_void",
        session_id=session_id,
        payload={"reason": reason},
        lines=lines,
    )

    now = _utcnow()
    for escrow in escrows:
        escrow.state = "VOIDED"
        escrow.settle_entry_id = entry.id
        escrow.settled_at = now
    await db.flush()

    return {"session_id": session_id, "voided": True, "reason": reason}


async def hold(db: AsyncSession, session_id: int, reason: str) -> dict:
    escrows = await _escrows_for(db, session_id)
    if not escrows:
        raise EscrowStateError(session_id, "MISSING", "hold")

    states = {e.state for e in escrows}
    if states == {"HELD"}:
        return {"session_id": session_id, "held": False, "reason": reason}
    if not states <= {"RESERVED"}:
        raise EscrowStateError(session_id, ", ".join(sorted(states)), "hold")

    for escrow in escrows:
        escrow.state = "HELD"

    db.add(Dispute(session_id=session_id, reason=reason, state="OPEN"))
    await db.flush()

    return {"session_id": session_id, "held": True, "reason": reason}


def _escrow_dict(escrow: Escrow) -> dict:
    return {
        "session_id": escrow.session_id,
        "user_id": escrow.user_id,
        "amount": escrow.amount,
        "state": escrow.state,
        "reserved_at": escrow.reserved_at.isoformat() if escrow.reserved_at else None,
        "settled_at": escrow.settled_at.isoformat() if escrow.settled_at else None,
    }
```

- [ ] **Step 4: Write the schemas**

Create `apps/resource_agent/app/schemas.py`:

```python
from pydantic import BaseModel, Field


class AccountCreateRequest(BaseModel):
    user_id: int


class ReserveParticipant(BaseModel):
    user_id: int
    trust_score: float = Field(ge=0.0, le=1.0)


class ReserveRequest(BaseModel):
    session_id: int
    participants: list[ReserveParticipant]


class ParticipantMetrics(BaseModel):
    quality: float = 0.0
    engagement: float = 0.0
    no_show: bool = False


class SettleRequest(BaseModel):
    session_id: int
    verdict_type: str
    qa_score: float
    per_user: dict[str, ParticipantMetrics] = {}


class VoidRequest(BaseModel):
    session_id: int
    reason: str


class HoldRequest(BaseModel):
    reason: str


class DisputeResolveRequest(BaseModel):
    resolution: str  # "settle" | "void"
    note: str = ""
    verdict_type: str = "PARTIAL"
    qa_score: float = 0.5
    resolved_by: str = "admin"
```

- [ ] **Step 5: Write the routes**

Create `apps/resource_agent/app/routes.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import escrow as escrow_service
from app.accounts import account_summary, ensure_accounts
from app.database import get_db
from app.models import Dispute, JournalEntry, LedgerLine, Account
from app.policy import PolicyConfig
from app.schemas import (
    AccountCreateRequest,
    DisputeResolveRequest,
    HoldRequest,
    ReserveRequest,
    SettleRequest,
    VoidRequest,
)

router = APIRouter(prefix="/resource")
cfg = PolicyConfig.from_env()


@router.post("/accounts")
async def create_account(req: AccountCreateRequest, db: AsyncSession = Depends(get_db)):
    summary = await ensure_accounts(db, req.user_id, cfg)
    await db.commit()
    return summary


@router.get("/accounts/{user_id}")
async def get_account(
    user_id: int,
    trust_score: float = Query(default=0.5, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
):
    summary = await account_summary(db, user_id, trust_score, cfg)
    await db.commit()
    return summary


@router.post("/escrow/reserve")
async def reserve_escrow(req: ReserveRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = await escrow_service.reserve(
            db, req.session_id, [p.model_dump() for p in req.participants], cfg
        )
    except escrow_service.InsufficientCredits as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "INSUFFICIENT_CREDITS",
                "user_id": exc.user_id,
                "required": exc.required,
                "available": exc.available,
                "shortfall": exc.shortfall,
            },
        )
    await db.commit()
    return result


@router.post("/escrow/settle")
async def settle_escrow(req: SettleRequest, db: AsyncSession = Depends(get_db)):
    per_user = {int(k): v.model_dump() for k, v in req.per_user.items()}
    try:
        result = await escrow_service.settle(
            db, req.session_id, req.verdict_type, req.qa_score, per_user, cfg
        )
    except escrow_service.EscrowStateError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "ESCROW_STATE", "state": exc.state, "action": exc.action},
        )
    await db.commit()
    return result


@router.post("/escrow/void")
async def void_escrow(req: VoidRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = await escrow_service.void(db, req.session_id, req.reason)
    except escrow_service.EscrowStateError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "ESCROW_STATE", "state": exc.state, "action": exc.action},
        )
    await db.commit()
    return result


@router.post("/escrow/{session_id}/hold")
async def hold_escrow(
    session_id: int, req: HoldRequest, db: AsyncSession = Depends(get_db)
):
    try:
        result = await escrow_service.hold(db, session_id, req.reason)
    except escrow_service.EscrowStateError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "ESCROW_STATE", "state": exc.state, "action": exc.action},
        )
    await db.commit()
    return result


@router.get("/escrow/{session_id}")
async def get_escrow(session_id: int, db: AsyncSession = Depends(get_db)):
    escrows = await escrow_service._escrows_for(db, session_id)
    return {
        "session_id": session_id,
        "escrows": [escrow_service._escrow_dict(e) for e in escrows],
    }


@router.get("/ledger/{user_id}")
async def get_ledger(
    user_id: int,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    account_ids = (
        await db.execute(select(Account.id).where(Account.user_id == user_id))
    ).scalars().all()
    if not account_ids:
        return {"user_id": user_id, "entries": [], "total": 0}

    stmt = (
        select(JournalEntry, LedgerLine.amount)
        .join(LedgerLine, LedgerLine.entry_id == JournalEntry.id)
        .where(LedgerLine.account_id.in_(account_ids))
        .order_by(JournalEntry.id.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await db.execute(stmt)).all()

    return {
        "user_id": user_id,
        "entries": [
            {
                "id": entry.id,
                "entry_type": entry.entry_type,
                "session_id": entry.session_id,
                "amount": amount,
                "payload": entry.payload,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
            }
            for entry, amount in rows
        ],
        "total": len(rows),
    }


@router.post("/disputes/{session_id}/resolve")
async def resolve_dispute(
    session_id: int, req: DisputeResolveRequest, db: AsyncSession = Depends(get_db)
):
    from datetime import datetime, timezone

    dispute = (
        await db.execute(
            select(Dispute).where(Dispute.session_id == session_id, Dispute.state == "OPEN")
        )
    ).scalar_one_or_none()
    if not dispute:
        raise HTTPException(status_code=404, detail="No open dispute for this session")

    escrows = await escrow_service._escrows_for(db, session_id)
    for item in escrows:
        item.state = "RESERVED"
    await db.flush()

    if req.resolution == "settle":
        result = await escrow_service.settle(
            db, session_id, req.verdict_type, req.qa_score, {}, cfg
        )
    elif req.resolution == "void":
        result = await escrow_service.void(db, session_id, f"dispute: {req.note}")
    else:
        raise HTTPException(status_code=400, detail="resolution must be 'settle' or 'void'")

    dispute.state = "RESOLVED"
    dispute.resolution = req.resolution
    dispute.resolved_by = req.resolved_by
    dispute.resolved_at = datetime.now(timezone.utc)
    await db.commit()

    return {"session_id": session_id, "resolution": req.resolution, "result": result}
```

- [ ] **Step 6: Mount the router**

In `apps/resource_agent/app/main.py`, add this import next to the existing ones:

```python
from app.routes import router
```

And add this line immediately after the `app.add_middleware(...)` block:

```python
app.include_router(router)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd apps/resource_agent && python -m pytest tests/ -v`
Expected: PASS — all tests from Tasks 1–5

- [ ] **Step 8: Commit**

```bash
git add apps/resource_agent
git commit -m "feat(resource): escrow state machine, reserve/settle/void/hold API"
```

---

## Task 6: Reconciler

**Files:**
- Create: `apps/resource_agent/app/reconciler.py`
- Create: `apps/resource_agent/tests/test_reconciler.py`
- Modify: `apps/resource_agent/app/main.py`
- Modify: `apps/resource_agent/app/routes.py`

**Interfaces:**
- Consumes: `app.escrow.void`, `app.ledger.balance_of`, `app.models.{Account, Escrow, JournalEntry, LedgerLine}`, `app.policy.PolicyConfig`.
- Produces:
  - `async reconcile_once(db, cfg, now: datetime | None = None) -> dict` with keys `voided_sessions: list[int]`, `balance_drift: list[dict]`, `unbalanced_entries: list[int]`, `checked_at: str`
  - `async reconciler_loop(interval_seconds: int = 300) -> None`
  - `LAST_REPORT: dict` module-level, read by `/resource/health`

- [ ] **Step 1: Write the failing tests**

Create `apps/resource_agent/tests/test_reconciler.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from app.accounts import ensure_accounts
from app.escrow import reserve
from app.ledger import get_or_create_account
from app.models import Escrow, LedgerLine
from app.policy import PolicyConfig
from app.reconciler import reconcile_once

CFG = PolicyConfig()
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)

PARTICIPANTS = [
    {"user_id": 1, "trust_score": 0.30},
    {"user_id": 2, "trust_score": 0.72},
]


@pytest.mark.asyncio
async def test_reconciler_voids_stale_reservations(db):
    await reserve(db, 100, PARTICIPANTS, CFG)
    await db.commit()

    stale = NOW + timedelta(hours=CFG.reserve_ttl_hours + 1)
    report = await reconcile_once(db, CFG, now=stale)
    await db.commit()

    assert 100 in report["voided_sessions"]

    account = await get_or_create_account(db, "user_available", 1)
    await db.refresh(account)
    assert account.balance == CFG.initial_grant


@pytest.mark.asyncio
async def test_reconciler_leaves_fresh_reservations_alone(db):
    await reserve(db, 101, PARTICIPANTS, CFG)
    await db.commit()

    report = await reconcile_once(db, CFG)
    await db.commit()

    assert report["voided_sessions"] == []


@pytest.mark.asyncio
async def test_reconciler_skips_held_escrows(db):
    await reserve(db, 102, PARTICIPANTS, CFG)
    for item in (await db.execute(__import__("sqlalchemy").select(Escrow))).scalars():
        item.state = "HELD"
    await db.commit()

    stale = NOW + timedelta(days=30)
    report = await reconcile_once(db, CFG, now=stale)
    await db.commit()

    assert report["voided_sessions"] == []


@pytest.mark.asyncio
async def test_reconciler_detects_balance_drift_without_repairing_it(db):
    await ensure_accounts(db, 1, CFG)
    account = await get_or_create_account(db, "user_available", 1)
    account.balance += 500  # corrupt the cache, leave the ledger alone
    await db.commit()

    report = await reconcile_once(db, CFG)
    await db.commit()

    drift = [d for d in report["balance_drift"] if d["account_id"] == account.id]
    assert len(drift) == 1
    assert drift[0]["cached"] == CFG.initial_grant + 500
    assert drift[0]["derived"] == CFG.initial_grant

    await db.refresh(account)
    assert account.balance == CFG.initial_grant + 500  # never silently repaired


@pytest.mark.asyncio
async def test_reconciler_detects_unbalanced_entries(db):
    await ensure_accounts(db, 1, CFG)
    await db.commit()

    account = await get_or_create_account(db, "user_available", 1)
    db.add(LedgerLine(entry_id=1, account_id=account.id, amount=7))
    await db.commit()

    report = await reconcile_once(db, CFG)

    assert 1 in report["unbalanced_entries"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/resource_agent && python -m pytest tests/test_reconciler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.reconciler'`

- [ ] **Step 3: Write the reconciler**

Create `apps/resource_agent/app/reconciler.py`:

```python
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.escrow import void
from app.ledger import balance_of
from app.models import Account, Escrow, LedgerLine
from app.policy import PolicyConfig

logger = logging.getLogger("resource-agent.reconciler")

LAST_REPORT: dict = {
    "checked_at": None,
    "voided_sessions": [],
    "balance_drift": [],
    "unbalanced_entries": [],
}


async def reconcile_once(
    db: AsyncSession, cfg: PolicyConfig, now: datetime | None = None
) -> dict:
    """One reconciliation pass.

    Drift is reported, never repaired — a mismatch means a bug, and silently
    correcting it would hide the bug.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=cfg.reserve_ttl_hours)

    stale_sessions = (
        await db.execute(
            select(Escrow.session_id)
            .where(Escrow.state == "RESERVED", Escrow.reserved_at < cutoff)
            .distinct()
        )
    ).scalars().all()

    voided: list[int] = []
    for session_id in stale_sessions:
        await void(db, session_id, "reconciler: reservation expired")
        voided.append(session_id)
        logger.warning("Voided stale reservation for session %s", session_id)

    drift: list[dict] = []
    accounts = (await db.execute(select(Account))).scalars().all()
    for account in accounts:
        derived = await balance_of(db, account.id)
        if derived != account.balance:
            row = {
                "account_id": account.id,
                "kind": account.kind,
                "user_id": account.user_id,
                "cached": account.balance,
                "derived": derived,
            }
            drift.append(row)
            logger.error("Balance drift on account %s: %s", account.id, row)

    unbalanced = (
        await db.execute(
            select(LedgerLine.entry_id)
            .group_by(LedgerLine.entry_id)
            .having(func.sum(LedgerLine.amount) != 0)
        )
    ).scalars().all()
    for entry_id in unbalanced:
        logger.error("Unbalanced journal entry %s", entry_id)

    report = {
        "checked_at": now.isoformat(),
        "voided_sessions": voided,
        "balance_drift": drift,
        "unbalanced_entries": list(unbalanced),
    }
    LAST_REPORT.update(report)
    return report


async def reconciler_loop(interval_seconds: int = 300) -> None:
    from app.database import async_session

    cfg = PolicyConfig.from_env()
    while True:
        try:
            async with async_session() as db:
                await reconcile_once(db, cfg)
                await db.commit()
        except Exception:
            logger.exception("Reconciliation pass failed")
        await asyncio.sleep(interval_seconds)
```

- [ ] **Step 4: Start the loop and report on health**

In `apps/resource_agent/app/main.py`, replace the `lifespan` function body's `yield` line so the function reads:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    logger.info("Resource Agent · Port 8004 · PostgreSQL")

    from app.database import async_session, init_db
    from app.ledger import SYSTEM_KINDS, get_or_create_account
    from app.reconciler import reconciler_loop

    await init_db()

    async with async_session() as db:
        for kind in SYSTEM_KINDS:
            await get_or_create_account(db, kind, None)
        await db.commit()

    logger.info("Database ready — tables created, system accounts seeded")

    task = asyncio.create_task(reconciler_loop())
    logger.info("Reconciler started (every 300s)")
    try:
        yield
    finally:
        task.cancel()
```

In `apps/resource_agent/app/main.py`, replace the `health` handler with:

```python
@app.get("/resource/health")
async def health():
    from app.reconciler import LAST_REPORT

    healthy = not LAST_REPORT["balance_drift"] and not LAST_REPORT["unbalanced_entries"]
    return {
        "service": "resource-agent",
        "status": "ok" if healthy else "degraded",
        "last_reconcile": LAST_REPORT,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd apps/resource_agent && python -m pytest tests/ -v`
Expected: PASS — all tests including the 5 new reconciler tests

- [ ] **Step 6: Commit**

```bash
git add apps/resource_agent
git commit -m "feat(resource): reconciler for stale reservations and ledger drift"
```

---

## Task 7: Backend cutover

This is the only destructive task. The old credit path dies here.

**Files:**
- Create: `apps/backend/app/clients/__init__.py`
- Create: `apps/backend/app/clients/resource.py`
- Delete: `apps/backend/app/escrow.py`
- Modify: `apps/backend/app/config.py`
- Modify: `apps/backend/app/models.py:110-150`
- Modify: `apps/backend/app/schemas.py`
- Modify: `apps/backend/app/routes.py:180-318`, `:779-895`, `:906-912`
- Modify: `apps/backend/app/main.py`
- Modify: `apps/frontend/nginx.conf`
- Modify: `tests/conftest.py`
- Modify: `tests/test_e2e_session_lifecycle.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: the `/resource/*` API from Tasks 4–6.
- Produces: `app.clients.resource.ResourceClient` with `async ensure_account(user_id) -> dict`, `async get_account(user_id, trust_score) -> dict`, `async reserve(session_id, participants) -> dict`, `async settle(session_id, verdict_type, qa_score, per_user) -> dict`, `async void(session_id, reason) -> dict`, `async get_escrow(session_id) -> dict`, `async get_ledger(user_id) -> dict`; and `app.clients.resource.InsufficientCredits(Exception)` with `.user_id`, `.required`, `.available`, `.shortfall`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_e2e_session_lifecycle.py` (append at end of file):

```python
@pytest.mark.asyncio
async def test_session_start_locks_real_escrow(backend_client, monkeypatch):
    """Backend must delegate escrow to resource_agent, not compute it locally."""
    calls = {}

    async def fake_reserve(self, session_id, participants):
        calls["session_id"] = session_id
        calls["participants"] = participants
        return {
            "session_id": session_id,
            "escrows": [
                {"user_id": p["user_id"], "amount": 28, "state": "RESERVED"}
                for p in participants
            ],
            "created": True,
        }

    from app.clients.resource import ResourceClient

    monkeypatch.setattr(ResourceClient, "reserve", fake_reserve)

    create = await backend_client.post(
        "/session/create",
        json={
            "teacher_user_id": 1,
            "learner_user_id": 2,
            "topic": "Python basics",
            "duration_minutes": 30,
        },
    )
    barter_id = create.json()["barter_id"]

    res = await backend_client.post(f"/session/{barter_id}/start")

    assert res.status_code == 200
    assert calls["session_id"] == barter_id
    assert {p["user_id"] for p in calls["participants"]} == {1, 2}
    assert all("trust_score" in p for p in calls["participants"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && python -m pytest test_e2e_session_lifecycle.py::test_session_start_locks_real_escrow -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.clients'`

- [ ] **Step 3: Write the resource client**

Create `apps/backend/app/clients/__init__.py` (empty file).

Create `apps/backend/app/clients/resource.py`:

```python
"""HTTP client for the Resource Agent.

Backend owns reputation; resource_agent owns credits. Every call here is
idempotent on the resource_agent side, so retries are safe.
"""

import httpx

from app.config import settings

TIMEOUT = 10.0


class ResourceUnavailable(Exception):
    """resource_agent could not be reached."""


class InsufficientCredits(Exception):
    def __init__(self, user_id: int, required: int, available: int, shortfall: int):
        self.user_id = user_id
        self.required = required
        self.available = available
        self.shortfall = shortfall
        super().__init__(f"user {user_id} short by {shortfall} credits")


class ResourceClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.RESOURCE_URL).rstrip("/")

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                res = await client.post(f"{self.base_url}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise ResourceUnavailable(str(exc)) from exc

        if res.status_code == 409:
            detail = res.json().get("detail", {})
            if detail.get("code") == "INSUFFICIENT_CREDITS":
                raise InsufficientCredits(
                    detail["user_id"],
                    detail["required"],
                    detail["available"],
                    detail["shortfall"],
                )
        res.raise_for_status()
        return res.json()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                res = await client.get(f"{self.base_url}{path}", params=params or {})
        except httpx.HTTPError as exc:
            raise ResourceUnavailable(str(exc)) from exc
        res.raise_for_status()
        return res.json()

    async def ensure_account(self, user_id: int) -> dict:
        return await self._post("/resource/accounts", {"user_id": user_id})

    async def get_account(self, user_id: int, trust_score: float = 0.5) -> dict:
        return await self._get(
            f"/resource/accounts/{user_id}", {"trust_score": trust_score}
        )

    async def reserve(self, session_id: int, participants: list[dict]) -> dict:
        return await self._post(
            "/resource/escrow/reserve",
            {"session_id": session_id, "participants": participants},
        )

    async def settle(
        self, session_id: int, verdict_type: str, qa_score: float, per_user: dict
    ) -> dict:
        return await self._post(
            "/resource/escrow/settle",
            {
                "session_id": session_id,
                "verdict_type": verdict_type,
                "qa_score": qa_score,
                "per_user": {str(k): v for k, v in per_user.items()},
            },
        )

    async def void(self, session_id: int, reason: str) -> dict:
        return await self._post(
            "/resource/escrow/void", {"session_id": session_id, "reason": reason}
        )

    async def get_escrow(self, session_id: int) -> dict:
        return await self._get(f"/resource/escrow/{session_id}")

    async def get_ledger(self, user_id: int) -> dict:
        return await self._get(f"/resource/ledger/{user_id}")


resource_client = ResourceClient()
```

- [ ] **Step 4: Add RESOURCE_URL to backend config**

In `apps/backend/app/config.py`, add this line inside `Settings`, after `WARNING_URL`:

```python
    RESOURCE_URL: str = "http://resource_agent:8004"
```

- [ ] **Step 5: Rewire session start**

In `apps/backend/app/routes.py`, replace the whole `start_session` function (lines 180–240) with:

```python
@router.post("/session/{barter_id}/start")
async def start_session(barter_id: int, db: AsyncSession = Depends(get_db)):
    from app.clients.resource import (
        InsufficientCredits,
        ResourceUnavailable,
        resource_client,
    )

    result = await db.execute(select(BarterSession).where(BarterSession.id == barter_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status in ("completed", "terminated"):
        raise HTTPException(status_code=400, detail=f"Session already {session.status}")
    if session.status == "active":
        return {
            "barter_id": session.id,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "status": session.status,
        }

    contract_result = await db.execute(
        select(SessionContract).where(SessionContract.barter_session_id == barter_id)
    )
    contract = contract_result.scalar_one_or_none()
    if not contract:
        raise HTTPException(status_code=404, detail="Session contract not found")

    user_ids = [contract.teacher_user_id, contract.learner_user_id]
    users = (
        await db.execute(select(User).where(User.id.in_(user_ids)))
    ).scalars().all()
    trust_by_id = {u.id: u.trust_score for u in users}

    participants = [
        {"user_id": uid, "trust_score": float(trust_by_id.get(uid, 0.5))}
        for uid in user_ids
    ]

    try:
        reservation = await resource_client.reserve(barter_id, participants)
    except InsufficientCredits as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Insufficient credits: user {exc.user_id} needs {exc.required} "
                f"but has {exc.available} (short by {exc.shortfall})"
            ),
        )
    except ResourceUnavailable as exc:
        raise HTTPException(
            status_code=503, detail=f"Resource Agent unavailable: {exc}"
        )

    session.status = "active"
    session.started_at = datetime.now(timezone.utc)
    await db.commit()

    stakes = {e["user_id"]: e["amount"] for e in reservation["escrows"]}
    _ok(
        f"Session {session.id} started — escrow locked  "
        f"teacher={stakes.get(contract.teacher_user_id)}cr  "
        f"learner={stakes.get(contract.learner_user_id)}cr"
    )

    return {
        "barter_id": session.id,
        "started_at": session.started_at.isoformat(),
        "status": session.status,
        "escrows": reservation["escrows"],
    }
```

If `User` is not already imported at the top of `routes.py`, add it to the existing `from app.models import ...` line.

- [ ] **Step 6: Rewire session confirm**

In `apps/backend/app/routes.py`, inside `confirm_session`, replace the import line:

```python
    from app.escrow import apply_settlement
```

with:

```python
    from app.clients.resource import ResourceUnavailable, resource_client
```

And replace the settlement block (the `qa_score = (...)` expression through `settlement_result = await apply_settlement(db, barter_id, qa_score)`) with:

```python
        qa_score = (
            1.0
            if verdict and verdict.verdict_type == "SUCCESSFUL"
            else 0.5
            if verdict and verdict.verdict_type == "PARTIAL"
            else 0.0
        )
        verdict_type = verdict.verdict_type if verdict else "DISPUTE"

        engagement = 0.0
        if verdict and verdict.drift_summary:
            try:
                engagement = float(
                    json.loads(verdict.drift_summary).get("engagement", {}).get("score", 0.0)
                )
            except (ValueError, AttributeError):
                engagement = 0.0

        quality = (verdict.on_topic_percentage or 0.0) / 100.0 if verdict else 0.0
        per_user = {
            uid: {"quality": quality, "engagement": engagement, "no_show": False}
            for uid in confirmed_users
        }

        try:
            settlement_result = await resource_client.settle(
                barter_id, verdict_type, qa_score, per_user
            )
        except ResourceUnavailable as exc:
            settlement_result = {"error": f"Resource Agent unavailable: {exc}"}
```

- [ ] **Step 7: Replace the credit read routes**

In `apps/backend/app/routes.py`, delete the `/wallet/{user_id}`, `/escrow/lock`, `/escrow/{barter_id}`, `/escrow/release`, `/settlement/{barter_id}`, and `/transactions/{user_id}` handlers (lines 779–895 and 906–912). Replace them with these thin pass-throughs, so the frontend has one origin during migration:

```python
@router.get("/wallet/{user_id}")
async def get_wallet(user_id: int, db: AsyncSession = Depends(get_db)):
    from app.clients.resource import ResourceUnavailable, resource_client

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    trust = float(user.trust_score) if user else 0.5
    try:
        return await resource_client.get_account(user_id, trust)
    except ResourceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/escrow/{barter_id}")
async def get_escrow(barter_id: int):
    from app.clients.resource import ResourceUnavailable, resource_client

    try:
        return await resource_client.get_escrow(barter_id)
    except ResourceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/transactions/{user_id}")
async def get_transactions(user_id: int):
    from app.clients.resource import ResourceUnavailable, resource_client

    try:
        return await resource_client.get_ledger(user_id)
    except ResourceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
```

- [ ] **Step 8: Delete the old credit code**

```bash
git rm apps/backend/app/escrow.py
```

In `apps/backend/app/models.py`, delete the `Wallet`, `Escrow`, and `CreditTransaction` classes (lines 110–150).

In `apps/backend/app/schemas.py`, delete `WalletResponse`, `EscrowResponse`, `EscrowLockRequest`, `EscrowReleaseRequest`, `SettlementRequest`, `SettlementResponse`, and `CreditTransactionResponse`.

In `apps/backend/app/routes.py`, remove any now-unused imports of those names from the `from app.models import ...` and `from app.schemas import ...` lines.

- [ ] **Step 9: Ensure resource accounts for seeded users**

In `apps/backend/app/main.py`, inside `lifespan`, immediately after the `await db.commit()` that follows the user seeding loop, add:

```python
    from app.clients.resource import ResourceUnavailable, resource_client

    for uid in (1, 2):
        try:
            await resource_client.ensure_account(uid)
        except ResourceUnavailable as exc:
            _warn(f"Resource Agent unreachable, account {uid} not provisioned ({exc})")
    _ok("Resource accounts provisioned")
```

- [ ] **Step 10: Wire compose and nginx**

In `docker-compose.yml`, add to the `backend` service's `environment:` block:

```yaml
      RESOURCE_URL: http://resource_agent:8004
```

And add `resource_agent` to the `backend` service's `depends_on:` list.

In `apps/frontend/nginx.conf`, add this location block before the existing backend regex block:

```nginx
    location /resource/ {
        proxy_pass http://resource_agent:8004;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `cd tests && python -m pytest -v`
Expected: PASS — the whole existing e2e suite plus the new start-session test

- [ ] **Step 12: Verify the full stack**

Run: `docker compose up --build -d && sleep 30 && curl -sk https://localhost/resource/health`
Expected: `{"service":"resource-agent","status":"ok",...}`

- [ ] **Step 13: Commit**

```bash
git add -A apps/backend apps/frontend/nginx.conf docker-compose.yml tests
git commit -m "refactor(backend): delegate all credit state to resource_agent"
```

---

## Task 8: Frontend — balance, escrow, settlement, ledger

**Files:**
- Create: `apps/frontend/src/api/resource.js`
- Create: `apps/frontend/src/components/BalanceWidget.jsx`
- Create: `apps/frontend/src/components/EscrowPanel.jsx`
- Create: `apps/frontend/src/components/SettlementBreakdown.jsx`
- Create: `apps/frontend/src/screens/Ledger.jsx`
- Modify: `apps/frontend/src/screens/Setup.jsx:23-24`
- Modify: `apps/frontend/src/screens/PostSession.jsx:46`
- Modify: `apps/frontend/src/App.jsx:101-102`

**Interfaces:**
- Consumes: `/resource/*` via nginx (Task 7 added the proxy).
- Produces: `getAccount(userId, trustScore)`, `getEscrow(sessionId)`, `getLedger(userId)` from `src/api/resource.js`; four React components.

- [ ] **Step 1: Write the API module**

Create `apps/frontend/src/api/resource.js`:

```javascript
const RESOURCE = '/resource'

async function json(res) {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export async function getAccount(userId, trustScore = 0.5) {
  return json(await fetch(`${RESOURCE}/accounts/${userId}?trust_score=${trustScore}`))
}

export async function getEscrow(sessionId) {
  return json(await fetch(`${RESOURCE}/escrow/${sessionId}`))
}

export async function getLedger(userId, limit = 50, offset = 0) {
  return json(await fetch(`${RESOURCE}/ledger/${userId}?limit=${limit}&offset=${offset}`))
}
```

- [ ] **Step 2: Write the balance widget**

Create `apps/frontend/src/components/BalanceWidget.jsx`:

```jsx
import { useEffect, useState } from 'react'
import { getAccount } from '../api/resource'

export default function BalanceWidget({ userId, label, trustScore = 0.5 }) {
  const [account, setAccount] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    getAccount(userId, trustScore)
      .then(data => { if (!cancelled) setAccount(data) })
      .catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [userId, trustScore])

  if (error) return <div className="text-xs text-red-400">wallet unavailable</div>
  if (!account) return <div className="text-xs text-slate-400">…</div>

  return (
    <div className="flex items-center gap-3 rounded-lg bg-slate-800 px-3 py-2">
      <span className="text-xs font-medium text-slate-300">{label}</span>
      <span className="text-sm font-semibold text-emerald-400">
        {account.available} cr
      </span>
      {account.locked > 0 && (
        <span className="text-xs text-amber-400">{account.locked} locked</span>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Write the escrow panel**

Create `apps/frontend/src/components/EscrowPanel.jsx`:

```jsx
export default function EscrowPanel({ escrows, names = {} }) {
  if (!escrows || escrows.length === 0) return null

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Escrow locked</h3>
      <p className="mb-3 text-xs text-slate-400">
        Stake scales inversely with trust — less trust means more skin in the game.
      </p>
      <ul className="space-y-1">
        {escrows.map(item => (
          <li key={item.user_id} className="flex justify-between text-sm">
            <span className="text-slate-300">
              {names[item.user_id] || `User ${item.user_id}`}
            </span>
            <span className="font-medium text-amber-400">{item.amount} cr</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
```

- [ ] **Step 4: Write the settlement breakdown**

Create `apps/frontend/src/components/SettlementBreakdown.jsx`:

```jsx
const ROWS = [
  ['stake_returned', 'Stake returned'],
  ['teaching_bonus', 'Teaching bonus'],
  ['engagement_bonus', 'Engagement bonus'],
  ['compensation', 'Compensation'],
  ['penalty', 'Penalty'],
  ['fee_share', 'Platform fee'],
]

export default function SettlementBreakdown({ settlement, names = {} }) {
  if (!settlement || !settlement.breakdown) return null

  const negative = new Set(['penalty', 'fee_share'])

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
      <h3 className="mb-1 text-sm font-semibold text-slate-200">Settlement</h3>
      <p className="mb-3 text-xs text-slate-400">{settlement.mode}</p>

      {Object.entries(settlement.breakdown).map(([userId, row]) => (
        <div key={userId} className="mb-4">
          <div className="mb-1 text-xs font-medium text-slate-300">
            {names[userId] || `User ${userId}`}
          </div>
          <table className="w-full text-sm">
            <tbody>
              {ROWS.filter(([key]) => row[key]).map(([key, label]) => (
                <tr key={key}>
                  <td className="py-0.5 text-slate-400">{label}</td>
                  <td className="py-0.5 text-right text-slate-200">
                    {negative.has(key) ? '−' : '+'}{row[key]} cr
                  </td>
                </tr>
              ))}
              <tr className="border-t border-slate-700">
                <td className="py-1 font-medium text-slate-300">Net</td>
                <td
                  className={`py-1 text-right font-semibold ${
                    row.net >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {row.net >= 0 ? '+' : ''}{row.net} cr
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 5: Write the ledger screen**

Create `apps/frontend/src/screens/Ledger.jsx`:

```jsx
import { useEffect, useState } from 'react'
import { getLedger } from '../api/resource'

const LABELS = {
  grant: 'Initial grant',
  regen: 'Regeneration',
  floor_topup: 'Floor top-up',
  escrow_reserve: 'Escrow locked',
  escrow_settle: 'Settlement',
  escrow_void: 'Escrow returned',
}

export default function Ledger({ userId, onBack }) {
  const [entries, setEntries] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    getLedger(userId)
      .then(data => setEntries(data.entries))
      .catch(err => setError(err.message))
  }, [userId])

  if (error) return <div className="p-6 text-red-400">Ledger unavailable: {error}</div>

  return (
    <div className="mx-auto max-w-2xl p-6">
      <button onClick={onBack} className="mb-4 text-sm text-slate-400 hover:text-slate-200">
        ← Back
      </button>
      <h2 className="mb-4 text-lg font-semibold text-slate-100">
        Credit history · User {userId}
      </h2>

      {entries.length === 0 && <p className="text-sm text-slate-400">No transactions yet.</p>}

      <ul className="divide-y divide-slate-800">
        {entries.map(entry => (
          <li key={`${entry.id}-${entry.amount}`} className="flex items-center justify-between py-2">
            <div>
              <div className="text-sm text-slate-200">
                {LABELS[entry.entry_type] || entry.entry_type}
              </div>
              <div className="text-xs text-slate-500">
                {entry.session_id ? `Session ${entry.session_id} · ` : ''}
                {new Date(entry.created_at).toLocaleString()}
              </div>
            </div>
            <span
              className={`text-sm font-medium ${
                entry.amount >= 0 ? 'text-emerald-400' : 'text-red-400'
              }`}
            >
              {entry.amount >= 0 ? '+' : ''}{entry.amount} cr
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
```

- [ ] **Step 6: Point existing screens at the resource API**

In `apps/frontend/src/screens/Setup.jsx`, replace lines 23–24:

```javascript
      fetch(`${API}/wallet/1`).then(r => r.ok ? r.json() : null),
      fetch(`${API}/wallet/2`).then(r => r.ok ? r.json() : null),
```

with:

```javascript
      getAccount(1).catch(() => null),
      getAccount(2).catch(() => null),
```

and add this import at the top of the file:

```javascript
import { getAccount } from '../api/resource'
```

In `apps/frontend/src/App.jsx`, replace lines 101–102:

```javascript
        fetch('/wallet/1').then(r => r.ok ? r.json() : null).catch(() => null),
        fetch('/wallet/2').then(r => r.ok ? r.json() : null).catch(() => null),
```

with:

```javascript
        getAccount(1).catch(() => null),
        getAccount(2).catch(() => null),
```

and add this import at the top of the file:

```javascript
import { getAccount } from './api/resource'
```

In `apps/frontend/src/screens/PostSession.jsx`, replace line 46:

```javascript
        fetch(`${API}/escrow/${barterId}`),
```

with a call through the resource API — add this import at the top:

```javascript
import { getEscrow } from '../api/resource'
import SettlementBreakdown from '../components/SettlementBreakdown'
```

and replace that array element with:

```javascript
        getEscrow(barterId).then(d => ({ ok: true, json: async () => d })),
```

Then render `<SettlementBreakdown settlement={settlement} />` in the results area, where `settlement` comes from the `/session/{id}/confirm` response already held in that screen's state.

In `apps/frontend/src/screens/LiveSession.jsx`, after the successful `start` call on line 226, store `res.json()`'s `escrows` array in state and render `<EscrowPanel escrows={escrows} />`, importing it with:

```javascript
import EscrowPanel from '../components/EscrowPanel'
```

- [ ] **Step 7: Build the frontend to verify it compiles**

Run: `cd apps/frontend && npm run build`
Expected: build succeeds with no unresolved imports

- [ ] **Step 8: Verify in the running stack**

Run: `docker compose up --build -d frontend backend resource_agent resource_db && sleep 20 && curl -sk https://localhost/resource/accounts/1`
Expected: JSON with `available`, `locked`, `regen_rate`

- [ ] **Step 9: Commit**

```bash
git add apps/frontend
git commit -m "feat(frontend): wallet, escrow, settlement, and ledger views"
```

---

## Task 9: Documentation refresh

**Files:**
- Modify: `CLAUDE.md`
- Modify: `apps/resource_agent/README.md` (create)

**Interfaces:**
- Consumes: everything built in Tasks 1–8.
- Produces: updated repo documentation. The design spec and the vault decision log were written before implementation and need no changes.

- [ ] **Step 1: Update the repo guide**

In `CLAUDE.md`, update the architecture block to include the new service:

```
apps/
├── backend/           # Port 8000 - FastAPI + SQLite
├── audio_pipeline/    # Port 8001 - AWS Transcribe STT
├── semantic_analysis/ # Port 8002 - Sentence-BERT
├── warning_engine/    # Port 8003 - Escalation logic
├── resource_agent/    # Port 8004 - Credits, escrow, ledger (PostgreSQL)
└── frontend/          # nginx (ports 80→443 redirect, 443 for SPA+proxies)
```

In the same file, replace the "Database" section with:

```markdown
## Databases

Two databases, split by ownership:

- **backend** — SQLite (`barter.db`): users, sessions, contracts, transcripts,
  window results, warnings, verdicts, confirmations.
- **resource_agent** — PostgreSQL (`resource_db`): accounts, journal entries,
  ledger lines, escrows, disputes. Escrow reservation needs row-level locking
  and multi-row atomicity, which SQLite's single writer cannot provide without
  serializing the whole application.

Backend holds no credit tables. All credit state is reached over HTTP.
```

In the same file, add `| /resource/ | resource_agent:8004 |` to the nginx routing table, and add a Resource Agent entry to the Service Communication list.

- [ ] **Step 2: Write the service README**

Create `apps/resource_agent/README.md`:

```markdown
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
cd apps/resource_agent && python -m pytest tests/ -v
```

Tests reach Postgres on `localhost:5433`. Override with `DATABASE_URL`.
```

- [ ] **Step 3: Verify the whole suite one more time**

Run: `cd apps/resource_agent && python -m pytest tests/ -v && cd ../../tests && python -m pytest -v`
Expected: PASS — both suites

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md apps/resource_agent/README.md
git commit -m "docs: document resource_agent service and split databases"
```

---

## Self-Review Notes

**Spec coverage:** §3 architecture → Tasks 1, 7. §4 data model → Task 2. §5 policy → Tasks 3, 4. §6 API → Task 5. §7 failure handling and reconciler → Tasks 5, 6. §8 cutover → Task 7. §9 testing → all tasks (layers 1–5). §10 build order → task ordering. §11 frontend → Task 8.

**Deviation from spec §10:** the spec's build order lists disputes as step 8, after cutover. This plan folds dispute holds into Task 5 (state machine) and admin resolution into Task 5's routes, because `HELD` is part of the state machine and splitting it would leave Task 5's state transitions untestable. Nothing is dropped.

**Spec ambiguity resolved in this plan:** spec §5 says "PENALTY: the stake is forfeited to the counterparty," which is unambiguous when one party no-showed but undefined for a mutual dispute where neither did. This plan makes forfeiture conditional on `no_show`:

- One party no-showed → their stake goes to the counterparty.
- Both no-showed → the whole pool goes to `platform_revenue`.
- Neither no-showed (mutual dispute) → stakes are returned, but no bonuses are paid and no fee is charged. Net zero for both.

Two tests in Task 3 pin this down: `test_mutual_dispute_returns_stakes_but_pays_no_bonuses` and `test_mutual_no_show_forfeits_the_whole_pool_to_the_platform`. If you want a mutual dispute to burn credits instead, that is a policy change in `_mode_for`/`plan_settlement`, not a bug fix — update the spec first.

**Known sharp edge:** `test_reserve_is_all_or_nothing_when_one_user_is_short` (Task 5, Step 1) reserves with `trust_score: 0.0` to force a 40-credit stake against the 100-credit grant, twice, to strand the third reservation. If `INITIAL_GRANT` or `BASE_ESCROW` changes, this test's arithmetic must change with it.
