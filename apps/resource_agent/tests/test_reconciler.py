from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

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

    # Derived from real wall-clock time, not the frozen NOW constant: NOW is
    # a fixed date (2026-08-26), but the escrow's real reserved_at is set by
    # the database clock when this test actually runs. Anchoring the cutoff
    # to NOW would make this reservation "fresh" (and the test fail) the
    # moment real time passes NOW + reserve_ttl_hours + 1.
    stale = datetime.now(timezone.utc) + timedelta(hours=CFG.reserve_ttl_hours + 1)
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
    for item in (await db.execute(select(Escrow))).scalars():
        item.state = "HELD"
    await db.commit()

    # Real wall-clock anchored, same reasoning as above -- safe by a wide
    # margin either way, but there's no reason to leave a latent dependency
    # on the frozen NOW constant when it costs nothing to avoid it.
    stale = datetime.now(timezone.utc) + timedelta(days=30)
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
