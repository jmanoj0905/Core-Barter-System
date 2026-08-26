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
