import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from sqlalchemy import select

from app.accounts import account_summary, ensure_accounts, materialize
from app.ledger import balance_of, get_or_create_account, post_entry
from app.models import Escrow, JournalEntry
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
async def test_materialize_does_not_top_up_during_a_held_escrow(db):
    await ensure_accounts(db, 1, CFG)
    account = await get_or_create_account(db, "user_available", 1)
    mint = await get_or_create_account(db, "platform_mint", None)
    await post_entry(
        db, idempotency_key="spend:4", entry_type="penalty", session_id=None,
        payload={}, lines=[(account.id, -100), (mint.id, 100)],
    )
    account.last_regen_at = NOW
    db.add(Escrow(session_id=98, user_id=1, amount=5, state="HELD"))
    await db.commit()

    await materialize(db, 1, trust_score=0.5, cfg=CFG, now=NOW)
    await db.commit()
    await db.refresh(account)

    assert account.balance == 0


@pytest.mark.asyncio
async def test_materialize_is_idempotent_for_the_same_instant(db):
    await ensure_accounts(db, 1, CFG)
    account = await get_or_create_account(db, "user_available", 1)
    mint = await get_or_create_account(db, "platform_mint", None)
    await post_entry(
        db, idempotency_key="spend:5", entry_type="penalty", session_id=None,
        payload={}, lines=[(account.id, -90), (mint.id, 90)],
    )
    account.last_regen_at = NOW - timedelta(days=2)
    await db.commit()

    await materialize(db, 1, trust_score=0.5, cfg=CFG, now=NOW)
    await db.commit()
    await db.refresh(account)
    balance_after_first = account.balance
    last_regen_after_first = account.last_regen_at

    await materialize(db, 1, trust_score=0.5, cfg=CFG, now=NOW)
    await db.commit()
    await db.refresh(account)

    assert account.balance == balance_after_first
    assert account.last_regen_at == last_regen_after_first

    entries = (
        await db.execute(
            select(JournalEntry).where(JournalEntry.entry_type == "regen")
        )
    ).scalars().all()
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_materialize_same_instant_calls_post_the_regen_entry_once(db, session_factory):
    """Two independent sessions calling materialize for the same user with the
    same `now` and regen genuinely owed. This pins that `post_entry`'s
    idempotency-key uniqueness alone is enough to stop a duplicate post when
    both calls land on the same day-scoped key (`regen:{user_id}:{iso_day}`).

    This is NOT a test of the row-lock/populate_existing race fix: with same
    `now`, both calls derive the identical idempotency key regardless of
    whether the read they used to decide to post was fresh or stale, so the
    ledger's unique-key dedup masks that class of bug entirely. It passes
    whether or not the lock in materialize repopulates the account row after
    locking. See test_materialize_is_race_safe_across_midnight_straddle below
    for a test that actually exercises that fix.
    """
    await ensure_accounts(db, 1, CFG)
    account = await get_or_create_account(db, "user_available", 1)
    mint = await get_or_create_account(db, "platform_mint", None)
    await post_entry(
        db, idempotency_key="spend:6", entry_type="penalty", session_id=None,
        payload={}, lines=[(account.id, -90), (mint.id, 90)],
    )
    account.last_regen_at = NOW - timedelta(days=1)
    await db.commit()

    async def attempt():
        async with session_factory() as session:
            await materialize(session, 1, trust_score=0.5, cfg=CFG, now=NOW)
            await session.commit()

    await asyncio.gather(attempt(), attempt())

    await db.refresh(account)
    assert account.balance == 15  # 10 remaining + exactly one day's regen (5)

    entries = (
        await db.execute(
            select(JournalEntry).where(JournalEntry.entry_type == "regen")
        )
    ).scalars().all()
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_materialize_is_race_safe_across_midnight_straddle(db, session_factory):
    """Two independent sessions calling materialize for the same user with
    `now` values 2 seconds apart that straddle UTC midnight, so they derive
    two DIFFERENT day-scoped idempotency keys (`regen:{user_id}:{iso_day}`)
    even though, correctly, only one day's regen is owed in total. This is
    the shape the ledger's per-key uniqueness cannot protect against — the
    two posts are legitimately different keys, not a replay.

    last_regen_at is set exactly one day before NOW_A, so the first call to
    settle sees exactly 1 elapsed day (owed = 5, at trust_score=0.5's rate of
    5/day) and advances last_regen_at to its own `now`. The second call,
    only 2 seconds later by wall clock, must see that update and compute 0
    elapsed days — not rederive 1 more elapsed day from a stale pre-lock
    read of the original last_regen_at.

    Without `db.refresh(available, with_for_update=True)` repopulating the
    locked row's attributes (i.e. with the round-1 `select(...)
    .with_for_update()` shape that leaves the identity-map copy stale), the
    second call still sees the original last_regen_at, computes 1 whole
    elapsed day relative to it, and posts a second 5-credit regen entry
    under its own (different) day key — minting a full extra day the user
    never earned: 2 regen entries instead of 1, and a true ledger total of
    20 credits (10 remaining + two days' worth of regen) instead of the
    correct 15 (10 remaining + one day's regen) — see the fix report.

    The credited-amount assertion below reads `balance_of()` (a fresh sum
    over `ledger_lines`), not the cached `Account.balance` column. The
    cached column turns out to be an unreliable witness for this specific
    regression: `post_entry` locks each line's account via its own
    `select(...).with_for_update()` (app/ledger.py), which has the exact
    same un-repopulated-identity-map hazard this fix addresses. When the
    round-1 (buggy) code is in place, that hazard also strikes inside the
    two racing `post_entry` calls for this same account, and the resulting
    lost update on `Account.balance` happens to converge on the same wrong
    total (15) as the correct answer — masking the double-mint even though
    two real 5-credit ledger entries were posted (verified empirically: see
    fix report). `balance_of()` reads the actual `ledger_lines` rows, which
    both post_entry calls always insert regardless of that column-update
    race, so it correctly shows 20 under the bug and 15 under the fix.
    """
    now_a = datetime(2026, 8, 26, 23, 59, 59, tzinfo=timezone.utc)
    now_b = now_a + timedelta(seconds=2)  # 2026-08-27T00:00:01Z — next UTC day
    assert now_a.date() != now_b.date()

    await ensure_accounts(db, 1, CFG)
    account = await get_or_create_account(db, "user_available", 1)
    mint = await get_or_create_account(db, "platform_mint", None)
    await post_entry(
        db, idempotency_key="spend:7", entry_type="penalty", session_id=None,
        payload={}, lines=[(account.id, -90), (mint.id, 90)],
    )
    account.last_regen_at = now_a - timedelta(days=1)
    await db.commit()

    async def attempt(now):
        async with session_factory() as session:
            await materialize(session, 1, trust_score=0.5, cfg=CFG, now=now)
            await session.commit()

    await asyncio.gather(attempt(now_a), attempt(now_b))

    # 10 remaining + exactly one day's regen (5), not two days (10) minted
    # across both racing calls. See balance_of() note in the docstring for
    # why this reads the ledger lines directly rather than account.balance.
    assert await balance_of(db, account.id) == 15

    entries = (
        await db.execute(
            select(JournalEntry).where(JournalEntry.entry_type == "regen")
        )
    ).scalars().all()
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_account_summary_reports_available_and_locked(db):
    await ensure_accounts(db, 1, CFG)
    await db.commit()

    summary = await account_summary(db, 1, trust_score=0.9, cfg=CFG)

    assert summary["user_id"] == 1
    assert summary["available"] == CFG.initial_grant
    assert summary["locked"] == 0
    assert summary["regen_rate"] == 20
