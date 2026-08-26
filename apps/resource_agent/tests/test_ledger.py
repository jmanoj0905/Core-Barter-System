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


@pytest.mark.asyncio
async def test_concurrent_distinct_entries_on_same_account_lose_no_update(db, session_factory):
    src, dst = await _two_accounts(db)
    await db.commit()
    src_id, dst_id = src.id, dst.id

    async def attempt(key):
        async with session_factory() as session:
            # Load the accounts into this session's identity map first, the
            # same way resolve_movements()/get_or_create_account() would
            # before handing lines to post_entry on the same session. This is
            # the path that must still take a real row lock, not an
            # identity-map hit that skips FOR UPDATE.
            await get_or_create_account(session, "platform_mint", None)
            await get_or_create_account(session, "user_available", 1)
            _, created = await post_entry(
                session, idempotency_key=key, entry_type="grant",
                session_id=None, payload={}, lines=[(src_id, -10), (dst_id, 10)],
            )
            await session.commit()
            return created

    keys = [f"grant:concurrent:{i}" for i in range(10)]
    results = await asyncio.gather(*(attempt(key) for key in keys))

    assert results == [True] * len(keys)
    await db.refresh(dst)
    expected = 10 * len(keys)
    assert await balance_of(db, dst_id) == expected
    assert dst.balance == expected
