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
async def test_post_entry_repopulates_a_preloaded_account_before_locking(db, session_factory):
    """Two sessions, each pre-loading src/dst via get_or_create_account
    BEFORE calling post_entry — mirroring exactly what resolve_movements()
    does on the real path (Task 5's reserve/settle both go through it) —
    then posting distinct entries to the same dst account concurrently.

    The pre-load is the ingredient that makes this bite: it puts dst into
    each session's own identity map with the account's balance at the time
    of the pre-load (before either session's post_entry has done anything).
    post_entry's own `select(...).with_for_update()` then targets that same
    already-mapped row. Without `execution_options(populate_existing=True)`,
    SQLAlchemy hands back the identity-mapped Python object as-is instead of
    overwriting its attributes from the row it just locked — so the second
    session to acquire the lock computes `account.balance += amount` off its
    own pre-lock balance, not the first session's already-committed result,
    and its commit overwrites (loses) that update on the cached column.

    Unlike test_concurrent_distinct_entries_on_same_account_lose_no_update
    above (10 writers), this uses exactly 2 concurrent sessions so both
    pre-loads reliably land before either session reaches the lock — with
    10 writers sharing a bounded connection pool, later writers tend to
    start after earlier ones have already committed, which happens to read
    fresh data regardless of the bug and doesn't reproduce it reliably.

    Asserts both `balance_of()` (derived from ledger_lines, unaffected by
    this bug — both LedgerLine rows are always inserted correctly) and the
    cached `Account.balance` column equal the true total: the whole point
    of the fix is that these two must not diverge.

    A `asyncio.Barrier` forces both sessions to finish their pre-load before
    either is allowed to call post_entry — relying on asyncio's scheduling
    to interleave the two pre-loads on its own is not reliable here (unlike
    materialize's single-extra-await shape, post_entry has several awaits
    between where a caller would pre-load and its own FOR UPDATE select —
    the idempotency-key check, a SAVEPOINT, and the JournalEntry insert —
    which is enough real-world timing slack for one session to occasionally
    run to completion before the other even starts its pre-load, silently
    reading fresh data and masking the bug). The barrier removes that
    timing dependency so the test is deterministic.
    """
    src, dst = await _two_accounts(db)
    await db.commit()
    src_id, dst_id = src.id, dst.id
    barrier = asyncio.Barrier(2)

    async def attempt(key, amount):
        async with session_factory() as session:
            # Pre-load both accounts into this session's identity map before
            # calling post_entry, exactly like resolve_movements() does, and
            # keep a live reference to the preloaded destination account for
            # the duration of the call — exactly like resolve_movements()
            # does when it holds `account` locally to read `.id` before
            # building the lines list. Without a live reference here, CPython
            # refcounting can collect the just-loaded object as soon as it
            # goes out of scope, which drops it from the session's (weak)
            # identity map and makes post_entry's own SELECT a fresh, correct
            # load — masking the bug rather than exercising it.
            await get_or_create_account(session, "platform_mint", None)
            preloaded_dst = await get_or_create_account(session, "user_available", 1)
            await barrier.wait()  # guarantee both pre-loads land before either posts
            await post_entry(
                session, idempotency_key=key, entry_type="grant",
                session_id=None, payload={}, lines=[(src_id, -amount), (dst_id, amount)],
            )
            await session.commit()
            assert preloaded_dst.id == dst_id  # keep the reference alive till here

    await asyncio.gather(
        attempt("grant:preload:1", 10),
        attempt("grant:preload:2", 15),
    )

    await db.refresh(dst)
    assert await balance_of(db, dst_id) == 25
    assert dst.balance == 25


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
