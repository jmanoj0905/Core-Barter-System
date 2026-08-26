from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import app.escrow as escrow_service
import app.reconciler as reconciler_module
from app.accounts import ensure_accounts
from app.escrow import reserve
from app.ledger import get_or_create_account
from app.models import Escrow, LedgerLine
from app.policy import PolicyConfig
from app.reconciler import reconcile_once, run_reconcile_cycle

CFG = PolicyConfig()

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


@pytest.mark.asyncio
async def test_reconciler_never_voids_a_session_that_becomes_held_mid_pass(
    db, session_factory
):
    """The stale-session scan only filters on Escrow.state == "RESERVED" at
    scan time. If an admin picks the session up for dispute review (hold())
    after that scan but before the reconciler gets to void() it, relying on
    the query filter alone would still void it: void()'s own ACTIVE_STATES
    legitimately accepts HELD (that's the real HELD -> VOIDED admin-resolve
    edge for its other callers), so it doesn't stop the reconciler either.
    The reconciler has to re-check -- and lock -- state itself, immediately
    before voiding, or this race silently destroys an admin's hold.
    """
    await reserve(db, 200, PARTICIPANTS, CFG)
    await db.commit()

    stale = datetime.now(timezone.utc) + timedelta(hours=CFG.reserve_ttl_hours + 1)

    original_execute = db.execute
    call_count = 0

    async def execute_and_race(statement, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = await original_execute(statement, *args, **kwargs)
        if call_count == 1:
            # reconcile_once's very first query against `db` is the
            # stale-session scan. Simulate an admin picking up the session
            # in a separate, already-committed transaction right after that
            # scan returns but before the reconciler acts on session 200.
            async with session_factory() as admin_session:
                await escrow_service.hold(admin_session, 200, "admin picked up dispute")
                await admin_session.commit()
        return result

    db.execute = execute_and_race
    try:
        report = await reconcile_once(db, CFG, now=stale)
        await db.commit()
    finally:
        db.execute = original_execute

    assert 200 not in report["voided_sessions"]

    escrows = await escrow_service._escrows_for(db, 200)
    assert {e.state for e in escrows} == {"HELD"}


@pytest.mark.asyncio
async def test_run_reconcile_cycle_records_a_successful_pass(db):
    report = await run_reconcile_cycle(CFG)

    last_report = reconciler_module.LAST_REPORT
    assert last_report["checked_at"] == report["checked_at"]
    assert last_report["last_attempt_at"] is not None
    assert last_report["last_success_at"] == last_report["last_attempt_at"]
    assert last_report["last_error"] is None
    assert last_report["consecutive_failures"] == 0
    assert reconciler_module.is_healthy(last_report) is True


@pytest.mark.asyncio
async def test_run_reconcile_cycle_records_failure_without_erasing_previous_success(
    db, monkeypatch
):
    """A reconcile pass that raises must leave health reporting the failure,
    not silently keep showing the previous success -- a stale "ok" is worse
    than no answer, because it actively reassures.
    """
    await run_reconcile_cycle(CFG)
    prior_success_at = reconciler_module.LAST_REPORT["last_success_at"]
    prior_checked_at = reconciler_module.LAST_REPORT["checked_at"]

    async def boom(db, cfg, now=None):
        raise RuntimeError("simulated reconcile failure")

    monkeypatch.setattr(reconciler_module, "reconcile_once", boom)

    with pytest.raises(RuntimeError):
        await run_reconcile_cycle(CFG)

    last_report = reconciler_module.LAST_REPORT
    assert "simulated reconcile failure" in last_report["last_error"]
    assert last_report["consecutive_failures"] == 1
    # The prior success is preserved -- not overwritten -- by the failed
    # attempt; only the failure/attempt bookkeeping changes.
    assert last_report["last_success_at"] == prior_success_at
    assert last_report["checked_at"] == prior_checked_at
    assert reconciler_module.is_healthy(last_report) is False


def test_is_healthy_treats_a_never_run_report_as_healthy():
    """Right after process start, before the first reconcile pass completes,
    there is nothing yet to report as broken."""
    fresh = {
        "checked_at": None,
        "voided_sessions": [],
        "balance_drift": [],
        "unbalanced_entries": [],
        "last_attempt_at": None,
        "last_success_at": None,
        "last_error": None,
        "consecutive_failures": 0,
    }
    assert reconciler_module.is_healthy(fresh) is True


def test_is_healthy_is_false_once_the_last_success_goes_stale():
    stale_success = (
        datetime.now(timezone.utc)
        - timedelta(seconds=reconciler_module.STALE_AFTER_SECONDS + 1)
    ).isoformat()
    report = {
        "checked_at": stale_success,
        "voided_sessions": [],
        "balance_drift": [],
        "unbalanced_entries": [],
        "last_attempt_at": stale_success,
        "last_success_at": stale_success,
        "last_error": None,
        "consecutive_failures": 0,
    }
    assert reconciler_module.is_healthy(report) is False
