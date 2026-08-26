import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.escrow import EscrowStateError, _escrows_for, void
from app.ledger import balance_of
from app.models import Account, Escrow, LedgerLine
from app.policy import PolicyConfig

logger = logging.getLogger("resource-agent.reconciler")

# The reconciler runs on this cadence in production (see main.py's
# lifespan). Health considers the last successful pass stale once it is
# older than 3x that interval -- tolerant of one or two slow/missed cycles,
# but not of a reconciler that has silently stopped succeeding altogether.
DEFAULT_INTERVAL_SECONDS = 300
STALE_AFTER_SECONDS = DEFAULT_INTERVAL_SECONDS * 3

LAST_REPORT: dict = {
    "checked_at": None,
    "voided_sessions": [],
    "balance_drift": [],
    "unbalanced_entries": [],
    # Bookkeeping distinct from the report fields above: these track the
    # health of the reconciler *process* itself (has it run recently? is it
    # currently failing?), not just the content of its last successful scan.
    "last_attempt_at": None,
    "last_success_at": None,
    "last_error": None,
    "consecutive_failures": 0,
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
        # Re-check state immediately before voiding, under the same lock
        # void() itself takes. The scan above only filters on
        # Escrow.state == "RESERVED" at scan time; a session can be picked
        # up by an admin (hold()) between that scan and this call. void()'s
        # own guard legitimately accepts HELD (HELD -> VOIDED is a real
        # admin-resolution edge for its other callers), so that guard can't
        # protect a held session from the reconciler specifically -- this
        # stricter check belongs here, not in void(). Locking first (rather
        # than just re-reading .state) is what actually closes the race:
        # it serializes against a concurrent hold() on the same rows instead
        # of merely narrowing the window.
        escrows = await _escrows_for(db, session_id, for_update=True)
        states = {e.state for e in escrows}
        if states != {"RESERVED"}:
            logger.warning(
                "Reconciler skipping session %s: state is %s, not purely "
                "RESERVED, at void time",
                session_id,
                ", ".join(sorted(states)) or "MISSING",
            )
            continue

        try:
            await void(db, session_id, "reconciler: reservation expired")
        except EscrowStateError:
            # Belt-and-braces: the lock+recheck above should make this
            # unreachable, but if it ever fires it's still just one
            # anticipated race on one session, not a reason to abandon the
            # rest of the pass -- the remaining sessions and the drift /
            # balance checks below still need to run. This is an expected
            # race, not a surprise, so no traceback is warranted.
            logger.error(
                "Reconciler could not void session %s (unexpected state)", session_id
            )
            continue
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


async def run_reconcile_cycle(cfg: PolicyConfig) -> dict:
    """Open a session, run one reconcile_once pass, commit, and record
    attempt/success/failure bookkeeping in LAST_REPORT regardless of outcome.

    Split out from reconciler_loop so a single cycle -- including a failing
    one -- can be driven directly in tests without spinning the infinite
    loop. Re-raises on failure after recording it, so reconciler_loop's own
    try/except is what decides whether the process keeps going (it does).
    """
    from app.database import async_session

    attempted_at = datetime.now(timezone.utc).isoformat()
    LAST_REPORT["last_attempt_at"] = attempted_at
    try:
        async with async_session() as db:
            report = await reconcile_once(db, cfg)
            await db.commit()
    except Exception as exc:
        LAST_REPORT["last_error"] = repr(exc)
        LAST_REPORT["consecutive_failures"] = LAST_REPORT.get("consecutive_failures", 0) + 1
        raise
    else:
        LAST_REPORT["last_success_at"] = attempted_at
        LAST_REPORT["last_error"] = None
        LAST_REPORT["consecutive_failures"] = 0
        return report


def is_healthy(report: dict) -> bool:
    """Whether `/resource/health` should report "ok" given a LAST_REPORT
    snapshot.

    Unhealthy if: the last scan found drift or unbalanced entries, the last
    attempt raised, or the last successful pass is older than
    STALE_AFTER_SECONDS. A report that has never run (last_success_at is
    None and no error has ever been recorded -- e.g. right after process
    start, before the first pass completes) is treated as healthy: there is
    nothing yet to report as broken.
    """
    if report["balance_drift"] or report["unbalanced_entries"]:
        return False
    if report.get("last_error"):
        return False
    last_success_at = report.get("last_success_at")
    if last_success_at is None:
        return True
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(last_success_at)).total_seconds()
    return age <= STALE_AFTER_SECONDS


async def reconciler_loop(interval_seconds: int = 300) -> None:
    cfg = PolicyConfig.from_env()
    while True:
        try:
            await run_reconcile_cycle(cfg)
        except Exception:
            logger.exception("Reconciliation pass failed")
        await asyncio.sleep(interval_seconds)
