import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.escrow import EscrowStateError, void
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
        try:
            await void(db, session_id, "reconciler: reservation expired")
        except EscrowStateError:
            # A concurrent settle/hold/void could have moved this session out
            # of a voidable state between the scan above and this call. That
            # is one bad (or merely raced) session, not a reason to abandon
            # the rest of the pass — the remaining sessions and the drift /
            # balance checks below still need to run.
            logger.exception(
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
