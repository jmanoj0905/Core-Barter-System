from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ledger import get_or_create_account, post_entry
from app.models import Escrow
from app.policy import PolicyConfig, floor_topup_amount, regen_amount, regen_rate

ACTIVE_ESCROW_STATES = ("RESERVED", "HELD")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _has_active_reservation(db: AsyncSession, user_id: int) -> bool:
    stmt = select(Escrow.id).where(
        Escrow.user_id == user_id, Escrow.state.in_(ACTIVE_ESCROW_STATES)
    )
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
    # Lock the row before reading last_regen_at/balance so a concurrent call
    # for the same user can't read the same pre-update clock and post twice
    # under two different day-scoped idempotency keys (e.g. straddling UTC
    # midnight). Acquire the lock before anything below could autoflush a
    # pending insert against this row — same ordering rule as app.ledger.
    #
    # A plain re-`select(...).with_for_update()` is not enough here: since
    # `available` is already in the session's identity map from
    # `get_or_create_account`, SQLAlchemy returns that same cached object
    # without overwriting its attributes from the freshly-locked row, so the
    # lock would serialize the write but the decision below would still be
    # made from stale data. `Session.refresh(..., with_for_update=...)`
    # re-fetches under the lock *and* repopulates the instance in place,
    # which is what makes the post-lock values actually visible here.
    await db.refresh(available, with_for_update=True)
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
