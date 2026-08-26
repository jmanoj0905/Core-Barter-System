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
        locked_account = (
            await db.execute(
                select(Account)
                .where(Account.id == account.id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        ).scalar_one()
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
