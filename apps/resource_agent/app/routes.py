from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import escrow as escrow_service
from app.accounts import account_summary, ensure_accounts
from app.database import get_db
from app.models import Dispute, JournalEntry, LedgerLine, Account
from app.policy import PolicyConfig
from app.schemas import (
    AccountCreateRequest,
    DisputeResolveRequest,
    HoldRequest,
    ReserveRequest,
    SettleRequest,
    VoidRequest,
)

router = APIRouter(prefix="/resource")
cfg = PolicyConfig.from_env()


@router.post("/accounts")
async def create_account(req: AccountCreateRequest, db: AsyncSession = Depends(get_db)):
    summary = await ensure_accounts(db, req.user_id, cfg)
    await db.commit()
    return summary


@router.get("/accounts/{user_id}")
async def get_account(
    user_id: int,
    trust_score: float = Query(default=0.5, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
):
    summary = await account_summary(db, user_id, trust_score, cfg)
    await db.commit()
    return summary


@router.post("/escrow/reserve")
async def reserve_escrow(req: ReserveRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = await escrow_service.reserve(
            db, req.session_id, [p.model_dump() for p in req.participants], cfg
        )
    except escrow_service.InsufficientCredits as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "INSUFFICIENT_CREDITS",
                "user_id": exc.user_id,
                "required": exc.required,
                "available": exc.available,
                "shortfall": exc.shortfall,
            },
        )
    await db.commit()
    return result


@router.post("/escrow/settle")
async def settle_escrow(req: SettleRequest, db: AsyncSession = Depends(get_db)):
    per_user = {int(k): v.model_dump() for k, v in req.per_user.items()}
    try:
        result = await escrow_service.settle(
            db, req.session_id, req.verdict_type, req.qa_score, per_user, cfg
        )
    except escrow_service.EscrowStateError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "ESCROW_STATE", "state": exc.state, "action": exc.action},
        )
    await db.commit()
    return result


@router.post("/escrow/void")
async def void_escrow(req: VoidRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = await escrow_service.void(db, req.session_id, req.reason)
    except escrow_service.EscrowStateError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "ESCROW_STATE", "state": exc.state, "action": exc.action},
        )
    await db.commit()
    return result


@router.post("/escrow/{session_id}/hold")
async def hold_escrow(
    session_id: int, req: HoldRequest, db: AsyncSession = Depends(get_db)
):
    try:
        result = await escrow_service.hold(db, session_id, req.reason)
    except escrow_service.EscrowStateError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "ESCROW_STATE", "state": exc.state, "action": exc.action},
        )
    await db.commit()
    return result


@router.get("/escrow/{session_id}")
async def get_escrow(session_id: int, db: AsyncSession = Depends(get_db)):
    escrows = await escrow_service._escrows_for(db, session_id)
    return {
        "session_id": session_id,
        "escrows": [escrow_service._escrow_dict(e) for e in escrows],
    }


@router.get("/ledger/{user_id}")
async def get_ledger(
    user_id: int,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    account_ids = (
        await db.execute(select(Account.id).where(Account.user_id == user_id))
    ).scalars().all()
    if not account_ids:
        return {"user_id": user_id, "entries": [], "total": 0}

    stmt = (
        select(JournalEntry, LedgerLine.amount)
        .join(LedgerLine, LedgerLine.entry_id == JournalEntry.id)
        .where(LedgerLine.account_id.in_(account_ids))
        .order_by(JournalEntry.id.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await db.execute(stmt)).all()

    return {
        "user_id": user_id,
        "entries": [
            {
                "id": entry.id,
                "entry_type": entry.entry_type,
                "session_id": entry.session_id,
                "amount": amount,
                "payload": entry.payload,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
            }
            for entry, amount in rows
        ],
        "total": len(rows),
    }


@router.post("/disputes/{session_id}/resolve")
async def resolve_dispute(
    session_id: int, req: DisputeResolveRequest, db: AsyncSession = Depends(get_db)
):
    from datetime import datetime, timezone

    dispute = (
        await db.execute(
            select(Dispute).where(Dispute.session_id == session_id, Dispute.state == "OPEN")
        )
    ).scalar_one_or_none()
    if not dispute:
        raise HTTPException(status_code=404, detail="No open dispute for this session")

    escrows = await escrow_service._escrows_for(db, session_id)
    for item in escrows:
        item.state = "RESERVED"
    await db.flush()

    if req.resolution == "settle":
        result = await escrow_service.settle(
            db, session_id, req.verdict_type, req.qa_score, {}, cfg
        )
    elif req.resolution == "void":
        result = await escrow_service.void(db, session_id, f"dispute: {req.note}")
    else:
        raise HTTPException(status_code=400, detail="resolution must be 'settle' or 'void'")

    dispute.state = "RESOLVED"
    dispute.resolution = req.resolution
    dispute.resolved_by = req.resolved_by
    dispute.resolved_at = datetime.now(timezone.utc)
    await db.commit()

    return {"session_id": session_id, "resolution": req.resolution, "result": result}
