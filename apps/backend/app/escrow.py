from datetime import datetime
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from .models import Wallet, Escrow, CreditTransaction, User, BarterSession, SessionContract


ESCROW_CONFIG = {
    "base_escrow": 40,
    "min_escrow": 5,
    "teaching_bonus": 5,
    "engagement_bonus": 2,
    "no_show_penalty": 20,
}


def calculate_escrow_amount(trust_score: float) -> int:
    escrow = int(ESCROW_CONFIG["base_escrow"] * (1 - trust_score))
    return max(ESCROW_CONFIG["min_escrow"], escrow)


async def get_or_create_wallet(db: AsyncSession, user_id: int) -> Wallet:
    result = await db.execute(select(Wallet).where(Wallet.user_id == user_id))
    wallet = result.scalar_one_or_none()

    if not wallet:
        wallet = Wallet(
            user_id=user_id,
            available_balance=999999,
            locked_balance=0,
            total_earned=0,
            total_spent=0,
        )
        db.add(wallet)
        await db.flush()

        transaction = CreditTransaction(
            user_id=user_id,
            barter_session_id=None,
            transaction_type="initial_allocation",
            amount=999999,
            balance_after=999999,
            description="Initial credit allocation on registration",
        )
        db.add(transaction)
        await db.flush()

    return wallet


async def lock_escrow(db: AsyncSession, barter_session_id: int, user_id: int) -> Escrow | None:
    # Idempotent: a repeated lock call for the same session/user reuses the
    # existing locked escrow instead of stranding a second deposit (ISSUE-007).
    existing_result = await db.execute(
        select(Escrow).where(
            Escrow.barter_session_id == barter_session_id,
            Escrow.user_id == user_id,
            Escrow.status == "locked",
        )
    )
    existing = existing_result.scalar_one_or_none()
    if existing:
        return existing

    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        return None

    escrow_amount = calculate_escrow_amount(user.trust_score)

    wallet = await get_or_create_wallet(db, user_id)

    if wallet.available_balance < escrow_amount:
        return None

    wallet.available_balance -= escrow_amount
    wallet.locked_balance += escrow_amount

    transaction = CreditTransaction(
        user_id=user_id,
        barter_session_id=barter_session_id,
        transaction_type="escrow_lock",
        amount=-escrow_amount,
        balance_after=wallet.available_balance,
        description=f"Escrow locked for session {barter_session_id}",
    )
    db.add(transaction)

    escrow = Escrow(
        barter_session_id=barter_session_id,
        user_id=user_id,
        amount=escrow_amount,
        status="locked",
    )
    db.add(escrow)
    try:
        await db.flush()
    except IntegrityError:
        # Lost a concurrent race to lock the same (session, user) escrow —
        # the unique index caught it where the earlier check couldn't.
        # Roll back this insert and hand back the winner's row.
        await db.rollback()
        winner = await db.execute(
            select(Escrow).where(
                Escrow.barter_session_id == barter_session_id,
                Escrow.user_id == user_id,
                Escrow.status == "locked",
            )
        )
        return winner.scalar_one_or_none()

    return escrow


async def release_escrow(
    db: AsyncSession,
    escrow_id: int,
    release_type: str,
    penalty_amount: int = 0,
) -> Escrow | None:
    result = await db.execute(
        select(Escrow).where(Escrow.id == escrow_id, Escrow.status == "locked")
    )
    escrow = result.scalar_one_or_none()
    if not escrow:
        return None

    # A penalty larger than the deposit would produce a negative released
    # amount and deduct credits the escrow never held (ISSUE-012).
    if not (0 <= penalty_amount <= escrow.amount):
        raise ValueError(
            f"penalty_amount must be between 0 and escrow amount ({escrow.amount})"
        )

    wallet_result = await db.execute(select(Wallet).where(Wallet.user_id == escrow.user_id))
    wallet = wallet_result.scalar_one_or_none()
    if not wallet:
        return None

    released_amount = escrow.amount - penalty_amount

    if release_type in ("full_release", "partial_release"):
        if penalty_amount > 0:
            wallet.locked_balance -= escrow.amount
            wallet.available_balance += released_amount
            wallet.total_spent += penalty_amount

            transaction = CreditTransaction(
                user_id=escrow.user_id,
                barter_session_id=escrow.barter_session_id,
                transaction_type="escrow_penalty",
                amount=-penalty_amount,
                balance_after=wallet.available_balance,
                description=f"Penalty applied from escrow for session {escrow.barter_session_id}",
            )
            db.add(transaction)

            transaction = CreditTransaction(
                user_id=escrow.user_id,
                barter_session_id=escrow.barter_session_id,
                transaction_type="escrow_release",
                amount=released_amount,
                balance_after=wallet.available_balance,
                description=f"Partial escrow released for session {escrow.barter_session_id}",
            )
            db.add(transaction)
        else:
            wallet.locked_balance -= escrow.amount
            wallet.available_balance += escrow.amount

            transaction = CreditTransaction(
                user_id=escrow.user_id,
                barter_session_id=escrow.barter_session_id,
                transaction_type="escrow_release",
                amount=escrow.amount,
                balance_after=wallet.available_balance,
                description=f"Escrow released for session {escrow.barter_session_id}",
            )
            db.add(transaction)

        escrow.status = "released"
        escrow.released_at = datetime.utcnow()
        escrow.release_type = release_type

    elif release_type == "refund":
        wallet.locked_balance -= escrow.amount
        wallet.available_balance += escrow.amount

        transaction = CreditTransaction(
            user_id=escrow.user_id,
            barter_session_id=escrow.barter_session_id,
            transaction_type="refund",
            amount=escrow.amount,
            balance_after=wallet.available_balance,
            description=f"Escrow refunded for session {escrow.barter_session_id}",
        )
        db.add(transaction)

        escrow.status = "refunded"
        escrow.released_at = datetime.utcnow()
        escrow.release_type = release_type

    elif release_type == "penalty":
        wallet.locked_balance -= escrow.amount
        wallet.total_spent += escrow.amount

        transaction = CreditTransaction(
            user_id=escrow.user_id,
            barter_session_id=escrow.barter_session_id,
            transaction_type="escrow_penalty",
            amount=-escrow.amount,
            balance_after=wallet.available_balance,
            description=f"Escrow penalized for session {escrow.barter_session_id}",
        )
        db.add(transaction)

        escrow.status = "penalized"
        escrow.released_at = datetime.utcnow()
        escrow.release_type = release_type

    await db.flush()
    return escrow


async def _release_group(
    db: AsyncSession, escrows: list[Escrow], release_type: str, penalty_fraction: float
) -> int:
    """Release every locked escrow in the group, splitting the same penalty
    fraction across each. Handles legacy sessions with more than one locked
    escrow per user (ISSUE-007) as well as the normal single-escrow case."""
    released_total = 0
    for escrow in escrows:
        penalty_amount = int(escrow.amount * penalty_fraction)
        await release_escrow(db, escrow.id, release_type, penalty_amount)
        released_total += escrow.amount - penalty_amount
    return released_total


async def apply_settlement(
    db: AsyncSession,
    barter_session_id: int,
    qa_score: float,
) -> dict:
    result = await db.execute(
        select(Escrow).where(
            Escrow.barter_session_id == barter_session_id, Escrow.status == "locked"
        )
    )
    escrows = result.scalars().all()

    if not escrows:
        return {"error": "No locked escrows found"}

    escrows_by_user: dict[int, list[Escrow]] = {}
    for escrow in escrows:
        escrows_by_user.setdefault(escrow.user_id, []).append(escrow)

    contract_result = await db.execute(
        select(SessionContract).where(SessionContract.barter_session_id == barter_session_id)
    )
    contract = contract_result.scalar_one_or_none()

    if not contract:
        return {"error": "Session contract not found"}

    teacher_escrows = escrows_by_user.get(contract.teacher_user_id, [])
    learner_escrows = escrows_by_user.get(contract.learner_user_id, [])

    if qa_score >= 0.8:
        release_type = "full_release"
        bonus = ESCROW_CONFIG["teaching_bonus"]

        teacher_released = await _release_group(db, teacher_escrows, release_type, 0.0)
        learner_released = await _release_group(db, learner_escrows, release_type, 0.0)

        teacher_wallet = await get_or_create_wallet(db, contract.teacher_user_id)
        teacher_wallet.available_balance += bonus
        teacher_wallet.total_earned += bonus

        bonus_tx = CreditTransaction(
            user_id=contract.teacher_user_id,
            barter_session_id=barter_session_id,
            transaction_type="bonus",
            amount=bonus,
            balance_after=teacher_wallet.available_balance,
            description=f"Teaching bonus for successful session {barter_session_id}",
        )
        db.add(bonus_tx)

        trust_delta = 0.05
        teacher_trust_delta = 0.05
        learner_trust_delta = 0.05

    elif qa_score >= 0.5:
        release_type = "partial_release"
        penalty_fraction = 1 - ((qa_score - 0.5) / 0.3)

        teacher_released = await _release_group(db, teacher_escrows, release_type, penalty_fraction)
        learner_released = await _release_group(db, learner_escrows, release_type, penalty_fraction)

        trust_delta = 0.02
        teacher_trust_delta = 0.02
        learner_trust_delta = 0.02

    else:
        release_type = "penalty"

        teacher_released = await _release_group(db, teacher_escrows, "penalty", 1.0)
        learner_released = await _release_group(db, learner_escrows, "refund", 0.0)

        trust_delta = -0.10
        teacher_trust_delta = -0.10
        learner_trust_delta = 0.0

    await db.flush()

    return {
        "release_type": release_type,
        "trust_delta": trust_delta,
        "teacher_trust_delta": teacher_trust_delta,
        "learner_trust_delta": learner_trust_delta,
        "teacher_escrow_released": teacher_released,
        "learner_escrow_released": learner_released,
        "teacher_bonus": ESCROW_CONFIG["teaching_bonus"] if qa_score >= 0.8 else 0,
    }


async def get_wallet_by_user(db: AsyncSession, user_id: int) -> Wallet | None:
    result = await db.execute(select(Wallet).where(Wallet.user_id == user_id))
    return result.scalar_one_or_none()


async def get_escrows_by_session(db: AsyncSession, barter_session_id: int) -> list[Escrow]:
    result = await db.execute(select(Escrow).where(Escrow.barter_session_id == barter_session_id))
    return list(result.scalars().all())


async def get_transactions_by_user(db: AsyncSession, user_id: int) -> list[CreditTransaction]:
    result = await db.execute(
        select(CreditTransaction)
        .where(CreditTransaction.user_id == user_id)
        .order_by(CreditTransaction.created_at.desc())
    )
    return list(result.scalars().all())
