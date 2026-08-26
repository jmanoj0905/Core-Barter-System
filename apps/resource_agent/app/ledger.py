from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Account, JournalEntry, LedgerLine

SYSTEM_KINDS = ("platform_mint", "platform_revenue")


class UnbalancedEntry(Exception):
    """Raised when the ledger lines of a journal entry do not sum to zero."""


@dataclass(frozen=True)
class Movement:
    """A policy-level instruction: move `amount` into the named account.

    `account` is one of: "user_available:{id}", "user_locked:{id}",
    "platform_mint", "platform_revenue".
    """

    account: str
    amount: int


async def get_or_create_account(db: AsyncSession, kind: str, user_id: int | None) -> Account:
    stmt = select(Account).where(Account.kind == kind, Account.user_id == user_id)
    account = (await db.execute(stmt)).scalar_one_or_none()
    if account:
        return account

    account = Account(kind=kind, user_id=user_id, balance=0)
    db.add(account)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        account = (await db.execute(stmt)).scalar_one()
    return account


def _parse_account_key(key: str) -> tuple[str, int | None]:
    if ":" in key:
        kind, raw_user_id = key.split(":", 1)
        return kind, int(raw_user_id)
    return key, None


async def resolve_movements(
    db: AsyncSession, movements: Sequence[Movement]
) -> list[tuple[int, int]]:
    """Turn policy Movements into (account_id, amount) ledger lines."""
    lines: list[tuple[int, int]] = []
    for movement in movements:
        kind, user_id = _parse_account_key(movement.account)
        account = await get_or_create_account(db, kind, user_id)
        lines.append((account.id, movement.amount))
    return lines


async def post_entry(
    db: AsyncSession,
    *,
    idempotency_key: str,
    entry_type: str,
    session_id: int | None,
    payload: dict,
    lines: Sequence[tuple[int, int]],
) -> tuple[JournalEntry, bool]:
    """Post a balanced journal entry. Returns (entry, created).

    On idempotent replay the existing entry is returned with created=False and
    no new ledger lines are written.
    """
    if sum(amount for _, amount in lines) != 0:
        raise UnbalancedEntry(
            f"lines for {idempotency_key} sum to {sum(a for _, a in lines)}, expected 0"
        )

    existing = (
        await db.execute(
            select(JournalEntry).where(JournalEntry.idempotency_key == idempotency_key)
        )
    ).scalar_one_or_none()
    if existing:
        return existing, False

    savepoint = await db.begin_nested()
    entry = JournalEntry(
        idempotency_key=idempotency_key,
        entry_type=entry_type,
        session_id=session_id,
        payload=payload,
    )
    db.add(entry)
    try:
        await db.flush()
    except IntegrityError:
        await savepoint.rollback()
        existing = (
            await db.execute(
                select(JournalEntry).where(
                    JournalEntry.idempotency_key == idempotency_key
                )
            )
        ).scalar_one()
        return existing, False

    for account_id, amount in lines:
        # Lock the account row with an explicit SELECT ... FOR UPDATE before
        # touching it, and before inserting the referencing LedgerLine. Three
        # reasons this shape matters:
        #  1. `db.get(Account, account_id, with_for_update=True)` depends on
        #     SQLAlchemy's internal `for_update_arg is None` check to decide
        #     whether to bypass the identity map; an explicit statement makes
        #     the lock unconditional and doesn't rely on that internal detail.
        #  2. Locking the row *before* adding the LedgerLine matters: the
        #     LedgerLine insert's FK reference takes a shared lock on the
        #     account row, and if two concurrent transactions both insert
        #     their LedgerLine first, they can each hold that shared lock and
        #     then deadlock trying to upgrade to the exclusive FOR UPDATE
        #     lock. Taking the exclusive lock first avoids that.
        #  3. `populate_existing=True` is required alongside the lock, not
        #     optional decoration: callers on the real path (resolve_movements)
        #     always load these same accounts via get_or_create_account just
        #     before calling post_entry, so the account is already in this
        #     session's identity map. Without populate_existing, SQLAlchemy
        #     returns that cached object as-is instead of overwriting its
        #     attributes from the row just locked — the lock would then
        #     serialize the write but `account.balance += amount` below would
        #     still increment a stale pre-lock value, losing a concurrent
        #     update to the cached balance column even though the ledger_lines
        #     themselves are both written correctly.
        account = (
            await db.execute(
                select(Account)
                .where(Account.id == account_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        ).scalar_one()
        db.add(LedgerLine(entry_id=entry.id, account_id=account_id, amount=amount))
        account.balance += amount

    await db.flush()
    return entry, True


async def balance_of(db: AsyncSession, account_id: int) -> int:
    """Derive an account balance from ledger lines (ignores the cached column)."""
    stmt = select(func.coalesce(func.sum(LedgerLine.amount), 0)).where(
        LedgerLine.account_id == account_id
    )
    return int((await db.execute(stmt)).scalar_one())
