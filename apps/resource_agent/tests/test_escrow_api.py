import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app import escrow as escrow_service
from app.accounts import ensure_accounts
from app.database import get_db
from app.main import app
from app.models import Account
from app.policy import PolicyConfig

CFG = PolicyConfig()


@pytest_asyncio.fixture
async def client(db):
    async def _override():
        yield db

    app.dependency_overrides[get_db] = _override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def _open_accounts(client, user_ids=(1, 2)):
    for user_id in user_ids:
        await client.post("/resource/accounts", json={"user_id": user_id})


RESERVE_BODY = {
    "session_id": 100,
    "participants": [
        {"user_id": 1, "trust_score": 0.30},
        {"user_id": 2, "trust_score": 0.72},
    ],
}


@pytest.mark.asyncio
async def test_create_account_grants_initial_credits(client):
    res = await client.post("/resource/accounts", json={"user_id": 1})

    assert res.status_code == 200
    assert res.json()["available"] == CFG.initial_grant


@pytest.mark.asyncio
async def test_reserve_locks_asymmetric_stakes(client):
    await _open_accounts(client)

    res = await client.post("/resource/escrow/reserve", json=RESERVE_BODY)

    assert res.status_code == 200
    body = res.json()
    stakes = {e["user_id"]: e["amount"] for e in body["escrows"]}
    assert stakes == {1: 28, 2: 11}

    account_1 = (await client.get("/resource/accounts/1")).json()
    assert account_1["available"] == CFG.initial_grant - 28
    assert account_1["locked"] == 28


@pytest.mark.asyncio
async def test_reserve_replay_does_not_double_lock(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    res = await client.post("/resource/escrow/reserve", json=RESERVE_BODY)

    assert res.status_code == 200
    account_1 = (await client.get("/resource/accounts/1")).json()
    assert account_1["locked"] == 28


@pytest.mark.asyncio
async def test_reserve_is_all_or_nothing_when_one_user_is_short(client):
    """Lock up funds across two sessions, then verify a third reservation
    fails cleanly without moving anyone's credits."""
    await _open_accounts(client)

    # trust 0.0 -> maximum 40-credit stake each, against a 100-credit grant.
    def body(session_id):
        return {
            "session_id": session_id,
            "participants": [
                {"user_id": 1, "trust_score": 0.0},
                {"user_id": 2, "trust_score": 0.0},
            ],
        }

    assert (await client.post("/resource/escrow/reserve", json=body(900))).status_code == 200
    assert (await client.post("/resource/escrow/reserve", json=body(901))).status_code == 200
    # each user: available 100 - 80 = 20, locked 80

    res = await client.post("/resource/escrow/reserve", json=body(902))

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "INSUFFICIENT_CREDITS"
    assert res.json()["detail"]["required"] == 40
    assert res.json()["detail"]["available"] == 20

    # Neither user may be touched by the failed reservation.
    for user_id in (1, 2):
        account = (await client.get(f"/resource/accounts/{user_id}")).json()
        assert account["locked"] == 80
        assert account["available"] == 20


@pytest.mark.asyncio
async def test_settle_full_release_returns_stakes_and_pays_bonuses(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)

    res = await client.post(
        "/resource/escrow/settle",
        json={
            "session_id": 100,
            "verdict_type": "SUCCESSFUL",
            "qa_score": 0.92,
            "per_user": {
                "1": {"quality": 0.9, "engagement": 0.8, "no_show": False},
                "2": {"quality": 0.9, "engagement": 0.8, "no_show": False},
            },
        },
    )

    assert res.status_code == 200
    assert res.json()["mode"] == "FULL_RELEASE"

    account_1 = (await client.get("/resource/accounts/1")).json()
    assert account_1["locked"] == 0
    assert account_1["available"] > CFG.initial_grant - 28


@pytest.mark.asyncio
async def test_settle_replay_does_not_double_pay(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    body = {
        "session_id": 100,
        "verdict_type": "SUCCESSFUL",
        "qa_score": 0.92,
        "per_user": {
            "1": {"quality": 0.9, "engagement": 0.8, "no_show": False},
            "2": {"quality": 0.9, "engagement": 0.8, "no_show": False},
        },
    }
    await client.post("/resource/escrow/settle", json=body)
    first = (await client.get("/resource/accounts/1")).json()["available"]
    await client.post("/resource/escrow/settle", json=body)
    second = (await client.get("/resource/accounts/1")).json()["available"]

    assert first == second


@pytest.mark.asyncio
async def test_settling_a_voided_escrow_is_rejected(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    await client.post("/resource/escrow/void", json={"session_id": 100, "reason": "abandoned"})

    res = await client.post(
        "/resource/escrow/settle",
        json={
            "session_id": 100,
            "verdict_type": "SUCCESSFUL",
            "qa_score": 0.92,
            "per_user": {"1": {"quality": 0.9, "engagement": 0.8, "no_show": False}},
        },
    )

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "ESCROW_STATE"


@pytest.mark.asyncio
async def test_void_returns_locked_credits(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    await client.post("/resource/escrow/void", json={"session_id": 100, "reason": "crash"})

    account_1 = (await client.get("/resource/accounts/1")).json()
    assert account_1["available"] == CFG.initial_grant
    assert account_1["locked"] == 0


@pytest.mark.asyncio
async def test_hold_blocks_settlement_until_resolved(client):
    await _open_accounts(client)
    await client.post("/resource/escrow/reserve", json=RESERVE_BODY)
    await client.post("/resource/escrow/100/hold", json={"reason": "toxicity flagged"})

    escrows = (await client.get("/resource/escrow/100")).json()
    assert {e["state"] for e in escrows["escrows"]} == {"HELD"}


@pytest.mark.asyncio
async def test_ledger_lists_entries_for_a_user(client):
    await _open_accounts(client, user_ids=(1,))

    res = await client.get("/resource/ledger/1")

    assert res.status_code == 200
    assert res.json()["entries"][0]["entry_type"] == "grant"


def _is_account_lock(stmt) -> bool:
    """True for a SELECT ... FOR UPDATE against the accounts table."""
    return (
        getattr(stmt, "_for_update_arg", None) is not None
        and bool(getattr(stmt, "column_descriptions", None))
        and stmt.column_descriptions[0]["type"] is Account
    )


def _pause_after_first_account_lock(session, delay: float = 0.2):
    """Instrument `session.execute` so it sleeps briefly right after this
    session acquires its FIRST account row lock (and only that one).

    This is a bounded delay, not a rendezvous: it cannot hang a session that
    never reaches an account lock (e.g. one that blocks *inside* its own
    FOR UPDATE select waiting on the other session, as happens once both
    sides canonicalise on the same first account). It only widens the window
    in which two concurrent transactions can genuinely interleave their lock
    acquisition, which real network-speed local Postgres queries otherwise
    complete too fast to reproduce reliably.
    """
    orig_execute = session.execute
    paused = False

    async def patched_execute(stmt, *a, **kw):
        nonlocal paused
        result = await orig_execute(stmt, *a, **kw)
        if not paused and _is_account_lock(stmt):
            paused = True
            await asyncio.sleep(delay)
        return result

    session.execute = patched_execute


@pytest.mark.asyncio
async def test_reserve_locks_accounts_in_the_same_canonical_order_as_settle(
    db, session_factory
):
    """A concurrent reserve and settle touching the same two users' accounts
    must not deadlock, even when account ids run opposite to user ids.

    post_entry (used by settle/void/reserve's own movement-posting) locks by
    ascending account_id. Before this fix, reserve's *sufficiency* pre-check
    locked by ascending user_id instead — a different, uncoordinated order.
    Those two orders only coincide when accounts happen to have been created
    in user_id order, which nothing guarantees: accounts come into existence
    lazily, whenever `POST /resource/accounts` first runs for that user.

    To make the two orders provably disagree, accounts are created here in
    *reverse* user_id order (user 2 before user 1), so account_id(2) <
    account_id(1) while user_id(1) < user_id(2). A concurrent settle
    (session A, already reserved) and reserve (session B, brand new) then
    touch the same two users' `user_available` accounts. Under the old
    user_id-ordered pre-check, reserve locks user 1's account first while
    settle's (correctly canonical) post_entry lock ends up requesting user
    1's account only after already holding user 2's — the classic opposite-
    order deadlock this whole fix family exists to remove, just relocated
    to a different boundary.
    """
    await ensure_accounts(db, 2, CFG)  # created first -> smaller account_id
    await ensure_accounts(db, 1, CFG)  # created second -> larger account_id
    await db.commit()

    participants = [
        {"user_id": 1, "trust_score": 0.3},
        {"user_id": 2, "trust_score": 0.3},
    ]

    async with session_factory() as setup:
        await escrow_service.reserve(setup, 500, participants, CFG)
        await setup.commit()

    async def do_settle():
        async with session_factory() as session:
            _pause_after_first_account_lock(session)
            result = await escrow_service.settle(
                session, 500, "SUCCESSFUL", 0.9,
                {
                    1: {"quality": 0.5, "engagement": 0.5, "no_show": False},
                    2: {"quality": 0.5, "engagement": 0.5, "no_show": False},
                },
                CFG,
            )
            await session.commit()
            return result

    async def do_reserve():
        async with session_factory() as session:
            _pause_after_first_account_lock(session)
            result = await escrow_service.reserve(session, 501, participants, CFG)
            await session.commit()
            return result

    results = await asyncio.wait_for(
        asyncio.gather(do_settle(), do_reserve()), timeout=10
    )

    settle_result, reserve_result = results
    assert settle_result["mode"] == "FULL_RELEASE"
    assert reserve_result["created"] is True
