import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app
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
