"""
E2E tests for the session lifecycle:
  create → start → confirm (both users) → completed
Also covers edge cases: 404s, double confirm, terminate.
"""

import pytest


@pytest.mark.asyncio
async def test_create_session(backend_client, create_session_payload):
    resp = await backend_client.post("/session/create", json=create_session_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "barter_id" in data
    assert data["status"] == "proposed"
    assert data["teacher_user_id"] == 1
    assert data["learner_user_id"] == 2


@pytest.mark.asyncio
async def test_start_session(backend_client, create_session_payload):
    # Create
    resp = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = resp.json()["barter_id"]

    # Start
    resp = await backend_client.post(f"/session/{barter_id}/start")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "active"
    assert "started_at" in data


@pytest.mark.asyncio
async def test_start_nonexistent_session(backend_client):
    resp = await backend_client.post("/session/9999/start")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_start_already_active_session(backend_client, create_session_payload):
    resp = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = resp.json()["barter_id"]

    await backend_client.post(f"/session/{barter_id}/start")
    # Starting again should return current state, not error
    resp = await backend_client.post(f"/session/{barter_id}/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


@pytest.mark.asyncio
async def test_session_status(backend_client, create_session_payload):
    resp = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = resp.json()["barter_id"]

    resp = await backend_client.get(f"/session/{barter_id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "proposed"
    assert data["agreed_duration_minutes"] == 10
    assert data["both_confirmed"] is False


@pytest.mark.asyncio
async def test_confirm_single_user(backend_client, create_session_payload):
    resp = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = resp.json()["barter_id"]
    await backend_client.post(f"/session/{barter_id}/start")

    # Confirm user 1
    resp = await backend_client.post(
        f"/session/{barter_id}/confirm", json={"user_id": 1}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 1 in data["confirmed_by"]
    assert data["both_confirmed"] is False


@pytest.mark.asyncio
async def test_confirm_both_users_completes_session(backend_client, create_session_payload):
    resp = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = resp.json()["barter_id"]
    await backend_client.post(f"/session/{barter_id}/start")

    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 1})
    resp = await backend_client.post(
        f"/session/{barter_id}/confirm", json={"user_id": 2}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["both_confirmed"] is True

    # Session should now be completed
    status_resp = await backend_client.get(f"/session/{barter_id}/status")
    assert status_resp.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_double_confirm_rejected(backend_client, create_session_payload):
    resp = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = resp.json()["barter_id"]
    await backend_client.post(f"/session/{barter_id}/start")

    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 1})
    resp = await backend_client.post(
        f"/session/{barter_id}/confirm", json={"user_id": 1}
    )
    assert resp.status_code == 400
    assert "already confirmed" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_terminate_session(backend_client, create_session_payload):
    resp = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = resp.json()["barter_id"]
    await backend_client.post(f"/session/{barter_id}/start")

    resp = await backend_client.post(
        f"/session/{barter_id}/terminate",
        json={"reason": "Testing termination"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "terminated"

    # Cannot start a terminated session
    resp = await backend_client.post(f"/session/{barter_id}/start")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_session_status_shows_elapsed_time(backend_client, create_session_payload):
    resp = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = resp.json()["barter_id"]
    await backend_client.post(f"/session/{barter_id}/start")

    resp = await backend_client.get(f"/session/{barter_id}/status")
    data = resp.json()
    assert data["can_complete"] is True
    assert data["elapsed_minutes"] >= 0.0


@pytest.mark.asyncio
async def test_session_start_locks_real_escrow(backend_client, monkeypatch):
    """Backend must delegate escrow to resource_agent, not compute it locally."""
    calls = {}

    async def fake_reserve(self, session_id, participants):
        calls["session_id"] = session_id
        calls["participants"] = participants
        return {
            "session_id": session_id,
            "escrows": [
                {"user_id": p["user_id"], "amount": 28, "state": "RESERVED"}
                for p in participants
            ],
            "created": True,
        }

    from app.clients.resource import ResourceClient

    monkeypatch.setattr(ResourceClient, "reserve", fake_reserve)

    create = await backend_client.post(
        "/session/create",
        json={
            "teacher_user_id": 1,
            "learner_user_id": 2,
            "topic": "Python basics",
            "scope": "Variables, loops, functions",
            "agreed_duration_minutes": 30,
        },
    )
    barter_id = create.json()["barter_id"]

    res = await backend_client.post(f"/session/{barter_id}/start")

    assert res.status_code == 200
    assert calls["session_id"] == barter_id
    assert {p["user_id"] for p in calls["participants"]} == {1, 2}
    assert all("trust_score" in p for p in calls["participants"])


@pytest.mark.asyncio
async def test_session_start_insufficient_credits_names_regen_wait(
    backend_client, create_session_payload, monkeypatch
):
    """A 400 from a failed reserve must name who is short and how long
    regeneration takes to cover the gap."""
    from app.clients.resource import InsufficientCredits, ResourceClient

    async def fake_reserve(self, session_id, participants):
        raise InsufficientCredits(
            user_id=1,
            required=40,
            available=20,
            shortfall=20,
            regen_eta={
                "coverable": True,
                "regen_rate_per_day": 5,
                "days_until_covered": 4,
                "eta": "2026-08-30T00:00:00+00:00",
            },
        )

    monkeypatch.setattr(ResourceClient, "reserve", fake_reserve)

    create = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = create.json()["barter_id"]

    res = await backend_client.post(f"/session/{barter_id}/start")

    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "user 1" in detail
    assert "short by 20" in detail
    assert "4 day" in detail


@pytest.mark.asyncio
async def test_session_start_insufficient_credits_uncoverable_by_regen(
    backend_client, create_session_payload, monkeypatch
):
    """When regeneration alone can never close the gap, say so instead of
    fabricating a day count."""
    from app.clients.resource import InsufficientCredits, ResourceClient

    async def fake_reserve(self, session_id, participants):
        raise InsufficientCredits(
            user_id=1,
            required=120,
            available=20,
            shortfall=100,
            regen_eta={
                "coverable": False,
                "regen_rate_per_day": 20,
                "days_until_covered": None,
                "eta": None,
            },
        )

    monkeypatch.setattr(ResourceClient, "reserve", fake_reserve)

    create = await backend_client.post("/session/create", json=create_session_payload)
    barter_id = create.json()["barter_id"]

    res = await backend_client.post(f"/session/{barter_id}/start")

    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "never" in detail.lower()


@pytest.mark.asyncio
async def test_resource_client_post_degrades_on_malformed_409(monkeypatch):
    """A 409 body resource_agent sends that this client doesn't recognize
    (missing the INSUFFICIENT_CREDITS keys, or a different code entirely)
    must raise a handled ResourceProtocolError, not fall through to
    `raise_for_status()` and blow up as an unhandled httpx.HTTPStatusError.

    `backend_client`'s conftest stubs ResourceClient.reserve/settle wholesale
    for the rest of this suite, which bypasses `_post` entirely -- so this
    hits `_post` directly instead of going through a route.
    """
    import httpx

    from app.clients.resource import ResourceClient, ResourceProtocolError

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None):
            request = httpx.Request("POST", url)
            return httpx.Response(
                409, json={"detail": {"code": "SOMETHING_ELSE"}}, request=request
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    client = ResourceClient(base_url="http://resource-agent")
    with pytest.raises(ResourceProtocolError):
        await client._post("/resource/escrow/reserve", {"session_id": 1, "participants": []})
