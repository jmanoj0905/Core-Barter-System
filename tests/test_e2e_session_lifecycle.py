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
