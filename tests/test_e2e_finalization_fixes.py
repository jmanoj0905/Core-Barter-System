"""
Regression tests for ISSUES.md findings 001, 002, 003, 005, 006, 007.

Covers the single server-controlled finalization path in confirm_session:
verdict generated from actual topic evidence before settlement, settlement
and trust applied exactly once, participant membership enforced, manual
termination refunds escrow, and repeated escrow locking doesn't strand
deposits.
"""

import pytest


async def _create_and_start(client, **overrides):
    payload = {
        "topic": "Python programming basics",
        "scope": "Variables, loops, functions, and data types in Python",
        "agreed_duration_minutes": 10,
        "teacher_user_id": 1,
        "learner_user_id": 2,
    }
    payload.update(overrides)
    resp = await client.post("/session/create", json=payload)
    barter_id = resp.json()["barter_id"]
    await client.post(f"/session/{barter_id}/start")
    return barter_id


# ---------------------------------------------------------------------------
# ISSUE-006 — only contract participants can confirm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_rejects_nonparticipant(backend_client):
    barter_id = await _create_and_start(backend_client)

    resp = await backend_client.post(
        f"/session/{barter_id}/confirm", json={"user_id": 99}
    )
    assert resp.status_code == 403

    status = await backend_client.get(f"/session/{barter_id}/status")
    assert status.json()["both_confirmed"] is False


# ---------------------------------------------------------------------------
# ISSUE-001 / ISSUE-003 — settlement waits for the authoritative verdict,
# and topic monitoring evidence can veto it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_settles_using_real_verdict_not_pending(backend_client):
    """A session with fully on-topic evidence should not be penalized just
    because the verdict didn't exist yet when confirmation landed."""
    barter_id = await _create_and_start(backend_client)

    for i in range(1, 4):
        await backend_client.post("/window/result", json={
            "barter_id": barter_id,
            "window_id": i,
            "classification": "correct",
            "similarity_score": 0.8,
        })

    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 1})
    resp = await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 2})
    data = resp.json()
    assert data["both_confirmed"] is True

    settlement = data["settlement"]
    assert "error" not in settlement
    # On-topic evidence + confirmation should not fall into the penalty branch.
    assert settlement["release_type"] != "penalty"

    escrows = (await backend_client.get(f"/escrow/{barter_id}")).json()
    assert all(e["status"] != "locked" for e in escrows)


@pytest.mark.asyncio
async def test_confirm_verdict_reflects_wholly_off_topic_evidence(backend_client):
    """ISSUE-003: ten-of-ten incorrect windows must not settle as a success
    just because both users confirmed and duration/elapsed checks pass."""
    barter_id = await _create_and_start(backend_client)

    for i in range(1, 11):
        await backend_client.post("/window/result", json={
            "barter_id": barter_id,
            "window_id": i,
            "classification": "incorrect",
            "similarity_score": 0.05,
        })
    for _ in range(9):
        await backend_client.post("/warnings/log", json={
            "barter_id": barter_id,
            "severity": "severe",
            "reason": "off-topic",
        })

    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 1})
    resp = await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 2})
    data = resp.json()

    verdict = (await backend_client.get(f"/verdict/{barter_id}")).json()
    assert verdict["verdict"] == "DISPUTE"

    settlement = data["settlement"]
    assert settlement["release_type"] == "penalty"
    assert settlement["teacher_trust_delta"] < 0


# ---------------------------------------------------------------------------
# ISSUE-002 — trust update is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_trust_update_does_not_remutate(backend_client):
    barter_id = await _create_and_start(backend_client)

    for i in range(1, 4):
        await backend_client.post("/window/result", json={
            "barter_id": barter_id,
            "window_id": i,
            "classification": "correct",
            "similarity_score": 0.8,
        })

    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 1})
    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 2})

    first = (await backend_client.post(f"/trust/{barter_id}/update")).json()
    second = (await backend_client.post(f"/trust/{barter_id}/update")).json()
    third = (await backend_client.post(f"/trust/{barter_id}/update")).json()

    assert first == second == third

    users = {u["id"]: u["trust_score"] for u in (await backend_client.get("/users")).json()}
    assert users[1] == pytest.approx(first["user_1_trust"]["after"], abs=1e-4)


# ---------------------------------------------------------------------------
# ISSUE-005 — manual termination refunds locked escrow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminate_refunds_locked_escrow(backend_client):
    barter_id = await _create_and_start(backend_client)

    escrows_before = (await backend_client.get(f"/escrow/{barter_id}")).json()
    assert all(e["status"] == "locked" for e in escrows_before)

    resp = await backend_client.post(
        f"/session/{barter_id}/terminate", json={"reason": "test"}
    )
    assert resp.status_code == 200

    escrows_after = (await backend_client.get(f"/escrow/{barter_id}")).json()
    assert all(e["status"] == "refunded" for e in escrows_after)

    # Re-terminating an already-terminated session is rejected, not a silent
    # duplicate refund.
    resp = await backend_client.post(
        f"/session/{barter_id}/terminate", json={"reason": "test again"}
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# ISSUE-007 — repeated escrow locking does not strand deposits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_escrow_lock_is_idempotent(backend_client):
    barter_id = await _create_and_start(backend_client)

    # /escrow/lock again after /start already locked both users' deposits.
    await backend_client.post("/escrow/lock", params={"barter_id": barter_id})
    await backend_client.post("/escrow/lock", params={"barter_id": barter_id})

    escrows = (await backend_client.get(f"/escrow/{barter_id}")).json()
    locked = [e for e in escrows if e["status"] == "locked"]
    # One locked escrow per user, not one per lock call.
    assert len(locked) == 2

    resp = await backend_client.post(
        f"/settlement/{barter_id}",
        json={"barter_session_id": barter_id, "qa_score": 1.0},
    )
    assert resp.status_code == 200

    escrows_after = (await backend_client.get(f"/escrow/{barter_id}")).json()
    assert all(e["status"] != "locked" for e in escrows_after)
