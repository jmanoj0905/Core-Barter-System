"""
E2E tests for verdict generation and trust score updates.

Tests the backend endpoints:
  - /session/{id}/drift-summary (receives from warning engine)
  - /verdict/{id}/generate
  - /verdict/{id} (GET)
  - /trust/{id}/update
"""

import pytest


async def _create_and_start(client, payload):
    """Helper: create + start a session, return barter_id."""
    resp = await client.post("/session/create", json=payload)
    barter_id = resp.json()["barter_id"]
    await client.post(f"/session/{barter_id}/start")
    return barter_id


# ---------------------------------------------------------------------------
# Drift Summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drift_summary_creates_pending_verdict(backend_client, create_session_payload):
    barter_id = await _create_and_start(backend_client, create_session_payload)

    resp = await backend_client.post(f"/session/{barter_id}/drift-summary", json={
        "barter_id": barter_id,
        "total_windows": 5,
        "incorrect_windows": 1,
        "percent_incorrect": 20.0,
        "max_consecutive_incorrect": 1,
        "warning_count": 0,
        "warnings": [],
        "terminated_early": False,
    })
    assert resp.status_code == 200

    # Verdict should exist in PENDING state
    verdict_resp = await backend_client.get(f"/verdict/{barter_id}")
    assert verdict_resp.status_code == 200
    data = verdict_resp.json()
    assert data["verdict"] == "PENDING"
    assert data["on_topic_percentage"] == 80.0


@pytest.mark.asyncio
async def test_engagement_summary_stored(backend_client, create_session_payload):
    barter_id = await _create_and_start(backend_client, create_session_payload)

    # First create drift summary (creates verdict row)
    await backend_client.post(f"/session/{barter_id}/drift-summary", json={
        "barter_id": barter_id,
        "total_windows": 3,
        "incorrect_windows": 0,
        "percent_incorrect": 0.0,
        "max_consecutive_incorrect": 0,
        "warning_count": 0,
    })

    # Post engagement summary
    resp = await backend_client.post(f"/session/{barter_id}/engagement-summary", json={
        "learner_engagement_score": 0.72,
        "learner_speaking_seconds": 120.0,
        "teacher_speaking_seconds": 300.0,
        "learner_question_count": 5,
    })
    assert resp.status_code == 200

    # Verify it's stored in drift_summary
    verdict_resp = await backend_client.get(f"/verdict/{barter_id}")
    drift = verdict_resp.json()["drift_summary"]
    assert "engagement" in drift
    assert drift["engagement"]["learner_engagement_score"] == 0.72


# ---------------------------------------------------------------------------
# Verdict Generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verdict_successful_with_both_confirms_and_duration(
    backend_client, create_session_payload
):
    barter_id = await _create_and_start(backend_client, create_session_payload)

    # Confirm both users → marks session completed
    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 1})
    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 2})

    # Generate verdict — duration won't pass (session was too short)
    # but confirmation passes
    resp = await backend_client.post(f"/verdict/{barter_id}/generate")
    assert resp.status_code == 200
    data = resp.json()
    # Only confirmation passes, not duration → PARTIAL
    assert data["confirmation_check"] is True
    assert data["verdict"] == "PARTIAL"


@pytest.mark.asyncio
async def test_verdict_dispute_on_terminated_session(
    backend_client, create_session_payload
):
    barter_id = await _create_and_start(backend_client, create_session_payload)

    await backend_client.post(
        f"/session/{barter_id}/terminate", json={"reason": "test"}
    )

    resp = await backend_client.post(f"/verdict/{barter_id}/generate")
    assert resp.status_code == 200
    assert resp.json()["verdict"] == "DISPUTE"


@pytest.mark.asyncio
async def test_verdict_dispute_no_confirms_short_duration(
    backend_client, create_session_payload
):
    barter_id = await _create_and_start(backend_client, create_session_payload)

    resp = await backend_client.post(f"/verdict/{barter_id}/generate")
    data = resp.json()
    assert data["verdict"] == "DISPUTE"
    assert data["duration_check"] is False
    assert data["confirmation_check"] is False


@pytest.mark.asyncio
async def test_verdict_not_found_for_unknown_session(backend_client):
    resp = await backend_client.get("/verdict/9999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Trust Score Updates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trust_update_after_successful_verdict(
    backend_client, create_session_payload
):
    barter_id = await _create_and_start(backend_client, create_session_payload)

    # Confirm both
    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 1})
    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 2})

    # Generate verdict (PARTIAL because duration is short)
    await backend_client.post(f"/verdict/{barter_id}/generate")

    # Update trust
    resp = await backend_client.post(f"/trust/{barter_id}/update")
    assert resp.status_code == 200
    data = resp.json()

    # Both users should have trust updated
    assert data["user_1_trust"]["after"] != data["user_1_trust"]["before"]
    assert data["user_2_trust"]["after"] != data["user_2_trust"]["before"]
    # PARTIAL verdict → qa_score=0.5, quality_adjusted=(0.5+0.8)/2=0.65
    # new_trust = 0.30 * 0.3 + 0.65 * 0.7 = 0.09 + 0.455 = 0.545
    assert data["user_1_trust"]["after"] == pytest.approx(0.545, abs=0.01)


@pytest.mark.asyncio
async def test_trust_update_dispute_lowers_trust(
    backend_client, create_session_payload
):
    barter_id = await _create_and_start(backend_client, create_session_payload)

    await backend_client.post(f"/verdict/{barter_id}/generate")  # DISPUTE

    resp = await backend_client.post(f"/trust/{barter_id}/update")
    data = resp.json()
    # DISPUTE → qa_score=0.0, quality_adjusted=(0+0.8)/2=0.4
    # new_trust = 0.30 * 0.3 + 0.4 * 0.7 = 0.09 + 0.28 = 0.37
    assert data["user_1_trust"]["after"] == pytest.approx(0.37, abs=0.01)


@pytest.mark.asyncio
async def test_trust_update_without_verdict_fails(
    backend_client, create_session_payload
):
    barter_id = await _create_and_start(backend_client, create_session_payload)

    resp = await backend_client.post(f"/trust/{barter_id}/update")
    assert resp.status_code == 404
    assert "Generate verdict first" in resp.json()["detail"]
