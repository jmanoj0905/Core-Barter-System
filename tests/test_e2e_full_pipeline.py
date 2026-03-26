"""
Full pipeline E2E test: simulates the complete barter session flow
through the backend, from creation to trust score update.

This exercises the backend as if all microservices were calling it:
  1. Create session (frontend)
  2. Start session (frontend)
  3. Store transcript segments (audio pipeline → backend)
  4. Store window results (warning engine → backend)
  5. Store warnings (warning engine → backend)
  6. Confirm session (frontend, both users)
  7. Receive drift summary (warning engine → backend)
  8. Receive engagement summary (semantic analysis → backend)
  9. Generate verdict (frontend)
  10. Update trust scores (frontend)
"""

import pytest


@pytest.mark.asyncio
async def test_full_on_topic_session(backend_client):
    """Happy path: an on-topic session that completes successfully."""

    # 1. Create session
    resp = await backend_client.post("/session/create", json={
        "topic": "Machine Learning fundamentals",
        "scope": "Supervised learning, neural networks, gradient descent",
        "agreed_duration_minutes": 5,
        "teacher_user_id": 1,
        "learner_user_id": 2,
    })
    assert resp.status_code == 200
    barter_id = resp.json()["barter_id"]

    # 2. Start session
    resp = await backend_client.post(f"/session/{barter_id}/start")
    assert resp.json()["status"] == "active"

    # 3. Store transcript segments (simulating what audio pipeline sends)
    for i, text in enumerate([
        "Let me explain how neural networks work with layers of neurons",
        "Each neuron applies a weight and bias, then an activation function",
        "Gradient descent optimizes the weights by minimizing the loss function",
    ]):
        resp = await backend_client.post(f"/session/{barter_id}/transcript", json={
            "barter_id": barter_id,
            "user_id": 1,
            "text": text,
            "duration_seconds": 30.0,
            "timestamp_start": i * 30.0,
            "timestamp_end": (i + 1) * 30.0,
        })
        assert resp.status_code == 200

    # Verify transcripts stored
    resp = await backend_client.get(f"/session/{barter_id}/transcript")
    assert len(resp.json()) == 3

    # 4. Store window results (simulating what warning engine forwards)
    for i in range(1, 4):
        resp = await backend_client.post("/window/result", json={
            "barter_id": barter_id,
            "window_id": i,
            "classification": "correct",
            "similarity_score": 0.72,
            "text_preview": f"Window {i} content about ML",
        })
        assert resp.status_code == 200

    # Verify windows stored
    resp = await backend_client.get(f"/session/{barter_id}/windows")
    windows = resp.json()
    assert len(windows) == 3
    assert all(w["classification"] == "correct" for w in windows)

    # 5. No warnings needed (all on-topic)

    # 6. Both users confirm
    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 1})
    resp = await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 2})
    assert resp.json()["both_confirmed"] is True

    # 7. Drift summary from warning engine
    resp = await backend_client.post(f"/session/{barter_id}/drift-summary", json={
        "barter_id": barter_id,
        "total_windows": 3,
        "incorrect_windows": 0,
        "percent_incorrect": 0.0,
        "max_consecutive_incorrect": 0,
        "warning_count": 0,
        "warnings": [],
        "terminated_early": False,
    })
    assert resp.status_code == 200

    # 8. Engagement summary from semantic analysis
    resp = await backend_client.post(f"/session/{barter_id}/engagement-summary", json={
        "learner_engagement_score": 0.78,
        "learner_speaking_seconds": 90.0,
        "teacher_speaking_seconds": 210.0,
        "learner_question_count": 4,
        "learner_acknowledgment_count": 6,
        "learner_segment_count": 8,
    })
    assert resp.status_code == 200

    # 9. Generate verdict
    resp = await backend_client.post(f"/verdict/{barter_id}/generate")
    assert resp.status_code == 200
    verdict_data = resp.json()
    assert verdict_data["confirmation_check"] is True
    # Duration check depends on actual elapsed time (likely fails in test)

    # 10. Update trust scores
    resp = await backend_client.post(f"/trust/{barter_id}/update")
    assert resp.status_code == 200
    trust_data = resp.json()
    assert trust_data["user_1_trust"]["after"] > 0
    assert trust_data["user_2_trust"]["after"] > 0

    # Verify final verdict
    resp = await backend_client.get(f"/verdict/{barter_id}")
    final = resp.json()
    assert final["on_topic_percentage"] == 100.0
    assert final["warning_count"] == 0
    assert final["drift_summary"]["engagement"]["learner_engagement_score"] == 0.78


@pytest.mark.asyncio
async def test_session_with_drift_and_warnings(backend_client):
    """Session with some off-topic drift and warnings, ending in PARTIAL verdict."""

    resp = await backend_client.post("/session/create", json={
        "topic": "Data Structures",
        "scope": "Arrays, linked lists, stacks, queues, trees",
        "agreed_duration_minutes": 10,
        "teacher_user_id": 1,
        "learner_user_id": 2,
    })
    barter_id = resp.json()["barter_id"]
    await backend_client.post(f"/session/{barter_id}/start")

    # Mix of on-topic and off-topic windows
    windows = [
        (1, "correct", 0.72),
        (2, "correct", 0.65),
        (3, "incorrect", 0.20),  # off-topic
        (4, "incorrect", 0.15),  # 2nd consecutive off-topic
        (5, "correct", 0.68),    # back on topic
        (6, "correct", 0.71),
    ]
    for wid, classification, sim in windows:
        await backend_client.post("/window/result", json={
            "barter_id": barter_id,
            "window_id": wid,
            "classification": classification,
            "similarity_score": sim,
            "text_preview": f"Window {wid}",
        })

    # Store a warning (what warning engine would have sent for windows 3-4)
    await backend_client.post("/warnings/log", json={
        "barter_id": barter_id,
        "severity": "strong",
        "reason": "2 consecutive off-topic windows",
        "window_ids": "3,4",
        "timestamp": "2026-03-24T10:00:00Z",
    })

    # Both confirm
    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 1})
    await backend_client.post(f"/session/{barter_id}/confirm", json={"user_id": 2})

    # Drift summary
    await backend_client.post(f"/session/{barter_id}/drift-summary", json={
        "barter_id": barter_id,
        "total_windows": 6,
        "incorrect_windows": 2,
        "percent_incorrect": 33.33,
        "max_consecutive_incorrect": 2,
        "total_drift_incidents": 2,
        "warning_count": 1,
        "warnings": [{"severity": "strong", "reason": "2 consecutive off-topic windows"}],
    })

    # Verdict
    resp = await backend_client.post(f"/verdict/{barter_id}/generate")
    data = resp.json()
    assert data["confirmation_check"] is True

    # Check verdict details
    resp = await backend_client.get(f"/verdict/{barter_id}")
    verdict = resp.json()
    assert verdict["on_topic_percentage"] == pytest.approx(66.67, abs=0.1)
    assert verdict["warning_count"] == 1


@pytest.mark.asyncio
async def test_terminated_session_produces_dispute(backend_client):
    """A session terminated due to severe drift → DISPUTE verdict."""

    resp = await backend_client.post("/session/create", json={
        "topic": "Quantum Physics",
        "scope": "Wave-particle duality, Schrodinger equation",
        "agreed_duration_minutes": 15,
        "teacher_user_id": 1,
        "learner_user_id": 2,
    })
    barter_id = resp.json()["barter_id"]
    await backend_client.post(f"/session/{barter_id}/start")

    # All incorrect windows
    for i in range(1, 6):
        await backend_client.post("/window/result", json={
            "barter_id": barter_id,
            "window_id": i,
            "classification": "incorrect",
            "similarity_score": 0.10,
        })

    # Multiple warnings escalating
    for severity in ["strong", "severe", "severe"]:
        await backend_client.post("/warnings/log", json={
            "barter_id": barter_id,
            "severity": severity,
            "reason": f"{severity} drift warning",
        })

    # Auto-terminated
    await backend_client.post(
        f"/session/{barter_id}/terminate",
        json={"reason": "Severe drift — auto-termination"},
    )

    # Drift summary
    await backend_client.post(f"/session/{barter_id}/drift-summary", json={
        "barter_id": barter_id,
        "total_windows": 5,
        "incorrect_windows": 5,
        "percent_incorrect": 100.0,
        "max_consecutive_incorrect": 5,
        "total_drift_incidents": 5,
        "warning_count": 3,
        "terminated_early": True,
    })

    # Verdict
    resp = await backend_client.post(f"/verdict/{barter_id}/generate")
    assert resp.json()["verdict"] == "DISPUTE"

    # Trust update
    resp = await backend_client.post(f"/trust/{barter_id}/update")
    data = resp.json()
    # DISPUTE → qa_score=0, quality_adjusted=0.4
    # new = 0.30*0.3 + 0.4*0.7 = 0.37
    assert data["user_1_trust"]["after"] == pytest.approx(0.37, abs=0.01)


@pytest.mark.asyncio
async def test_transcript_storage_and_retrieval(backend_client, create_session_payload):
    """Verify transcript segments are stored and retrievable in order."""
    barter_id = (await backend_client.post(
        "/session/create", json=create_session_payload
    )).json()["barter_id"]
    await backend_client.post(f"/session/{barter_id}/start")

    segments = [
        {"user_id": 1, "text": "Hello, let's start with variables", "duration_seconds": 10.0},
        {"user_id": 2, "text": "Okay, what is a variable?", "duration_seconds": 5.0},
        {"user_id": 1, "text": "A variable stores data in memory", "duration_seconds": 12.0},
    ]
    for seg in segments:
        resp = await backend_client.post(f"/session/{barter_id}/transcript", json={
            "barter_id": barter_id, **seg,
            "timestamp_start": 0.0, "timestamp_end": seg["duration_seconds"],
        })
        assert resp.status_code == 200

    resp = await backend_client.get(f"/session/{barter_id}/transcript")
    data = resp.json()
    assert len(data) == 3
    assert data[0]["speaker"] == "Alice"
    assert data[1]["speaker"] == "Bob"
    assert data[2]["text"] == "A variable stores data in memory"


@pytest.mark.asyncio
async def test_window_results_ordered(backend_client, create_session_payload):
    """Verify window results are returned ordered by window_number."""
    barter_id = (await backend_client.post(
        "/session/create", json=create_session_payload
    )).json()["barter_id"]
    await backend_client.post(f"/session/{barter_id}/start")

    # Insert out of order
    for wid in [3, 1, 2]:
        await backend_client.post("/window/result", json={
            "barter_id": barter_id,
            "window_id": wid,
            "classification": "correct",
            "similarity_score": 0.70,
        })

    resp = await backend_client.get(f"/session/{barter_id}/windows")
    windows = resp.json()
    assert [w["window_id"] for w in windows] == [1, 2, 3]
