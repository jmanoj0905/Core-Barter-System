"""
E2E tests for the warning engine:
  - Window result processing
  - Warning escalation: 1 off-topic = silent, 2 = strong, 3+ = severe
  - On-topic resets consecutive counter
  - Safety alerts (toxicity/NSFW)
  - Engagement alerts
  - Session end with drift summary
"""

import pytest


@pytest.mark.asyncio
async def test_init_session(warning_client):
    client, mock_http = warning_client
    resp = await client.post(
        "/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "initialized"


@pytest.mark.asyncio
async def test_init_session_idempotent(warning_client):
    client, _ = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})
    resp = await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})
    assert resp.json()["status"] == "already_initialized"


@pytest.mark.asyncio
async def test_on_topic_window_no_warning(warning_client):
    client, mock_http = warning_client
    await client.post("/session/1/init")

    resp = await client.post("/window/result", json={
        "barter_id": 1,
        "window_id": 1,
        "classification": "correct",
        "similarity_score": 0.75,
        "text_preview": "Python variables and loops",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "none"
    assert data["consecutive_incorrect"] == 0


@pytest.mark.asyncio
async def test_single_off_topic_is_silent(warning_client):
    client, _ = warning_client
    await client.post("/session/1/init")

    resp = await client.post("/window/result", json={
        "barter_id": 1,
        "window_id": 1,
        "classification": "incorrect",
        "similarity_score": 0.15,
        "text_preview": "Talking about cooking recipes",
    })
    data = resp.json()
    assert data["action"] == "silent"
    assert data["consecutive_incorrect"] == 1


@pytest.mark.asyncio
async def test_two_consecutive_off_topic_strong_warning(warning_client):
    client, mock_http = warning_client
    await client.post("/session/1/init")

    # First off-topic → silent
    await client.post("/window/result", json={
        "barter_id": 1, "window_id": 1,
        "classification": "incorrect", "similarity_score": 0.15,
    })

    # Second off-topic → strong
    resp = await client.post("/window/result", json={
        "barter_id": 1, "window_id": 2,
        "classification": "incorrect", "similarity_score": 0.10,
    })
    data = resp.json()
    assert data["action"] == "warning"
    assert data["severity"] == "strong"
    assert data["consecutive_incorrect"] == 2


@pytest.mark.asyncio
async def test_three_consecutive_off_topic_severe_warning(warning_client):
    client, mock_http = warning_client
    await client.post("/session/1/init")

    for i in range(1, 3):
        await client.post("/window/result", json={
            "barter_id": 1, "window_id": i,
            "classification": "incorrect", "similarity_score": 0.10,
        })

    # Third → severe
    resp = await client.post("/window/result", json={
        "barter_id": 1, "window_id": 3,
        "classification": "incorrect", "similarity_score": 0.08,
    })
    data = resp.json()
    assert data["action"] == "warning"
    assert data["severity"] == "severe"
    assert data["consecutive_incorrect"] == 3


@pytest.mark.asyncio
async def test_on_topic_resets_consecutive_counter(warning_client):
    client, _ = warning_client
    await client.post("/session/1/init")

    # One off-topic
    await client.post("/window/result", json={
        "barter_id": 1, "window_id": 1,
        "classification": "incorrect", "similarity_score": 0.10,
    })

    # Back on topic
    resp = await client.post("/window/result", json={
        "barter_id": 1, "window_id": 2,
        "classification": "correct", "similarity_score": 0.70,
    })
    assert resp.json()["consecutive_incorrect"] == 0

    # Next off-topic should be silent again (counter reset)
    resp = await client.post("/window/result", json={
        "barter_id": 1, "window_id": 3,
        "classification": "incorrect", "similarity_score": 0.12,
    })
    assert resp.json()["action"] == "silent"
    assert resp.json()["consecutive_incorrect"] == 1


@pytest.mark.asyncio
async def test_weakly_correct_resets_counter(warning_client):
    client, _ = warning_client
    await client.post("/session/1/init")

    await client.post("/window/result", json={
        "barter_id": 1, "window_id": 1,
        "classification": "incorrect", "similarity_score": 0.10,
    })

    resp = await client.post("/window/result", json={
        "barter_id": 1, "window_id": 2,
        "classification": "weakly_correct", "similarity_score": 0.45,
    })
    assert resp.json()["consecutive_incorrect"] == 0


@pytest.mark.asyncio
async def test_safety_alert(warning_client):
    client, mock_http = warning_client
    await client.post("/session/1/init")

    resp = await client.post("/safety/alert", json={
        "barter_id": 1,
        "user_id": 1,
        "warning_type": "toxicity",
        "details": {"categories": {"hate": 0.85}, "hard_block": False},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "safety_warning"
    assert data["severity"] == "strong"


@pytest.mark.asyncio
async def test_safety_alert_hard_block_is_severe(warning_client):
    client, _ = warning_client
    await client.post("/session/1/init")

    resp = await client.post("/safety/alert", json={
        "barter_id": 1,
        "user_id": 1,
        "warning_type": "nsfw",
        "details": {"categories": {"sexual": 0.95}, "hard_block": True},
    })
    assert resp.json()["severity"] == "severe"


@pytest.mark.asyncio
async def test_engagement_alert(warning_client):
    client, mock_http = warning_client
    await client.post("/session/1/init")

    resp = await client.post("/engagement/alert", json={
        "barter_id": 1,
        "alert_type": "low_engagement",
        "engagement_score": 0.15,
    })
    assert resp.status_code == 200
    assert resp.json()["action"] == "engagement_alert"


@pytest.mark.asyncio
async def test_session_end_returns_drift_summary(warning_client):
    client, mock_http = warning_client
    await client.post("/session/1/init")

    # Send 3 windows: 2 correct, 1 incorrect
    await client.post("/window/result", json={
        "barter_id": 1, "window_id": 1,
        "classification": "correct", "similarity_score": 0.70,
    })
    await client.post("/window/result", json={
        "barter_id": 1, "window_id": 2,
        "classification": "incorrect", "similarity_score": 0.15,
    })
    await client.post("/window/result", json={
        "barter_id": 1, "window_id": 3,
        "classification": "correct", "similarity_score": 0.65,
    })

    resp = await client.post("/session/1/end")
    assert resp.status_code == 200
    data = resp.json()
    summary = data["drift_summary"]
    assert summary["total_windows"] == 3
    assert summary["incorrect_windows"] == 1
    assert summary["max_consecutive_incorrect"] == 1
    assert summary["percent_incorrect"] == pytest.approx(33.33, abs=0.1)


@pytest.mark.asyncio
async def test_auto_init_on_unknown_session(warning_client):
    """Warning engine should auto-initialize if it gets a window for an unknown session."""
    client, _ = warning_client

    resp = await client.post("/window/result", json={
        "barter_id": 99, "window_id": 1,
        "classification": "correct", "similarity_score": 0.70,
    })
    assert resp.status_code == 200
    assert resp.json()["action"] == "none"
