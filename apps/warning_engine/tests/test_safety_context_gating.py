import pytest


async def _init_and_classify(client, barter_id, classification):
    await client.post(f"/session/{barter_id}/init", json={"teacher_user_id": 1, "learner_user_id": 2})
    await client.post("/window/result", json={
        "barter_id": barter_id, "window_id": 1, "classification": classification,
        "similarity_score": 0.5, "text_preview": "some text",
        "timestamp_start": 0.0, "timestamp_end": 25.0,
    })


@pytest.mark.asyncio
async def test_hard_block_downgraded_when_last_window_correct(warning_client):
    client, mock_http, we_main = warning_client
    await _init_and_classify(client, 1, "correct")

    resp = await client.post("/safety/alert", json={
        "barter_id": 1, "user_id": 2, "warning_type": "toxicity",
        "details": {"hard_block": True, "categories": {"hate": 0.95}},
    })
    assert resp.status_code == 200
    assert resp.json()["severity"] == "strong"


@pytest.mark.asyncio
async def test_hard_block_downgraded_when_last_window_weakly_correct(warning_client):
    client, mock_http, we_main = warning_client
    await _init_and_classify(client, 1, "weakly_correct")

    resp = await client.post("/safety/alert", json={
        "barter_id": 1, "user_id": 2, "warning_type": "toxicity",
        "details": {"hard_block": True, "categories": {"hate": 0.95}},
    })
    assert resp.json()["severity"] == "strong"


@pytest.mark.asyncio
async def test_hard_block_stays_severe_when_last_window_incorrect(warning_client):
    client, mock_http, we_main = warning_client
    await _init_and_classify(client, 1, "incorrect")

    resp = await client.post("/safety/alert", json={
        "barter_id": 1, "user_id": 2, "warning_type": "toxicity",
        "details": {"hard_block": True, "categories": {"hate": 0.95}},
    })
    assert resp.json()["severity"] == "severe"


@pytest.mark.asyncio
async def test_hard_block_stays_severe_when_no_window_classified_yet(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    resp = await client.post("/safety/alert", json={
        "barter_id": 1, "user_id": 2, "warning_type": "toxicity",
        "details": {"hard_block": True, "categories": {"hate": 0.95}},
    })
    assert resp.json()["severity"] == "severe"


@pytest.mark.asyncio
async def test_hard_block_stays_severe_when_session_never_initialized(warning_client):
    client, mock_http, we_main = warning_client

    resp = await client.post("/safety/alert", json={
        "barter_id": 999, "user_id": 2, "warning_type": "toxicity",
        "details": {"hard_block": True, "categories": {"hate": 0.95}},
    })
    assert resp.status_code == 200
    assert resp.json()["severity"] == "severe"


@pytest.mark.asyncio
async def test_non_hard_block_stays_strong_regardless_of_classification(warning_client):
    client, mock_http, we_main = warning_client
    await _init_and_classify(client, 1, "correct")

    resp = await client.post("/safety/alert", json={
        "barter_id": 1, "user_id": 2, "warning_type": "toxicity",
        "details": {"hard_block": False, "categories": {"hate": 0.7}},
    })
    assert resp.json()["severity"] == "strong"


@pytest.mark.asyncio
async def test_last_classification_updates_between_alerts(warning_client):
    client, mock_http, we_main = warning_client
    await _init_and_classify(client, 1, "incorrect")

    resp1 = await client.post("/safety/alert", json={
        "barter_id": 1, "user_id": 2, "warning_type": "toxicity",
        "details": {"hard_block": True, "categories": {"hate": 0.95}},
    })
    assert resp1.json()["severity"] == "severe"

    await client.post("/window/result", json={
        "barter_id": 1, "window_id": 2, "classification": "correct",
        "similarity_score": 0.5, "text_preview": "back on topic",
        "timestamp_start": 25.0, "timestamp_end": 50.0,
    })

    resp2 = await client.post("/safety/alert", json={
        "barter_id": 1, "user_id": 2, "warning_type": "toxicity",
        "details": {"hard_block": True, "categories": {"hate": 0.95}},
    })
    assert resp2.json()["severity"] == "strong"
