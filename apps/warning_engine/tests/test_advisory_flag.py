import pytest


@pytest.mark.asyncio
async def test_severe_warning_posts_advisory_true(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    for window_id in (1, 2, 3):
        await client.post("/window/result", json={
            "barter_id": 1, "window_id": window_id, "classification": "incorrect",
            "similarity_score": 0.1, "text_preview": "off topic",
            "timestamp_start": 0.0, "timestamp_end": 5.0,
        })

    warning_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/warnings/log")]
    assert len(warning_calls) == 2  # 2nd window (strong) and 3rd window (severe)
    payload = warning_calls[-1].kwargs["json"]
    assert payload["severity"] == "severe"
    assert payload["advisory"] is True


@pytest.mark.asyncio
async def test_strong_warning_is_not_advisory(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    for window_id in (1, 2):
        await client.post("/window/result", json={
            "barter_id": 1, "window_id": window_id, "classification": "incorrect",
            "similarity_score": 0.1, "text_preview": "off topic",
            "timestamp_start": 0.0, "timestamp_end": 5.0,
        })

    warning_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/warnings/log")]
    assert len(warning_calls) == 1
    payload = warning_calls[0].kwargs["json"]
    assert payload["severity"] == "strong"
    assert payload["advisory"] is False


@pytest.mark.asyncio
async def test_ladder_thresholds_are_unchanged(warning_client):
    # Spec F5: adding `advisory` must not shift when a warning fires.
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    def window(window_id):
        return {
            "barter_id": 1, "window_id": window_id, "classification": "incorrect",
            "similarity_score": 0.1, "text_preview": "off topic",
            "timestamp_start": 0.0, "timestamp_end": 5.0,
        }

    resp1 = await client.post("/window/result", json=window(1))
    assert resp1.json()["action"] == "silent"

    resp2 = await client.post("/window/result", json=window(2))
    body2 = resp2.json()
    assert body2["action"] == "warning"
    assert body2["severity"] == "strong"

    resp3 = await client.post("/window/result", json=window(3))
    body3 = resp3.json()
    assert body3["action"] == "warning"
    assert body3["severity"] == "severe"
