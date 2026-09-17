import pytest


@pytest.mark.asyncio
async def test_engagement_update_stores_speech_score(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    resp = await client.post("/engagement/update", json={
        "barter_id": 1, "user_id": 2, "engagement_score": 0.6,
    })
    assert resp.status_code == 200
    assert we_main.sessions[1]["speech_engagement_score"] == 0.6


@pytest.mark.asyncio
async def test_engagement_update_auto_inits_session(warning_client):
    client, mock_http, we_main = warning_client
    resp = await client.post("/engagement/update", json={
        "barter_id": 42, "user_id": 2, "engagement_score": 0.4,
    })
    assert resp.status_code == 200
    assert 42 in we_main.sessions
