import pytest


@pytest.mark.asyncio
async def test_learner_segment_posts_engagement_update(semantic_client):
    client, mock_http, sa_main = semantic_client

    resp = await client.post("/ingest/segment", json={
        "barter_id": 1,
        "user_id": 2,  # learner
        "text": "That makes sense, can you explain more?",
        "duration_seconds": 35.0,
        "timestamp_start": 0.0,
        "timestamp_end": 35.0,
    })
    assert resp.status_code == 200

    update_calls = [
        c for c in mock_http.post.call_args_list
        if c.args[0].endswith("/engagement/update")
    ]
    assert len(update_calls) == 1
    payload = update_calls[0].kwargs["json"]
    assert payload["barter_id"] == 1
    assert payload["user_id"] == 2
    assert isinstance(payload["engagement_score"], float)
