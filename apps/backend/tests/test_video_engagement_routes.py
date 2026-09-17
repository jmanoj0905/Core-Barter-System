import pytest
from app.models import VideoEngagementResult, EngagementScoreLog


@pytest.mark.asyncio
async def test_video_engagement_result_model_roundtrip(backend_client, db_session):
    row = VideoEngagementResult(
        barter_session_id=1,
        user_id=2,
        window_start=0.0,
        window_end=5.0,
        video_attention_score=0.72,
        backend_used="local",
        raw_signals='{"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.8}',
    )
    db_session.add(row)
    await db_session.commit()
    assert row.id is not None


@pytest.mark.asyncio
async def test_engagement_score_log_model_roundtrip(backend_client, db_session):
    row = EngagementScoreLog(
        barter_session_id=1,
        user_id=2,
        speech_engagement_score=0.6,
        video_attention_score=0.72,
        fused_engagement_score=0.65,
    )
    db_session.add(row)
    await db_session.commit()
    assert row.id is not None


@pytest.mark.asyncio
async def test_post_and_list_video_engagement(backend_client):
    resp = await backend_client.post("/session/1/video-engagement", json={
        "user_id": 2,
        "window_start": 0.0,
        "window_end": 5.0,
        "video_attention_score": 0.72,
        "backend_used": "local",
        "raw_signals": {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.8},
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "stored"

    resp = await backend_client.get("/session/1/video-engagement")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["video_attention_score"] == 0.72
    assert rows[0]["raw_signals"] == {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.8}


@pytest.mark.asyncio
async def test_post_engagement_log_and_get_latest(backend_client):
    await backend_client.post("/session/1/engagement-log", json={
        "user_id": 2,
        "speech_engagement_score": 0.6,
        "video_attention_score": None,
        "fused_engagement_score": 0.6,
    })
    resp = await backend_client.post("/session/1/engagement-log", json={
        "user_id": 2,
        "speech_engagement_score": 0.6,
        "video_attention_score": 0.72,
        "fused_engagement_score": 0.65,
    })
    assert resp.status_code == 200

    resp = await backend_client.get("/session/1/engagement")
    assert resp.status_code == 200
    latest = resp.json()
    assert latest["fused_engagement_score"] == 0.65

    resp = await backend_client.get("/session/1/engagement/history")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_get_engagement_no_data_returns_404(backend_client):
    resp = await backend_client.get("/session/999/engagement")
    assert resp.status_code == 404
