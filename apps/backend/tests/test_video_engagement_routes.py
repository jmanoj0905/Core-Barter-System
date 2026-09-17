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
