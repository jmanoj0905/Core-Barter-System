import time

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


@pytest.mark.asyncio
async def test_video_update_ignored_for_non_learner(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    resp = await client.post("/video-engagement/update", json={
        "barter_id": 1, "user_id": 1, "video_attention_score": 0.9,  # user 1 is the teacher
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    assert we_main.sessions[1]["video_attention_score"] is None


@pytest.mark.asyncio
async def test_fusion_combines_speech_and_video_for_learner(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})
    await client.post("/engagement/update", json={
        "barter_id": 1, "user_id": 2, "engagement_score": 0.6,
    })
    resp = await client.post("/video-engagement/update", json={
        "barter_id": 1, "user_id": 2, "video_attention_score": 0.8,
    })
    assert resp.status_code == 200

    # w_speech=0.7, w_video=0.3 defaults -> 0.7*0.6 + 0.3*0.8 = 0.66
    logged_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/engagement-log")]
    assert len(logged_calls) == 2  # one log on the speech-only update, one on the combined update
    payload = logged_calls[-1].kwargs["json"]
    assert payload["fused_engagement_score"] == pytest.approx(0.66, abs=1e-6)
    assert payload["speech_engagement_score"] == 0.6
    assert payload["video_attention_score"] == 0.8


@pytest.mark.asyncio
async def test_fusion_falls_back_to_speech_only_when_no_video(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})
    await client.post("/engagement/update", json={
        "barter_id": 1, "user_id": 2, "engagement_score": 0.55,
    })

    logged_call = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/engagement-log")]
    assert len(logged_call) == 1
    assert logged_call[0].kwargs["json"]["fused_engagement_score"] == 0.55
    assert logged_call[0].kwargs["json"]["video_attention_score"] is None


@pytest.mark.asyncio
async def test_fusion_excludes_stale_video_score(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    await client.post("/video-engagement/update", json={
        "barter_id": 1, "user_id": 2, "video_attention_score": 0.9,
    })

    # Backdate the video score's timestamp past the staleness window.
    we_main.sessions[1]["video_attention_score_at"] = time.time() - 20

    resp = await client.post("/engagement/update", json={
        "barter_id": 1, "user_id": 2, "engagement_score": 0.5,
    })
    assert resp.status_code == 200

    logged_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/engagement-log")]
    payload = logged_calls[-1].kwargs["json"]
    # Stale video score must be excluded — fused equals speech alone, not a blend.
    assert payload["fused_engagement_score"] == 0.5
