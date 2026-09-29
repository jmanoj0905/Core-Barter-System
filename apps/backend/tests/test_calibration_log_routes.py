import pytest
from app.models import CalibrationLog


@pytest.mark.asyncio
async def test_calibration_log_model_roundtrip(backend_client, db_session):
    row = CalibrationLog(
        barter_session_id=1, user_id=2, outcome="calibrated",
        old_baseline_head_ratio=None, old_baseline_gaze_ratio=None,
        new_baseline_head_ratio=0.51, new_baseline_gaze_ratio=0.48,
        recalibration_count=0,
    )
    db_session.add(row)
    await db_session.commit()
    assert row.id is not None


@pytest.mark.asyncio
async def test_post_and_list_calibration_log(backend_client):
    resp = await backend_client.post("/session/1/video-engagement/calibration-log", json={
        "user_id": 2, "outcome": "calibrated",
        "old_baseline_head_ratio": None, "old_baseline_gaze_ratio": None,
        "new_baseline_head_ratio": 0.51, "new_baseline_gaze_ratio": 0.48,
        "recalibration_count": 0,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "stored"
    assert isinstance(resp.json()["id"], int)

    resp = await backend_client.post("/session/1/video-engagement/calibration-log", json={
        "user_id": 2, "outcome": "recalibration_limit_reached",
        "old_baseline_head_ratio": 0.51, "old_baseline_gaze_ratio": 0.48,
        "new_baseline_head_ratio": None, "new_baseline_gaze_ratio": None,
        "recalibration_count": 0,
    })
    assert resp.status_code == 200

    resp = await backend_client.get("/session/1/video-engagement/calibration-log")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert rows[0]["outcome"] == "calibrated"
    assert rows[0]["new_baseline_head_ratio"] == 0.51
    assert rows[0]["old_baseline_head_ratio"] is None
    assert rows[1]["outcome"] == "recalibration_limit_reached"
    assert rows[1]["old_baseline_gaze_ratio"] == 0.48
    assert rows[1]["created_at"]


@pytest.mark.asyncio
async def test_list_calibration_log_scoped_to_session(backend_client):
    await backend_client.post("/session/1/video-engagement/calibration-log", json={
        "user_id": 2, "outcome": "insufficient_face_detections", "recalibration_count": 0,
    })
    resp = await backend_client.get("/session/99/video-engagement/calibration-log")
    assert resp.status_code == 200
    assert resp.json() == []
