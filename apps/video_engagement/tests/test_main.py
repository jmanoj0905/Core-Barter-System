from fastapi.testclient import TestClient
from main import app


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_root():
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.json()["service"] == "video-engagement"


from unittest.mock import MagicMock, patch

import numpy as np
import cv2

import main


def _make_jpeg_bytes(width=100, height=100) -> bytes:
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_process_frame_local_no_face_returns_none():
    # A blank grey image has no face -> MediaPipe finds nothing.
    frame_bytes = _make_jpeg_bytes()
    result = main.process_frame_local(frame_bytes)
    assert result is None


def test_process_frame_aws_no_face_returns_none():
    frame_bytes = _make_jpeg_bytes()
    fake_client = MagicMock()
    fake_client.detect_faces.return_value = {"FaceDetails": []}
    with patch.object(main, "_rekognition_client", return_value=fake_client):
        result = main.process_frame_aws(frame_bytes)
    assert result is None


def test_process_frame_aws_maps_face_detail():
    frame_bytes = _make_jpeg_bytes()
    fake_client = MagicMock()
    fake_client.detect_faces.return_value = {
        "FaceDetails": [{
            "EyesOpen": {"Value": True, "Confidence": 99.0},
            "Pose": {"Yaw": 2.0, "Pitch": 1.0, "Roll": 0.0},
        }]
    }
    with patch.object(main, "_rekognition_client", return_value=fake_client):
        result = main.process_frame_aws(frame_bytes)
    assert result is not None
    assert result["eyes_open"] == 1.0


def test_process_frame_aws_fails_open_on_exception():
    frame_bytes = _make_jpeg_bytes()
    fake_client = MagicMock()
    fake_client.detect_faces.side_effect = Exception("throttled")
    with patch.object(main, "_rekognition_client", return_value=fake_client):
        result = main.process_frame_aws(frame_bytes)
    assert result is None


import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest


def test_video_config_get_default():
    with TestClient(app) as client:
        resp = client.get("/video/config")
        assert resp.status_code == 200
        assert resp.json()["backend"] == "local"
        assert set(resp.json()["available"]) == {"local", "aws", "both"}


def test_video_config_post_invalid_backend_rejected():
    with TestClient(app) as client:
        resp = client.post("/video/config", json={"backend": "gcp"})
        assert resp.status_code == 400


def test_video_config_post_switches_backend():
    with TestClient(app) as client:
        resp = client.post("/video/config", json={"backend": "aws"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "aws"
        main.current_video_backend = "local"  # reset for other tests


@pytest.mark.asyncio
async def test_process_buffer_local_posts_result_when_face_found():
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}
    main.http_client = AsyncMock()

    with patch.object(main, "process_frame_local", return_value={
        "eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.85,
    }):
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="local")

    posted_paths = [c.args[0] for c in main.http_client.post.call_args_list]
    assert any("/video-engagement" in p for p in posted_paths)
    assert any("/video-engagement/update" in p for p in posted_paths)


@pytest.mark.asyncio
async def test_process_buffer_skips_window_when_no_face_detected():
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}
    main.http_client = AsyncMock()

    with patch.object(main, "process_frame_local", return_value=None):
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="local")

    assert main.http_client.post.call_count == 0


@pytest.mark.asyncio
async def test_process_buffer_both_mode_posts_twice_with_different_backend_tag():
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}
    main.http_client = AsyncMock()

    with patch.object(main, "process_frame_local", return_value={
        "eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.85,
    }), patch.object(main, "process_frame_aws", return_value={
        "eyes_open": 0.8, "head_deviation": 0.2, "gaze_centered": 0.75,
    }):
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="both")

    backend_store_calls = [
        c.kwargs["json"]["backend_used"]
        for c in main.http_client.post.call_args_list
        if c.args[0].endswith("/video-engagement")
    ]
    assert sorted(backend_store_calls) == ["aws", "local"]


@pytest.mark.asyncio
async def test_process_buffer_local_raises_aws_still_posts_in_both_mode():
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}
    main.http_client = AsyncMock()

    with patch.object(main, "process_frame_local", side_effect=RuntimeError("boom")), \
         patch.object(main, "process_frame_aws", return_value={
             "eyes_open": 0.8, "head_deviation": 0.2, "gaze_centered": 0.75,
         }):
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="both")

    backend_store_calls = [
        c.kwargs["json"]["backend_used"]
        for c in main.http_client.post.call_args_list
        if c.args[0].endswith("/video-engagement")
    ]
    assert backend_store_calls == ["aws"]


# --- calibration -----------------------------------------------------------

def _calibrate_files(n=3):
    return [("frames", ("f.jpg", _make_jpeg_bytes(), "image/jpeg"))] * n


def _stored(last_calibrated_at, count=0):
    return {"head_ratio": 0.5, "gaze_ratio": 0.5, "recalibration_count": count,
            "last_calibrated_at": last_calibrated_at}


def _post_calibrate(user_id=2, n=3, http_post_side_effect=None):
    """POST to /calibrate inside the TestClient context so the lifespan-created
    http_client is replaced by a mock *after* startup. Returns (response, mock)."""
    with TestClient(main.app) as client:
        main.http_client = AsyncMock()
        if http_post_side_effect is not None:
            main.http_client.post.side_effect = http_post_side_effect
        resp = client.post(f"/video/1/{user_id}/calibrate", files=_calibrate_files(n))
        return resp, main.http_client


def test_calibrate_success_stores_baseline_and_logs_audit():
    main.calibration_baselines.clear()
    with patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch.object(main.scoring, "raw_ratios_from_landmarks",
                      return_value={"head_ratio": 0.51, "gaze_ratio": 0.48}):
        resp, mock = _post_calibrate()
    assert resp.status_code == 200
    assert resp.json() == {
        "calibrated": True, "baseline_head_ratio": 0.51, "baseline_gaze_ratio": 0.48,
        "frames_used": 3, "recalibration_count": 0,
    }
    assert main.calibration_baselines[(1, 2)]["head_ratio"] == 0.51
    log_calls = [c for c in mock.post.call_args_list if "/calibration-log" in c.args[0]]
    assert len(log_calls) == 1
    assert log_calls[0].args[0].endswith("/session/1/video-engagement/calibration-log")
    assert log_calls[0].kwargs["json"] == {
        "user_id": 2, "outcome": "calibrated",
        "old_baseline_head_ratio": None, "old_baseline_gaze_ratio": None,
        "new_baseline_head_ratio": 0.51, "new_baseline_gaze_ratio": 0.48,
        "recalibration_count": 0,
    }
    main.calibration_baselines.clear()


def test_calibrate_insufficient_faces_falls_back():
    main.calibration_baselines.clear()
    with patch.object(main, "_landmarks_from_frame_bytes", return_value=None):
        resp, mock = _post_calibrate(n=2)
    assert resp.json() == {"calibrated": False, "reason": "insufficient_face_detections"}
    assert (1, 2) not in main.calibration_baselines
    log = [c for c in mock.post.call_args_list if "/calibration-log" in c.args[0]]
    assert log[0].kwargs["json"]["outcome"] == "insufficient_face_detections"


def test_calibrate_insufficient_faces_keeps_existing_baseline():
    main.calibration_baselines.clear()
    old = _stored(time.time() - main.CALIBRATION_COOLDOWN_SECONDS - 5)
    main.calibration_baselines[(1, 2)] = old
    with patch.object(main, "_landmarks_from_frame_bytes", return_value=None):
        resp, _ = _post_calibrate()
    assert resp.json() == {"calibrated": False, "reason": "insufficient_face_detections"}
    assert main.calibration_baselines[(1, 2)] is old
    main.calibration_baselines.clear()


def test_calibrate_degenerate_head_ratio_frames_not_counted():
    main.calibration_baselines.clear()
    with patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch.object(main.scoring, "raw_ratios_from_landmarks",
                      return_value={"head_ratio": None, "gaze_ratio": 0.5}):
        resp, _ = _post_calibrate()
    assert resp.json() == {"calibrated": False, "reason": "insufficient_face_detections"}


def test_calibrate_within_cooldown_is_rejected():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = _stored(time.time())
    resp, mock = _post_calibrate()
    body = resp.json()
    assert body["calibrated"] is False
    assert body["reason"] == "recalibration_limit_reached"
    assert body["retry_after_seconds"] > 0
    log = [c for c in mock.post.call_args_list if "/calibration-log" in c.args[0]]
    assert log[0].kwargs["json"]["outcome"] == "recalibration_limit_reached"
    assert log[0].kwargs["json"]["old_baseline_head_ratio"] == 0.5
    main.calibration_baselines.clear()


def test_calibrate_cooldown_boundary_exact_120_succeeds():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = _stored(1000.0)
    with patch.object(main.time, "time", return_value=1000.0 + main.CALIBRATION_COOLDOWN_SECONDS), \
         patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch.object(main.scoring, "raw_ratios_from_landmarks",
                      return_value={"head_ratio": 0.6, "gaze_ratio": 0.4}):
        resp, _ = _post_calibrate()
    assert resp.json()["calibrated"] is True
    assert resp.json()["recalibration_count"] == 1
    main.calibration_baselines.clear()


def test_calibrate_cooldown_boundary_just_under_120_rejected():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = _stored(1000.0)
    with patch.object(main.time, "time", return_value=1000.0 + 119.9):
        resp, _ = _post_calibrate()
    body = resp.json()
    assert body["reason"] == "recalibration_limit_reached"
    assert body["retry_after_seconds"] > 0
    main.calibration_baselines.clear()


def test_calibrate_retry_after_never_zero_near_boundary():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = _stored(1000.0)
    with patch.object(main.time, "time", return_value=1000.0 + 119.96):
        resp, _ = _post_calibrate()
    assert resp.json()["retry_after_seconds"] > 0
    main.calibration_baselines.clear()


def test_calibrate_after_cooldown_elapsed_succeeds():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = _stored(
        time.time() - main.CALIBRATION_COOLDOWN_SECONDS - 1)
    with patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch.object(main.scoring, "raw_ratios_from_landmarks",
                      return_value={"head_ratio": 0.6, "gaze_ratio": 0.4}):
        resp, _ = _post_calibrate()
    body = resp.json()
    assert body["calibrated"] is True
    assert body["recalibration_count"] == 1
    assert main.calibration_baselines[(1, 2)]["head_ratio"] == 0.6
    main.calibration_baselines.clear()


def test_calibrate_two_users_independent():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = _stored(time.time())
    with patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch.object(main.scoring, "raw_ratios_from_landmarks",
                      return_value={"head_ratio": 0.55, "gaze_ratio": 0.45}):
        resp, _ = _post_calibrate(user_id=3)
    assert resp.json()["calibrated"] is True  # user 3 unaffected by user 2's cooldown
    assert main.calibration_baselines[(1, 2)]["head_ratio"] == 0.5
    main.calibration_baselines.clear()


def test_calibrate_backend_log_post_failure_does_not_break_response():
    main.calibration_baselines.clear()
    with patch.object(main, "_landmarks_from_frame_bytes", return_value={0: (0, 0)}), \
         patch.object(main.scoring, "raw_ratios_from_landmarks",
                      return_value={"head_ratio": 0.5, "gaze_ratio": 0.5}):
        resp, mock = _post_calibrate(http_post_side_effect=Exception("backend down"))
    assert resp.status_code == 200
    assert resp.json()["calibrated"] is True
    assert mock.post.call_count == 1  # the log POST was attempted
    main.calibration_baselines.clear()


@pytest.mark.asyncio
async def test_post_calibration_log_swallows_payload_construction_errors():
    main.http_client = AsyncMock()
    # old_baseline missing keys -> KeyError while building the payload; must not raise.
    await main._post_calibration_log(1, 2, "calibrated", {"bogus": 1}, None, 0)
    assert main.http_client.post.call_count == 0


@pytest.mark.asyncio
async def test_process_buffer_uses_stored_baseline():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = {
        "head_ratio": 0.6, "gaze_ratio": 0.4, "recalibration_count": 0,
        "last_calibrated_at": time.time(),
    }
    main.http_client = AsyncMock()
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}

    captured = {}

    def fake_process_frame_local(frame_bytes, baseline=None):
        captured["baseline"] = baseline
        return {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.85}

    with patch.object(main, "process_frame_local", side_effect=fake_process_frame_local):
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="local")

    assert captured["baseline"] == {"head_ratio": 0.6, "gaze_ratio": 0.4}
    main.calibration_baselines.clear()


@pytest.mark.asyncio
async def test_process_buffer_no_baseline_defaults_to_none():
    main.calibration_baselines.clear()
    main.http_client = AsyncMock()
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}

    captured = {}

    def fake_process_frame_local(frame_bytes, baseline=None):
        captured["baseline"] = baseline
        return {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.85}

    with patch.object(main, "process_frame_local", side_effect=fake_process_frame_local):
        await main.process_buffer(barter_id=1, user_id=99, buf=buf, backend="local")

    assert captured["baseline"] is None


@pytest.mark.asyncio
async def test_process_buffer_aws_path_unaffected_by_baseline():
    main.calibration_baselines.clear()
    main.calibration_baselines[(1, 2)] = _stored(time.time())
    main.http_client = AsyncMock()
    buf = {"frames": [b"fake-jpeg-bytes"], "wall_start": time.time()}
    with patch.object(main, "process_frame_aws", return_value={
        "eyes_open": 0.8, "head_deviation": 0.2, "gaze_centered": 0.75,
    }) as aws:
        await main.process_buffer(barter_id=1, user_id=2, buf=buf, backend="aws")
    aws.assert_called_once_with(b"fake-jpeg-bytes")
    assert main.http_client.post.call_count == 2
    main.calibration_baselines.clear()


@pytest.mark.asyncio
async def test_end_session_clears_calibration_baselines_for_that_barter_only():
    main.calibration_baselines.clear()
    main.buffers.clear()
    main.calibration_baselines[(1, 2)] = _stored(time.time())
    main.calibration_baselines[(1, 3)] = _stored(time.time())
    main.calibration_baselines[(9, 2)] = _stored(time.time())
    main.http_client = AsyncMock()
    result = await main.end_session(1)
    assert result == {"status": "ended", "barter_id": 1}
    assert list(main.calibration_baselines) == [(9, 2)]
    main.calibration_baselines.clear()


def test_ws_disconnect_clears_calibration_baseline():
    main.calibration_baselines.clear()
    main.buffers.clear()
    main.calibration_baselines[(1, 2)] = _stored(time.time())
    main.calibration_baselines[(1, 3)] = _stored(time.time())
    with TestClient(main.app) as client:
        main.http_client = AsyncMock()
        with client.websocket_connect("/video/1/2"):
            pass
    assert (1, 2) not in main.calibration_baselines
    assert (1, 3) in main.calibration_baselines
    main.calibration_baselines.clear()
