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
