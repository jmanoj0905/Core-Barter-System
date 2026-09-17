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
