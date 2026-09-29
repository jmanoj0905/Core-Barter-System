import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("video-engagement")

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=15.0)
    yield
    await http_client.aclose()


app = FastAPI(title="Video Engagement", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

import cv2
import mediapipe as mp
import numpy as np

import scoring
from scoring import (
    DEFAULT_WEIGHTS,
    sub_signals_from_mediapipe_landmarks,
    sub_signals_from_rekognition_face_detail,
    video_attention_score,
)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

WEIGHT_EYES = float(os.getenv("VIDEO_WEIGHT_EYES", str(DEFAULT_WEIGHTS[0])))
WEIGHT_HEAD = float(os.getenv("VIDEO_WEIGHT_HEAD", str(DEFAULT_WEIGHTS[1])))
WEIGHT_GAZE = float(os.getenv("VIDEO_WEIGHT_GAZE", str(DEFAULT_WEIGHTS[2])))
ACTIVE_WEIGHTS = (WEIGHT_EYES, WEIGHT_HEAD, WEIGHT_GAZE)

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def _landmarks_from_mediapipe_result(result, width: int, height: int) -> dict[int, tuple[float, float]]:
    face = result.multi_face_landmarks[0]
    return {i: (lm.x * width, lm.y * height) for i, lm in enumerate(face.landmark)}


def _landmarks_from_frame_bytes(frame_bytes: bytes) -> dict[int, tuple[float, float]] | None:
    """Decode a JPEG frame and run MediaPipe Face Mesh; return pixel landmarks or None."""
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    height, width = img.shape[:2]
    return _landmarks_from_mediapipe_result(result, width, height)


def process_frame_local(frame_bytes: bytes, baseline: dict | None = None) -> dict | None:
    """Decode a JPEG frame, run MediaPipe Face Mesh, return sub-signals or None."""
    landmarks = _landmarks_from_frame_bytes(frame_bytes)
    if landmarks is None:
        return None
    return sub_signals_from_mediapipe_landmarks(landmarks, baseline)


def _rekognition_client():
    import boto3
    return boto3.client(
        "rekognition",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def process_frame_aws(frame_bytes: bytes) -> dict | None:
    """Send one JPEG frame to Rekognition DetectFaces, return sub-signals or None.

    Fails open: any error (including throttling) logs and returns None so
    the window simply skips the cloud score rather than raising.
    """
    try:
        client = _rekognition_client()
        resp = client.detect_faces(Image={"Bytes": frame_bytes}, Attributes=["ALL"])
        face_details = resp.get("FaceDetails", [])
        if not face_details:
            return None
        return sub_signals_from_rekognition_face_detail(face_details[0])
    except Exception as e:
        logger.error("Rekognition detect_faces failed: %s", e)
        return None


@app.get("/")
async def root():
    return {"service": "video-engagement", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


import time

from fastapi import File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
WARNING_ENGINE_URL = os.getenv("WARNING_ENGINE_URL", "http://localhost:8003")
BUFFER_THRESHOLD_SECONDS = 5.0

_VALID_VIDEO_BACKENDS = {"local", "aws", "both"}
current_video_backend: str = os.getenv("VIDEO_BACKEND", "local")

CALIBRATION_COOLDOWN_SECONDS = 120.0
CALIBRATION_MIN_FACE_FRAMES = 3

buffers: dict[tuple[int, int], dict] = {}
# (barter_id, user_id) -> {"head_ratio", "gaze_ratio", "recalibration_count", "last_calibrated_at"}
calibration_baselines: dict[tuple[int, int], dict] = {}


def _new_buffer() -> dict:
    return {"frames": [], "wall_start": time.time()}


class VideoConfigRequest(BaseModel):
    backend: str


@app.get("/video/config")
async def get_video_config():
    return {"backend": current_video_backend, "available": sorted(_VALID_VIDEO_BACKENDS)}


@app.post("/video/config")
async def set_video_config(body: VideoConfigRequest):
    global current_video_backend
    if body.backend not in _VALID_VIDEO_BACKENDS:
        raise HTTPException(400, f"Unknown backend '{body.backend}'. Choose from: {sorted(_VALID_VIDEO_BACKENDS)}")
    current_video_backend = body.backend
    return {"backend": current_video_backend}


async def _score_and_post(barter_id: int, user_id: int, window_start: float, window_end: float,
                           sub_signals: dict, backend_used: str):
    score = video_attention_score(sub_signals, ACTIVE_WEIGHTS)

    payload = {
        "user_id": user_id,
        "window_start": window_start,
        "window_end": window_end,
        "video_attention_score": score,
        "backend_used": backend_used,
        "raw_signals": sub_signals,
    }
    try:
        await http_client.post(f"{BACKEND_URL}/session/{barter_id}/video-engagement", json=payload)
    except Exception as e:
        logger.error("Failed to POST video-engagement to backend: %s", e)

    try:
        await http_client.post(f"{WARNING_ENGINE_URL}/video-engagement/update", json={
            "barter_id": barter_id, "user_id": user_id, "video_attention_score": score,
        })
    except Exception as e:
        logger.error("Failed to POST video-engagement update to warning_engine: %s", e)


async def process_buffer(barter_id: int, user_id: int, buf: dict, backend: str):
    """Run the configured backend(s) on the buffered frames and post any resulting score(s)."""
    frames = buf["frames"]
    if not frames:
        return

    window_start = buf["wall_start"]
    window_end = time.time()

    stored = calibration_baselines.get((barter_id, user_id))
    baseline = (
        {"head_ratio": stored["head_ratio"], "gaze_ratio": stored["gaze_ratio"]}
        if stored else None
    )

    if backend in ("local", "both"):
        try:
            detected = [s for s in (process_frame_local(f, baseline=baseline) for f in frames)
                        if s is not None]
            avg = None
            if detected:
                avg = {
                    key: sum(s[key] for s in detected) / len(detected)
                    for key in ("eyes_open", "head_deviation", "gaze_centered")
                }
        except Exception as e:
            logger.error("Local frame processing failed: %s", e)
            avg = None

        if avg:
            await _score_and_post(barter_id, user_id, window_start, window_end, avg, "local")

    if backend in ("aws", "both"):
        mid_frame = frames[len(frames) // 2]
        sub_signals = process_frame_aws(mid_frame)
        if sub_signals is not None:
            await _score_and_post(barter_id, user_id, window_start, window_end, sub_signals, "aws")


def _compute_calibration_baseline(frames: list[bytes]) -> dict | None:
    """Average raw head/gaze ratios over frames with a usable face; None if too few."""
    ratios = []
    for frame in frames:
        landmarks = _landmarks_from_frame_bytes(frame)
        if landmarks is None:
            continue
        r = scoring.raw_ratios_from_landmarks(landmarks)
        if r["head_ratio"] is None:
            continue
        ratios.append(r)
    if len(ratios) < CALIBRATION_MIN_FACE_FRAMES:
        return None
    return {
        "head_ratio": sum(r["head_ratio"] for r in ratios) / len(ratios),
        "gaze_ratio": sum(r["gaze_ratio"] for r in ratios) / len(ratios),
        "frames_used": len(ratios),
    }


async def _post_calibration_log(barter_id: int, user_id: int, outcome: str,
                                old_baseline: dict | None, new_baseline: dict | None,
                                recalibration_count: int):
    """Best-effort audit POST; never raises."""
    try:
        payload = {
            "user_id": user_id,
            "outcome": outcome,
            "old_baseline_head_ratio": old_baseline["head_ratio"] if old_baseline else None,
            "old_baseline_gaze_ratio": old_baseline["gaze_ratio"] if old_baseline else None,
            "new_baseline_head_ratio": new_baseline["head_ratio"] if new_baseline else None,
            "new_baseline_gaze_ratio": new_baseline["gaze_ratio"] if new_baseline else None,
            "recalibration_count": recalibration_count,
        }
        await http_client.post(
            f"{BACKEND_URL}/session/{barter_id}/video-engagement/calibration-log", json=payload
        )
    except Exception as e:
        logger.error("Failed to POST calibration-log to backend: %s", e)


@app.post("/video/{barter_id}/{user_id}/calibrate")
async def calibrate(barter_id: int, user_id: int, frames: list[UploadFile] = File(...)):
    key = (barter_id, user_id)
    existing = calibration_baselines.get(key)
    now = time.time()
    recalibration_count = existing["recalibration_count"] if existing else 0

    if existing is not None:
        elapsed = now - existing["last_calibrated_at"]
        if elapsed < CALIBRATION_COOLDOWN_SECONDS:
            # Floor at 0.1 so a near-boundary call never reports a 0.0 wait.
            retry_after = max(0.1, round(CALIBRATION_COOLDOWN_SECONDS - elapsed, 1))
            await _post_calibration_log(barter_id, user_id, "recalibration_limit_reached",
                                        existing, None, recalibration_count)
            return {"calibrated": False, "reason": "recalibration_limit_reached",
                    "retry_after_seconds": retry_after}

    frame_bytes = [await f.read() for f in frames]
    baseline = _compute_calibration_baseline(frame_bytes)
    if baseline is None:
        await _post_calibration_log(barter_id, user_id, "insufficient_face_detections",
                                    existing, None, recalibration_count)
        return {"calibrated": False, "reason": "insufficient_face_detections"}

    new_count = recalibration_count + 1 if existing else 0
    calibration_baselines[key] = {
        "head_ratio": baseline["head_ratio"], "gaze_ratio": baseline["gaze_ratio"],
        "recalibration_count": new_count, "last_calibrated_at": now,
    }
    await _post_calibration_log(barter_id, user_id, "calibrated",
                                existing, calibration_baselines[key], new_count)
    return {
        "calibrated": True, "baseline_head_ratio": baseline["head_ratio"],
        "baseline_gaze_ratio": baseline["gaze_ratio"], "frames_used": baseline["frames_used"],
        "recalibration_count": new_count,
    }


def reset_buffer(buf: dict):
    buf["frames"] = []
    buf["wall_start"] = time.time()


@app.websocket("/video/{barter_id}/{user_id}")
async def video_ws(barter_id: int, user_id: int, ws: WebSocket):
    await ws.accept()
    key = (barter_id, user_id)
    buffers[key] = _new_buffer()
    buf = buffers[key]

    try:
        while True:
            frame = await ws.receive_bytes()
            if not frame:
                continue
            buf["frames"].append(frame)

            if time.time() - buf["wall_start"] >= BUFFER_THRESHOLD_SECONDS:
                await process_buffer(barter_id, user_id, buf, current_video_backend)
                reset_buffer(buf)

    except (WebSocketDisconnect, RuntimeError):
        if buf["frames"]:
            await process_buffer(barter_id, user_id, buf, current_video_backend)
        if key in buffers:
            del buffers[key]
        calibration_baselines.pop(key, None)


@app.post("/session/{barter_id}/end")
async def end_session(barter_id: int):
    keys_to_delete = [k for k in buffers if k[0] == barter_id]
    for key in keys_to_delete:
        user_id = key[1]
        buf = buffers[key]
        if buf["frames"]:
            await process_buffer(barter_id, user_id, buf, current_video_backend)
        del buffers[key]
    # Calibration can happen before (or without) a video WebSocket, so clean it up by barter_id.
    for key in [k for k in calibration_baselines if k[0] == barter_id]:
        calibration_baselines.pop(key, None)
    return {"status": "ended", "barter_id": barter_id}
