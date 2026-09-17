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


def process_frame_local(frame_bytes: bytes) -> dict | None:
    """Decode a JPEG frame, run MediaPipe Face Mesh, return sub-signals or None."""
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    height, width = img.shape[:2]
    landmarks = _landmarks_from_mediapipe_result(result, width, height)
    return sub_signals_from_mediapipe_landmarks(landmarks)


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

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
WARNING_ENGINE_URL = os.getenv("WARNING_ENGINE_URL", "http://localhost:8003")
BUFFER_THRESHOLD_SECONDS = 5.0

_VALID_VIDEO_BACKENDS = {"local", "aws", "both"}
current_video_backend: str = os.getenv("VIDEO_BACKEND", "local")

buffers: dict[tuple[int, int], dict] = {}


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

    if backend in ("local", "both"):
        detected = [s for s in (process_frame_local(f) for f in frames) if s is not None]
        if detected:
            avg = {
                key: sum(s[key] for s in detected) / len(detected)
                for key in ("eyes_open", "head_deviation", "gaze_centered")
            }
            await _score_and_post(barter_id, user_id, window_start, window_end, avg, "local")

    if backend in ("aws", "both"):
        mid_frame = frames[len(frames) // 2]
        sub_signals = process_frame_aws(mid_frame)
        if sub_signals is not None:
            await _score_and_post(barter_id, user_id, window_start, window_end, sub_signals, "aws")


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


@app.post("/session/{barter_id}/end")
async def end_session(barter_id: int):
    keys_to_delete = [k for k in buffers if k[0] == barter_id]
    for key in keys_to_delete:
        user_id = key[1]
        buf = buffers[key]
        if buf["frames"]:
            await process_buffer(barter_id, user_id, buf, current_video_backend)
        del buffers[key]
    return {"status": "ended", "barter_id": barter_id}
