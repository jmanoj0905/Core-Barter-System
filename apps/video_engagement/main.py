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
