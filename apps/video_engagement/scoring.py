"""Pure video-engagement scoring functions — no I/O, no ML model loading.

Formula and weights are experimental (see docs/video_engagement/design-choices.md).
DEFAULT_WEIGHTS is a placeholder until weight_search.py picks a real value.
"""

import math

# w_eyes, w_head, w_gaze — placeholder, pending weight_search.py
DEFAULT_WEIGHTS: tuple[float, float, float] = (0.4, 0.4, 0.2)

# MediaPipe Face Mesh landmark indices used (468/478-point mesh, refine_landmarks=True)
_RIGHT_EYE = {"p1": 33, "p2": 160, "p3": 158, "p4": 133, "p5": 153, "p6": 144}
_LEFT_EYE = {"p1": 362, "p2": 385, "p3": 387, "p4": 263, "p5": 373, "p6": 380}
_NOSE_TIP = 1
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473

_EAR_CLOSED = 0.15
_EAR_OPEN = 0.30


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _eye_aspect_ratio(landmarks: dict[int, tuple[float, float]], eye: dict[str, int]) -> float:
    p1, p2, p3, p4, p5, p6 = (landmarks[eye[k]] for k in ("p1", "p2", "p3", "p4", "p5", "p6"))
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def sub_signals_from_mediapipe_landmarks(
    landmarks: dict[int, tuple[float, float]],
) -> dict[str, float]:
    """Map raw Face Mesh landmarks to the three formula sub-signals, each 0-1."""
    right_ear = _eye_aspect_ratio(landmarks, _RIGHT_EYE)
    left_ear = _eye_aspect_ratio(landmarks, _LEFT_EYE)
    avg_ear = (right_ear + left_ear) / 2.0
    eyes_open = _clamp01((avg_ear - _EAR_CLOSED) / (_EAR_OPEN - _EAR_CLOSED))

    nose = landmarks[_NOSE_TIP]
    left_cheek = landmarks[_LEFT_CHEEK]
    right_cheek = landmarks[_RIGHT_CHEEK]
    span = right_cheek[0] - left_cheek[0]
    if span == 0:
        head_deviation = 1.0
    else:
        ratio = (nose[0] - left_cheek[0]) / span
        head_deviation = _clamp01(abs(ratio - 0.5) * 2.0)

    def _gaze_ratio(iris_key: int, eye: dict[str, int]) -> float:
        iris = landmarks[iris_key]
        p1, p4 = landmarks[eye["p1"]], landmarks[eye["p4"]]
        eye_span = p4[0] - p1[0]
        if eye_span == 0:
            return 0.5
        return (iris[0] - p1[0]) / eye_span

    left_gaze = _gaze_ratio(_LEFT_IRIS_CENTER, _LEFT_EYE)
    right_gaze = _gaze_ratio(_RIGHT_IRIS_CENTER, _RIGHT_EYE)
    avg_gaze_ratio = (left_gaze + right_gaze) / 2.0
    gaze_centered = _clamp01(1.0 - abs(avg_gaze_ratio - 0.5) * 2.0)

    return {
        "eyes_open": eyes_open,
        "head_deviation": head_deviation,
        "gaze_centered": gaze_centered,
    }


def sub_signals_from_rekognition_face_detail(face_detail: dict) -> dict[str, float]:
    """Map an AWS Rekognition DetectFaces FaceDetails[i] entry to the sub-signals.

    Rekognition has no iris/gaze data, so gaze_centered is approximated from
    yaw alone — a known limitation, documented in design-choices.md.
    """
    eyes_open = 1.0 if face_detail["EyesOpen"]["Value"] else 0.0

    pose = face_detail["Pose"]
    yaw, pitch = abs(pose["Yaw"]), abs(pose["Pitch"])
    head_deviation = _clamp01((yaw / 90.0 + pitch / 90.0) / 2.0)
    gaze_centered = _clamp01(1.0 - yaw / 90.0)

    return {
        "eyes_open": eyes_open,
        "head_deviation": head_deviation,
        "gaze_centered": gaze_centered,
    }


def video_attention_score(
    sub_signals: dict[str, float], weights: tuple[float, float, float] = DEFAULT_WEIGHTS
) -> float:
    w_eyes, w_head, w_gaze = weights
    raw = (
        w_eyes * sub_signals["eyes_open"]
        + w_head * (1.0 - sub_signals["head_deviation"])
        + w_gaze * sub_signals["gaze_centered"]
    )
    return _clamp01(raw)
