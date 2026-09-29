"""Pure video-engagement scoring functions — no I/O, no ML model loading.

Weights and threshold below are FITTED values, not guesses: grid-searched over
3,272 labeled frames from two openly-licensed public datasets. Method, data,
licences, rubric and full results are in docs/video_engagement/design-choices.md
and docs/video_engagement/weight-search-results.json. Re-fit with
`weight_search.py --corpus`.
"""

import math

# w_eyes, w_head, w_gaze — fitted 2026-09-26 by `weight_search.py --corpus`
# over 2,295 labeled frames (Head Pose Image Database + closed-open-eyes),
# scored on 977 held-out frames from subjects/shards the fit never saw:
# mean per-split ROC-AUC 0.9894 on the fit set, 0.9886 on the holdout.
# w_gaze sits at the grid floor because neither public dataset contains
# head-forward/eyes-to-the-side frames, so no data could rank it; it is kept
# non-zero so the signal stays live in production for that failure mode.
DEFAULT_WEIGHTS: tuple[float, float, float] = (0.55, 0.4, 0.05)

# Decision threshold on the fused score: at or above this, the window is
# classified attentive. Fitted on the same corpus by maximising Youden's J;
# on the held-out groups it gives tpr 0.975 / fpr 0.052 (balanced accuracy
# 0.962). Nothing in the service thresholds the score today — warning_engine
# consumes the continuous value — so this is the number to use when a boolean
# "attentive" is needed; see design-choices.md.
ATTENTIVE_SCORE_THRESHOLD: float = 0.6123

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


def _deviation_from_baseline(value: float, baseline: float) -> float:
    """Distance of `value` from `baseline`, scaled to 0-1 by the farthest reachable point.

    With baseline 0.5 this is exactly abs(value - 0.5) * 2.0, clamped.
    """
    max_dev = max(baseline, 1.0 - baseline)
    if max_dev == 0:
        return 0.0
    return _clamp01(abs(value - baseline) / max_dev)


def raw_ratios_from_landmarks(
    landmarks: dict[int, tuple[float, float]],
) -> dict[str, float | None]:
    """Raw head/gaze ratios (each ~0-1, 0.5 = centered) before any baseline is applied.

    head_ratio is None when the cheek span is zero (degenerate face).
    """
    nose = landmarks[_NOSE_TIP]
    left_cheek = landmarks[_LEFT_CHEEK]
    right_cheek = landmarks[_RIGHT_CHEEK]
    span = right_cheek[0] - left_cheek[0]
    head_ratio = None if span == 0 else (nose[0] - left_cheek[0]) / span

    def _gaze_ratio(iris_key: int, eye: dict[str, int]) -> float:
        iris = landmarks[iris_key]
        p1, p4 = landmarks[eye["p1"]], landmarks[eye["p4"]]
        eye_span = p4[0] - p1[0]
        if eye_span == 0:
            return 0.5
        return (iris[0] - p1[0]) / eye_span

    left_gaze = _gaze_ratio(_LEFT_IRIS_CENTER, _LEFT_EYE)
    right_gaze = _gaze_ratio(_RIGHT_IRIS_CENTER, _RIGHT_EYE)
    return {"head_ratio": head_ratio, "gaze_ratio": (left_gaze + right_gaze) / 2.0}


def sub_signals_from_mediapipe_landmarks(
    landmarks: dict[int, tuple[float, float]],
    baseline: dict[str, float] | None = None,
) -> dict[str, float]:
    """Map raw Face Mesh landmarks to the three formula sub-signals, each 0-1.

    `baseline` recenters the head/gaze neutral point (default 0.5 / 0.5).
    """
    if baseline is None:
        baseline = {"head_ratio": 0.5, "gaze_ratio": 0.5}

    right_ear = _eye_aspect_ratio(landmarks, _RIGHT_EYE)
    left_ear = _eye_aspect_ratio(landmarks, _LEFT_EYE)
    avg_ear = (right_ear + left_ear) / 2.0
    eyes_open = _clamp01((avg_ear - _EAR_CLOSED) / (_EAR_OPEN - _EAR_CLOSED))

    ratios = raw_ratios_from_landmarks(landmarks)
    if ratios["head_ratio"] is None:
        head_deviation = 1.0
    else:
        head_deviation = _deviation_from_baseline(ratios["head_ratio"], baseline["head_ratio"])
    gaze_centered = 1.0 - _deviation_from_baseline(ratios["gaze_ratio"], baseline["gaze_ratio"])

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
