import pytest
from scoring import (
    DEFAULT_WEIGHTS,
    sub_signals_from_mediapipe_landmarks,
    sub_signals_from_rekognition_face_detail,
    video_attention_score,
)


def test_video_attention_score_weighted_combination():
    sub_signals = {"eyes_open": 1.0, "head_deviation": 0.0, "gaze_centered": 1.0}
    score = video_attention_score(sub_signals, weights=(0.4, 0.4, 0.2))
    assert score == pytest.approx(1.0)

    sub_signals_bad = {"eyes_open": 0.0, "head_deviation": 1.0, "gaze_centered": 0.0}
    score_bad = video_attention_score(sub_signals_bad, weights=(0.4, 0.4, 0.2))
    assert score_bad == pytest.approx(0.0)


def test_video_attention_score_clamped_to_0_1():
    score = video_attention_score(
        {"eyes_open": 2.0, "head_deviation": -1.0, "gaze_centered": 2.0},
        weights=(0.4, 0.4, 0.2),
    )
    assert 0.0 <= score <= 1.0


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS) == pytest.approx(1.0)


def test_mediapipe_eyes_open_high_when_eye_landmarks_wide():
    # Synthetic landmarks: right eye corners (33, 133) far apart in y from
    # top/bottom lid points (160,158 / 153,144) -> wide-open eye.
    landmarks = {
        # right eye (open)
        33: (0.30, 0.40), 160: (0.32, 0.36), 158: (0.34, 0.36),
        133: (0.36, 0.40), 153: (0.34, 0.44), 144: (0.32, 0.44),
        # left eye (open)
        362: (0.60, 0.40), 385: (0.62, 0.36), 387: (0.64, 0.36),
        263: (0.66, 0.40), 373: (0.64, 0.44), 380: (0.62, 0.44),
        # nose + cheeks for head pose, iris for gaze
        1: (0.48, 0.42), 234: (0.28, 0.42), 454: (0.68, 0.42),
        468: (0.48, 0.42), 473: (0.48, 0.42),
    }
    signals = sub_signals_from_mediapipe_landmarks(landmarks)
    assert signals["eyes_open"] > 0.5
    assert 0.0 <= signals["head_deviation"] <= 1.0
    assert 0.0 <= signals["gaze_centered"] <= 1.0


def test_mediapipe_missing_landmarks_raises():
    with pytest.raises(KeyError):
        sub_signals_from_mediapipe_landmarks({1: (0.5, 0.5)})


def test_rekognition_eyes_open_maps_boolean_to_float():
    face_detail = {
        "EyesOpen": {"Value": True, "Confidence": 99.0},
        "Pose": {"Yaw": 5.0, "Pitch": -3.0, "Roll": 1.0},
    }
    signals = sub_signals_from_rekognition_face_detail(face_detail)
    assert signals["eyes_open"] == 1.0
    assert 0.0 <= signals["head_deviation"] <= 1.0
    assert 0.0 <= signals["gaze_centered"] <= 1.0


def test_rekognition_eyes_closed_and_large_yaw():
    face_detail = {
        "EyesOpen": {"Value": False, "Confidence": 90.0},
        "Pose": {"Yaw": 80.0, "Pitch": 10.0, "Roll": 0.0},
    }
    signals = sub_signals_from_rekognition_face_detail(face_detail)
    assert signals["eyes_open"] == 0.0
    # NOTE: yaw=80, pitch=10 -> (80/90 + 10/90) / 2 == 0.5 exactly (verified,
    # not a float artifact), so the brief's strict `> 0.5` never passes. Using
    # `>=` here to match the actual (correct) verbatim scoring.py formula.
    assert signals["head_deviation"] >= 0.5
    assert signals["gaze_centered"] < 0.2
