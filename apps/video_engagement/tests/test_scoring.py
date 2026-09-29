import pytest
from scoring import (
    DEFAULT_WEIGHTS,
    _LEFT_CHEEK,
    _RIGHT_CHEEK,
    _deviation_from_baseline,
    raw_ratios_from_landmarks,
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


def test_default_weights_match_the_recorded_experiment():
    """The shipped weights must be the ones the recorded fit actually chose.

    Guards against the failure this whole exercise exists to prevent: a weight
    quietly drifting away from the experiment that justified it.
    """
    import json
    from pathlib import Path

    from scoring import ATTENTIVE_SCORE_THRESHOLD

    results_path = (
        Path(__file__).resolve().parents[3] / "docs" / "video_engagement" / "weight-search-results.json"
    )
    chosen = json.loads(results_path.read_text(encoding="utf-8"))["chosen"]
    assert list(DEFAULT_WEIGHTS) == pytest.approx(chosen["weights"])
    assert ATTENTIVE_SCORE_THRESHOLD == pytest.approx(chosen["threshold"], abs=5e-5)


def test_attentive_threshold_sits_between_the_class_means():
    """A threshold outside the observed class means would classify everything
    one way, whatever its Youden score claimed."""
    import json
    from pathlib import Path

    from scoring import ATTENTIVE_SCORE_THRESHOLD

    results_path = (
        Path(__file__).resolve().parents[3] / "docs" / "video_engagement" / "weight-search-results.json"
    )
    means = json.loads(results_path.read_text(encoding="utf-8"))["chosen"]["mean_score_per_label"]
    inattentive = [v for k, v in means.items() if k != "attentive"]
    assert max(inattentive) < ATTENTIVE_SCORE_THRESHOLD < means["attentive"]


def _off_center_landmarks():
    # Open eyes, nose and irises deliberately off-center so ratios != 0.5.
    return {
        33: (0.30, 0.40), 160: (0.32, 0.36), 158: (0.34, 0.36),
        133: (0.36, 0.40), 153: (0.34, 0.44), 144: (0.32, 0.44),
        362: (0.60, 0.40), 385: (0.62, 0.36), 387: (0.64, 0.36),
        263: (0.66, 0.40), 373: (0.64, 0.44), 380: (0.62, 0.44),
        1: (0.52, 0.42), 234: (0.28, 0.42), 454: (0.68, 0.42),
        468: (0.64, 0.42), 473: (0.31, 0.42),
    }


def test_deviation_from_baseline_at_baseline_is_zero():
    assert _deviation_from_baseline(0.5, 0.5) == 0.0


def test_deviation_from_baseline_matches_old_formula_when_baseline_is_half():
    for x in (0.0, 0.2, 0.5, 0.8, 1.0, 1.3, -0.4):
        assert _deviation_from_baseline(x, 0.5) == max(0.0, min(1.0, abs(x - 0.5) * 2.0))


def test_deviation_from_baseline_asymmetric_off_center_baseline():
    # baseline 0.2: max_dev = max(0.2, 0.8) = 0.8
    assert _deviation_from_baseline(1.0, 0.2) == pytest.approx(1.0)
    assert _deviation_from_baseline(0.2, 0.2) == 0.0
    assert _deviation_from_baseline(0.6, 0.2) == pytest.approx(0.5)


def test_sub_signals_no_baseline_matches_default_baseline():
    landmarks = _off_center_landmarks()
    assert sub_signals_from_mediapipe_landmarks(landmarks) == sub_signals_from_mediapipe_landmarks(
        landmarks, baseline={"head_ratio": 0.5, "gaze_ratio": 0.5}
    )


def test_sub_signals_default_baseline_matches_pre_calibration_formula():
    landmarks = _off_center_landmarks()
    result = sub_signals_from_mediapipe_landmarks(landmarks)
    ratios = raw_ratios_from_landmarks(landmarks)
    assert result["head_deviation"] == max(0.0, min(1.0, abs(ratios["head_ratio"] - 0.5) * 2.0))
    assert result["gaze_centered"] == max(
        0.0, min(1.0, 1.0 - abs(ratios["gaze_ratio"] - 0.5) * 2.0)
    )
    assert result["head_deviation"] > 0.0
    assert result["gaze_centered"] < 1.0


def test_sub_signals_personalized_baseline_shifts_neutral_point():
    landmarks = _off_center_landmarks()
    ratios = raw_ratios_from_landmarks(landmarks)
    result = sub_signals_from_mediapipe_landmarks(
        landmarks, baseline={"head_ratio": ratios["head_ratio"], "gaze_ratio": ratios["gaze_ratio"]}
    )
    assert result["head_deviation"] == pytest.approx(0.0)
    assert result["gaze_centered"] == pytest.approx(1.0)


def test_raw_ratios_degenerate_span_returns_none_head_ratio():
    landmarks = _off_center_landmarks()
    landmarks[_LEFT_CHEEK] = landmarks[_RIGHT_CHEEK]  # force span == 0
    assert raw_ratios_from_landmarks(landmarks)["head_ratio"] is None


def test_sub_signals_degenerate_span_forces_max_head_deviation_regardless_of_baseline():
    landmarks = _off_center_landmarks()
    landmarks[_LEFT_CHEEK] = landmarks[_RIGHT_CHEEK]
    result = sub_signals_from_mediapipe_landmarks(
        landmarks, baseline={"head_ratio": 0.9, "gaze_ratio": 0.5}
    )
    assert result["head_deviation"] == 1.0
