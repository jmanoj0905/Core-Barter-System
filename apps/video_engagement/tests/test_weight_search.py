import pytest
from weight_search import pearson_correlation, sweep_subsignal_weights, sweep_fusion_weights


def test_pearson_correlation_perfect_positive():
    assert pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative():
    assert pearson_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_correlation_constant_series_returns_zero():
    # Undefined correlation (zero variance) should degrade to 0.0, not raise.
    assert pearson_correlation([1, 1, 1], [1, 2, 3]) == 0.0


def test_sweep_subsignal_weights_picks_best_correlation():
    raw_signals = [
        {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.9},
        {"eyes_open": 0.2, "head_deviation": 0.8, "gaze_centered": 0.3},
        {"eyes_open": 0.8, "head_deviation": 0.2, "gaze_centered": 0.7},
    ]
    ground_truth = [0.9, 0.2, 0.75]  # matches eyes_open closely

    results = sweep_subsignal_weights(raw_signals, ground_truth, step=0.2)
    assert len(results) > 0
    best = max(results, key=lambda r: r["correlation"])
    assert best["correlation"] > 0.9
    assert best["weights"][0] > 0.3  # eyes_open weight should dominate


def test_sweep_fusion_weights_returns_sorted_by_correlation():
    video_scores = [0.9, 0.2, 0.8]
    speech_scores = [0.85, 0.25, 0.75]
    results = sweep_fusion_weights(video_scores, speech_scores, step=0.1)
    assert results == sorted(results, key=lambda r: -r["correlation"])
    assert all(abs(r["w_video"] + r["w_speech"] - 1.0) < 1e-9 for r in results)
