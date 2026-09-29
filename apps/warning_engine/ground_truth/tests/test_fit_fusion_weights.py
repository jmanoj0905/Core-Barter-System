import math
import pytest

from fit_fusion_weights import (
    _sigmoid, fit_logistic_regression, split_by_scenario, roc_auc, accuracy_at_threshold,
)


def test_sigmoid_bounds():
    assert _sigmoid(-100) == pytest.approx(0.0, abs=1e-6)
    assert _sigmoid(100) == pytest.approx(1.0, abs=1e-6)
    assert _sigmoid(0) == pytest.approx(0.5)


def test_fit_logistic_regression_separates_a_trivially_separable_dataset():
    rows = [
        {"speech_score": 0.9, "video_score": 0.9, "engaged_label": 1},
        {"speech_score": 0.85, "video_score": 0.88, "engaged_label": 1},
        {"speech_score": 0.1, "video_score": 0.1, "engaged_label": 0},
        {"speech_score": 0.05, "video_score": 0.12, "engaged_label": 0},
    ] * 10  # repeated to give gradient descent enough signal
    a, b, bias = fit_logistic_regression(rows, learning_rate=0.5, epochs=500)
    high_pred = _sigmoid(a * 0.9 + b * 0.9 + bias)
    low_pred = _sigmoid(a * 0.1 + b * 0.1 + bias)
    assert high_pred > 0.8
    assert low_pred < 0.2


def test_split_by_scenario_never_splits_a_scenario_across_both_halves():
    rows = [{"scenario_id": i % 10, "speech_score": 0.5, "video_score": 0.5, "engaged_label": i % 2}
            for i in range(100)]
    train, test = split_by_scenario(rows, holdout_fraction=0.3, seed=1)
    train_ids = {r["scenario_id"] for r in train}
    test_ids = {r["scenario_id"] for r in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train) + len(test) == len(rows)


def test_roc_auc_perfect_separation_is_one():
    assert roc_auc(scores=[0.9, 0.8, 0.2, 0.1], labels=[1, 1, 0, 0]) == pytest.approx(1.0)


def test_accuracy_at_threshold():
    acc = accuracy_at_threshold(scores=[0.9, 0.4, 0.6, 0.1], labels=[1, 0, 1, 0], threshold=0.5)
    assert acc == 1.0


def test_fit_logistic_regression_does_not_raise_on_all_one_label():
    rows = [{"speech_score": 0.5, "video_score": 0.5, "engaged_label": 1}] * 10
    a, b, bias = fit_logistic_regression(rows, learning_rate=0.1, epochs=100)
    assert math.isfinite(a) and math.isfinite(b) and math.isfinite(bias)
