"""Exercise calibration/evaluation separation through the command-line interface."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("threshold_experiment.py")
FIELDS = ["source", "topic", "scope", "text", "expected_label", "similarity", "session_id", "family"]


def rows_for(topic, session, scores):
    return [
        dict(zip(FIELDS, ["real", topic, "scope", f"{session} utterance {i}", label, score, session, topic]))
        for i, (label, score) in enumerate(scores)
    ]


def datasets():
    calibration = rows_for("A", "cal-A", [
        ("correct", .55), ("correct", .60), ("weakly_correct", .35),
        ("weakly_correct", .40), ("incorrect", .05), ("incorrect", .07),
    ]) + rows_for("B", "cal-B", [
        ("correct", .90), ("correct", .95), ("weakly_correct", .70),
        ("weakly_correct", .75), ("incorrect", .05), ("incorrect", .07),
    ])
    evaluation = rows_for("A", "eval-A", [
        ("correct", .58), ("weakly_correct", .38), ("incorrect", .04),
    ]) + rows_for("B", "eval-B", [
        ("correct", .92), ("weakly_correct", .72), ("incorrect", .04),
    ])
    return calibration, evaluation


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(tmp_path, calibration, evaluation, *options):
    cal_path, eval_path, output = (tmp_path / p for p in ("cal.csv", "eval.csv", "report.json"))
    write_csv(cal_path, calibration)
    write_csv(eval_path, evaluation)
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--calibration", str(cal_path),
        "--evaluation", str(eval_path), "--output", str(output), *options,
    ], capture_output=True, text=True)
    report = json.loads(output.read_text()) if output.exists() else None
    return result, report


def test_group_thresholds_improve_shifted_topics_without_changing_lower(tmp_path):
    result, report = run_comparison(tmp_path, *datasets())
    assert result.returncode == 0, result.stderr
    assert report["metrics"]["global"]["accuracy"] == pytest.approx(5 / 6)
    assert report["metrics"]["grouped"]["accuracy"] == 1
    assert report["metrics"]["global"]["macro_f1"] == pytest.approx((.8 + 2/3 + 1)/3)
    assert report["metrics"]["grouped"]["false_incorrect_rate"] == 0
    assert report["paired_changes"] == {"improved": 1, "regressed": 0, "net_correct": 1}
    assert all(p["global_lower"] == p["grouped_lower"] for p in report["predictions"])


def test_evaluation_labels_cannot_change_fitted_thresholds(tmp_path):
    calibration, evaluation = datasets()
    first, before = run_comparison(tmp_path, calibration, evaluation)
    assert first.returncode == 0, first.stderr
    for row in evaluation:
        row["expected_label"] = "incorrect"
    second, after = run_comparison(tmp_path, calibration, evaluation)
    assert second.returncode == 0, second.stderr
    assert before["thresholds"] == after["thresholds"]
    assert [p["grouped"] for p in before["predictions"]] == [p["grouped"] for p in after["predictions"]]


def test_unseen_topic_falls_back_to_global_thresholds(tmp_path):
    calibration, _ = datasets()
    evaluation = rows_for("unseen", "test-unseen", [("weakly_correct", .72)])
    result, report = run_comparison(tmp_path, calibration, evaluation)
    assert result.returncode == 0, result.stderr
    assert report["fallback_count"] == 1
    assert report["predictions"][0]["grouped"] == report["predictions"][0]["global"]


def test_family_calibration_can_cover_an_unseen_topic_in_a_known_family(tmp_path):
    calibration, _ = datasets()
    evaluation = rows_for("new topic", "new-session", [("weakly_correct", .72)])
    evaluation[0]["family"] = "B"
    result, report = run_comparison(tmp_path, calibration, evaluation, "--group-by", "family")
    assert result.returncode == 0, result.stderr
    assert report["fallback_count"] == 0
    assert report["predictions"][0]["grouped"] == "weakly_correct"


@pytest.mark.parametrize("leak", ["session", "duplicate"])
def test_rejects_calibration_evaluation_leakage(tmp_path, leak):
    calibration, evaluation = datasets()
    if leak == "session":
        evaluation[0]["session_id"] = calibration[0]["session_id"]
    else:
        evaluation[0]["text"] = calibration[0]["text"]
    result, _ = run_comparison(tmp_path, calibration, evaluation)
    assert result.returncode != 0
    assert leak in result.stderr.lower()


@pytest.mark.parametrize("field,value", [
    ("session_id", ""), ("expected_label", "made_up"), ("similarity", "nan"),
])
def test_rejects_unusable_evaluation_rows(tmp_path, field, value):
    calibration, evaluation = datasets()
    evaluation[0][field] = value
    result, _ = run_comparison(tmp_path, calibration, evaluation)
    assert result.returncode != 0
    assert field in result.stderr


def test_fixture_diagnostic_labels_its_limits_and_tests_unseen_fallback(tmp_path):
    calibration, evaluation = datasets()
    rows = calibration + evaluation
    for row in rows:
        row["source"] = "synthetic"
        row["session_id"] = ""
    path, output = tmp_path / "fixtures.csv", tmp_path / "diagnostic.json"
    write_csv(path, rows)
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--fixture-diagnostic", str(path),
        "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["production_evidence"] is False
    assert report["leave_one_example_out"]["metrics"]["global"]["n"] == 18
    unseen = report["leave_one_topic_out"]
    assert unseen["fallback_count"] == 18
    assert unseen["metrics"]["global"] == unseen["metrics"]["grouped"]


def test_sparse_group_uses_global_instead_of_fitting_one_boundary_class(tmp_path):
    calibration, evaluation = datasets()
    calibration = [r for r in calibration if not (r["topic"] == "B" and r["expected_label"] == "weakly_correct")]
    result, report = run_comparison(tmp_path, calibration, evaluation)
    assert result.returncode == 0, result.stderr
    assert report["fallback_count"] == 3
    assert all(p["global"] == p["grouped"] for p in report["predictions"] if p["fallback"])


def test_calibration_covers_negative_cosine_scores(tmp_path):
    scores = [("correct", -.10), ("correct", -.15), ("weakly_correct", -.40),
              ("weakly_correct", -.45), ("incorrect", -.80), ("incorrect", -.85)]
    calibration = rows_for("A", "cal-A", scores)
    evaluation = rows_for("A", "eval-A", [("correct", -.12), ("weakly_correct", -.42), ("incorrect", -.90)])
    result, report = run_comparison(tmp_path, calibration, evaluation)
    assert result.returncode == 0, result.stderr
    assert report["metrics"]["global"]["accuracy"] == 1


def test_fixture_diagnostic_refuses_real_data(tmp_path):
    calibration, _ = datasets()
    path = tmp_path / "real.csv"
    write_csv(path, calibration)
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--fixture-diagnostic", str(path),
        "--output", str(tmp_path / "report.json"),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "source=synthetic" in result.stderr
