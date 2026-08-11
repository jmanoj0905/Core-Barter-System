import json
from eval_dataset.tools.compute_kappa import load_session_annotations, report_all_sessions, overall_kappa


def test_load_session_annotations_orders_by_window_index(tmp_path):
    (tmp_path / "sess_A01_bob.json").write_text(json.dumps([
        {"window_index": 1, "label": "incorrect"},
        {"window_index": 0, "label": "correct"},
    ]))
    (tmp_path / "sess_A01_amy.json").write_text(json.dumps([
        {"window_index": 0, "label": "correct"},
        {"window_index": 1, "label": "incorrect"},
    ]))

    result = load_session_annotations(str(tmp_path), "sess_A01")

    assert sorted(result) == sorted([["correct", "incorrect"], ["correct", "incorrect"]])


def test_report_all_sessions_computes_kappa_per_session(tmp_path):
    (tmp_path / "sess_A01_bob.json").write_text(json.dumps([{"window_index": 0, "label": "correct"}]))
    (tmp_path / "sess_A01_amy.json").write_text(json.dumps([{"window_index": 0, "label": "correct"}]))
    (tmp_path / "sess_B01_bob.json").write_text(json.dumps([{"window_index": 0, "label": "correct"}]))

    result = report_all_sessions(str(tmp_path))

    assert result == {"sess_A01": 1.0}  # sess_B01 skipped: only 1 annotator


def test_load_session_annotations_intersects_different_window_counts(tmp_path):
    # bob labeled windows 0,1,2; amy labeled only 0,2 (partial completion) -
    # only the common windows (0, 2) should be compared, in index order.
    (tmp_path / "sess_A01_bob.json").write_text(json.dumps([
        {"window_index": 0, "label": "correct"},
        {"window_index": 1, "label": "incorrect"},
        {"window_index": 2, "label": "weakly_correct"},
    ]))
    (tmp_path / "sess_A01_amy.json").write_text(json.dumps([
        {"window_index": 2, "label": "weakly_correct"},
        {"window_index": 0, "label": "correct"},
    ]))

    result = load_session_annotations(str(tmp_path), "sess_A01")

    assert result == [["correct", "weakly_correct"], ["correct", "weakly_correct"]]


def test_load_session_annotations_empty_intersection(tmp_path):
    (tmp_path / "sess_A01_bob.json").write_text(json.dumps([
        {"window_index": 0, "label": "correct"},
    ]))
    (tmp_path / "sess_A01_amy.json").write_text(json.dumps([
        {"window_index": 1, "label": "incorrect"},
    ]))

    result = load_session_annotations(str(tmp_path), "sess_A01")

    assert result == [[], []]


def test_report_all_sessions_skips_session_with_empty_intersection(tmp_path):
    (tmp_path / "sess_A01_bob.json").write_text(json.dumps([{"window_index": 0, "label": "correct"}]))
    (tmp_path / "sess_A01_amy.json").write_text(json.dumps([{"window_index": 1, "label": "incorrect"}]))

    result = report_all_sessions(str(tmp_path))

    assert result == {}


def test_overall_kappa_averages_per_session_values():
    assert overall_kappa({"sess_A01": 1.0, "sess_A02": 0.5}) == 0.75


def test_overall_kappa_empty_report_is_zero():
    assert overall_kappa({}) == 0.0
