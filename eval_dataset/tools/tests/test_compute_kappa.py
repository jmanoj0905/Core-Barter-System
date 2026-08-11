import json
from eval_dataset.tools.compute_kappa import load_session_annotations, report_all_sessions


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
