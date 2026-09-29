from unittest.mock import patch

import evaluate_calibration as ec


def _touch(tmp_path, *names):
    paths = [tmp_path / n for n in names]
    for p in paths:
        p.touch()
    return paths


def test_group_frames_by_subject_groups_correctly(tmp_path):
    files = _touch(tmp_path, "person01100-15+30.jpg", "person01101+00+00.jpg", "person02100-15+30.jpg")
    grouped = ec.group_frames_by_subject(files)
    assert set(grouped.keys()) == {"person01", "person02"}
    assert len(grouped["person01"]) == 2


def test_group_frames_by_subject_skips_unparseable_names(tmp_path):
    files = _touch(tmp_path, "person01101+00+00.jpg", "notes.jpg")
    assert set(ec.group_frames_by_subject(files)) == {"person01"}


def test_evaluate_subject_uses_frontal_frame_as_baseline(tmp_path):
    # Subject's neutral head_ratio is 0.7, not 0.5.
    frontal, near, away = _touch(
        tmp_path, "person01101+00+00.jpg", "person01102+00+10.jpg", "person01103+00+60.jpg"
    )
    ratios_by_file = {
        frontal: {"head_ratio": 0.7, "gaze_ratio": 0.5},
        near: {"head_ratio": 0.72, "gaze_ratio": 0.5},
        away: {"head_ratio": 0.15, "gaze_ratio": 0.5},
    }
    with patch.object(ec, "_raw_ratios_for_file", side_effect=lambda p: ratios_by_file[p]):
        result = ec.evaluate_subject([frontal, near, away])

    assert result["subject"] == "person01"
    assert result["n_frames"] == 2
    assert 0.0 <= result["balanced_accuracy_calibrated"] <= 1.0
    assert 0.0 <= result["balanced_accuracy_uncalibrated"] <= 1.0
    # near: uncalibrated dev = 0.44 (attentive); away: dev 0.7 (away) -> both right.
    # With this neutral offset calibration must do at least as well.
    assert result["balanced_accuracy_calibrated"] == 1.0
    assert result["balanced_accuracy_calibrated"] >= result["balanced_accuracy_uncalibrated"]


def test_evaluate_subject_calibration_fixes_offset_neutral(tmp_path):
    frontal, near, away = _touch(
        tmp_path, "person01101+00+00.jpg", "person01102+00+10.jpg", "person01103+00+60.jpg"
    )
    ratios_by_file = {
        frontal: {"head_ratio": 0.85, "gaze_ratio": 0.5},
        near: {"head_ratio": 0.9, "gaze_ratio": 0.5},   # uncal dev 0.8 -> wrongly "away"
        away: {"head_ratio": 0.2, "gaze_ratio": 0.5},
    }
    with patch.object(ec, "_raw_ratios_for_file", side_effect=lambda p: ratios_by_file[p]):
        result = ec.evaluate_subject([frontal, near, away])
    assert result["balanced_accuracy_uncalibrated"] == 0.5
    assert result["balanced_accuracy_calibrated"] == 1.0


def test_evaluate_subject_returns_none_without_frontal_frame(tmp_path):
    frames = _touch(tmp_path, "person01102+00+10.jpg", "person01103+00+60.jpg")
    with patch.object(ec, "_raw_ratios_for_file", return_value={"head_ratio": 0.5, "gaze_ratio": 0.5}):
        assert ec.evaluate_subject(frames) is None


def test_evaluate_subject_returns_none_for_single_class(tmp_path):
    frames = _touch(
        tmp_path, "person01101+00+00.jpg", "person01102+00+10.jpg", "person01103+00-10.jpg"
    )
    with patch.object(ec, "_raw_ratios_for_file", return_value={"head_ratio": 0.5, "gaze_ratio": 0.5}):
        assert ec.evaluate_subject(frames) is None


def test_evaluate_subject_skips_ambiguous_and_undetected_frames(tmp_path):
    frontal, ambiguous, undetected, near, away = _touch(
        tmp_path,
        "person01101+00+00.jpg",
        "person01102+00+30.jpg",  # 30 degrees: excluded band
        "person01103+00+15.jpg",  # no face detected
        "person01104+00+10.jpg",
        "person01105+00+60.jpg",
    )
    ratios_by_file = {
        frontal: {"head_ratio": 0.5, "gaze_ratio": 0.5},
        ambiguous: {"head_ratio": 0.7, "gaze_ratio": 0.5},
        undetected: None,
        near: {"head_ratio": 0.55, "gaze_ratio": 0.5},
        away: {"head_ratio": 0.05, "gaze_ratio": 0.5},
    }
    with patch.object(ec, "_raw_ratios_for_file", side_effect=lambda p: ratios_by_file[p]):
        result = ec.evaluate_subject([frontal, ambiguous, undetected, near, away])
    assert result["n_frames"] == 2


def test_balanced_accuracy_and_undefined_case():
    assert ec.balanced_accuracy([True, True, False, False], [True, False, False, False]) == 0.75
    assert ec.balanced_accuracy([True, True], [True, True]) is None


def test_summarize_pools_frames_and_skips_nothing_undefined():
    a = {
        "balanced_accuracy_uncalibrated": 0.5,
        "balanced_accuracy_calibrated": 1.0,
        "outcomes": [[True, False, True], [False, False, False]],
    }
    b = {
        "balanced_accuracy_uncalibrated": 1.0,
        "balanced_accuracy_calibrated": 1.0,
        "outcomes": [[True, True, True], [False, False, False]],
    }
    s = ec.summarize([a, b])
    assert s["mean_subject_balanced_accuracy_uncalibrated"] == 0.75
    assert s["mean_subject_balanced_accuracy_calibrated"] == 1.0
    assert s["pooled_frames_balanced_accuracy_calibrated"] == 1.0
    assert s["pooled_frames_balanced_accuracy_uncalibrated"] == 0.75


def test_main_writes_results_json(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    frames = _touch(
        corpus, "person01101+00+00.jpg", "person01102+00+10.jpg", "person01103+00+60.jpg"
    )
    ratios_by_file = {
        frames[0]: {"head_ratio": 0.5, "gaze_ratio": 0.5},
        frames[1]: {"head_ratio": 0.52, "gaze_ratio": 0.5},
        frames[2]: {"head_ratio": 0.1, "gaze_ratio": 0.5},
    }
    with patch.object(ec, "_raw_ratios_for_file", side_effect=lambda p: ratios_by_file[p]):
        assert ec.main(["--corpus", str(corpus)]) == 0
    assert (corpus / "evaluation-results.json").exists()
    assert "person01" in capsys.readouterr().out


def test_main_missing_corpus_returns_error(tmp_path):
    assert ec.main(["--corpus", str(tmp_path / "nope")]) == 2
