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
    top_correlation = max(r["correlation"] for r in results)
    assert top_correlation > 0.9
    # Multiple weight triples can tie at the max correlation (Pearson
    # correlation is affine-invariant, and with only 3 ground-truth points
    # many weight combinations produce affinely-equivalent fused-score
    # series). Among the tied-best triples, eyes_open should still be able
    # to dominate — i.e. a high-w_eyes solution should be among the ties,
    # not excluded by them.
    best_w_eyes_among_ties = max(
        r["weights"][0] for r in results
        if abs(r["correlation"] - top_correlation) < 1e-6
    )
    assert best_w_eyes_among_ties > 0.3


def test_sweep_fusion_weights_returns_sorted_by_correlation():
    video_scores = [0.9, 0.2, 0.8]
    speech_scores = [0.85, 0.25, 0.75]
    results = sweep_fusion_weights(video_scores, speech_scores, step=0.1)
    assert results == sorted(results, key=lambda r: -r["correlation"])
    assert all(abs(r["w_video"] + r["w_speech"] - 1.0) < 1e-9 for r in results)


# ===========================================================================
# Corpus mode: the labeled-frame experiment.
#
# Everything below runs under a plain system python: no cv2, no mediapipe.
# The detector is injected, so score_corpus_frames is exercised without the
# real model.
# ===========================================================================

import json

from weight_search import (
    LABEL_TARGETS,
    best_threshold,
    choose_weights,
    metrics_at_threshold,
    evaluate_corpus_weights,
    format_corpus_report,
    label_to_target,
    load_corpus_manifest,
    mean_score_per_label,
    roc_auc,
    score_corpus_frames,
    sweep_corpus_weights,
    _weight_grid,
)


# --- label -> target mapping ------------------------------------------------

def test_label_to_target_maps_attentive_high_and_inattentive_low():
    assert label_to_target("attentive") == 1.0
    for label in ("eyes_closed", "looking_away", "gaze_off"):
        assert label_to_target(label) == 0.0


def test_absent_is_excluded_not_scored_zero():
    # A frame with no face has no landmarks, so there is no score to fit; the
    # service already skips such frames rather than treating them as zero.
    assert label_to_target("absent") is None


def test_label_to_target_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unknown label"):
        label_to_target("distracted")


def test_every_label_target_is_none_or_in_unit_range():
    for target in LABEL_TARGETS.values():
        assert target is None or 0.0 <= target <= 1.0


# --- weight grid ------------------------------------------------------------

def test_weight_grid_triples_sum_to_one():
    grid = _weight_grid(0.25)
    assert grid
    assert all(abs(sum(w) - 1.0) < 1e-9 for w in grid)
    assert (1.0, 0.0, 0.0) in grid


def test_weight_grid_is_finer_with_smaller_step():
    assert len(_weight_grid(0.1)) > len(_weight_grid(0.25))


# --- ROC-AUC ----------------------------------------------------------------

def test_roc_auc_perfect_separation():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_roc_auc_inverted_separation():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_roc_auc_all_ties_is_chance():
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.5)


def test_roc_auc_single_tie_counts_as_half():
    # pairs: (0.9,0.4) win, (0.9,0.5) win, (0.5,0.4) win, (0.5,0.5) tie -> 3.5/4
    assert roc_auc([0.9, 0.5, 0.5, 0.4], [1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.875)


def test_roc_auc_requires_both_classes():
    with pytest.raises(ValueError, match="at least one positive"):
        roc_auc([0.1, 0.2], [1.0, 1.0])


def test_roc_auc_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        roc_auc([0.1, 0.2], [1.0])


# --- threshold selection ----------------------------------------------------

def test_best_threshold_separates_cleanly_and_sits_in_the_gap():
    result = best_threshold([0.9, 0.85, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0])
    assert 0.2 < result["threshold"] < 0.85
    assert result["tpr"] == pytest.approx(1.0)
    assert result["fpr"] == pytest.approx(0.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["youden_j"] == pytest.approx(1.0)


def test_best_threshold_prefers_balanced_accuracy_over_plain_accuracy():
    # 8 inattentive, 2 attentive. Always predicting "inattentive" scores 0.8
    # plain accuracy; the balanced objective must not pick that degenerate cut.
    scores = [0.9, 0.8] + [0.3] * 8
    targets = [1.0, 1.0] + [0.0] * 8
    result = best_threshold(scores, targets)
    assert result["tpr"] == pytest.approx(1.0)
    assert result["threshold"] < 0.8


def test_best_threshold_reported_metrics_match_the_returned_threshold():
    scores = [0.9, 0.6, 0.55, 0.2]
    targets = [1.0, 1.0, 0.0, 0.0]
    result = best_threshold(scores, targets)
    threshold = result["threshold"]
    tp = sum(1 for s, t in zip(scores, targets) if s >= threshold and t >= 0.5)
    fp = sum(1 for s, t in zip(scores, targets) if s >= threshold and t < 0.5)
    assert result["tpr"] == pytest.approx(tp / 2)
    assert result["fpr"] == pytest.approx(fp / 2)


def test_best_threshold_requires_both_classes():
    with pytest.raises(ValueError, match="at least one positive"):
        best_threshold([0.1, 0.9], [0.0, 0.0])


# --- per-label means --------------------------------------------------------

def test_mean_score_per_label_groups_and_averages():
    means = mean_score_per_label([1.0, 0.5, 0.2], ["attentive", "attentive", "gaze_off"])
    assert means == {"attentive": pytest.approx(0.75), "gaze_off": pytest.approx(0.2)}


def test_mean_score_per_label_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        mean_score_per_label([1.0], ["attentive", "gaze_off"])


# --- corpus loading ---------------------------------------------------------

def _write_corpus(tmp_path, rows, make_frames=True):
    for row in rows:
        if make_frames and "frame" in row:
            frame = tmp_path / row["frame"]
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"not-a-real-jpeg")
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


TINY_MANIFEST = [
    {"frame": "attentive/attentive_1_0000.jpg", "label": "attentive", "timestamp": "2026-09-26T10:00:00.000+00:00"},
    {"frame": "eyes_closed/eyes_closed_1_0000.jpg", "label": "eyes_closed", "timestamp": "2026-09-26T10:00:05.000+00:00"},
    {"frame": "absent/absent_1_0000.jpg", "label": "absent", "timestamp": "2026-09-26T10:00:10.000+00:00"},
]


def test_load_corpus_manifest_reads_labels_and_targets(tmp_path):
    corpus = _write_corpus(tmp_path, TINY_MANIFEST)
    records = load_corpus_manifest(corpus)
    assert [r["label"] for r in records] == ["attentive", "eyes_closed", "absent"]
    assert [r["target"] for r in records] == [1.0, 0.0, None]
    assert records[0]["frame"].is_file()
    assert records[0]["timestamp"].startswith("2026-09-26")


def test_load_corpus_manifest_skips_blank_lines(tmp_path):
    corpus = _write_corpus(tmp_path, TINY_MANIFEST)
    manifest = corpus / "labels.jsonl"
    manifest.write_text("\n" + manifest.read_text() + "\n\n", encoding="utf-8")
    assert len(load_corpus_manifest(corpus)) == 3


def test_load_corpus_manifest_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="No labels.jsonl"):
        load_corpus_manifest(tmp_path)


def test_load_corpus_manifest_rejects_unknown_label(tmp_path):
    corpus = _write_corpus(tmp_path, [{"frame": "x/y.jpg", "label": "bored"}])
    with pytest.raises(ValueError, match="Unknown label"):
        load_corpus_manifest(corpus)


def test_load_corpus_manifest_rejects_missing_frame_file(tmp_path):
    corpus = _write_corpus(tmp_path, [{"frame": "attentive/gone.jpg", "label": "attentive"}], make_frames=False)
    with pytest.raises(FileNotFoundError, match="missing frame"):
        load_corpus_manifest(corpus)


def test_load_corpus_manifest_rejects_malformed_json(tmp_path):
    (tmp_path / "labels.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_corpus_manifest(tmp_path)


def test_load_corpus_manifest_rejects_row_without_label(tmp_path):
    corpus = _write_corpus(tmp_path, [{"frame": "attentive/a.jpg"}])
    with pytest.raises(ValueError, match="missing label"):
        load_corpus_manifest(corpus)


def test_load_corpus_manifest_rejects_empty_manifest(tmp_path):
    (tmp_path / "labels.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no labeled frames"):
        load_corpus_manifest(tmp_path)


# --- scoring the corpus with an injected detector ---------------------------

def _fake_detector_factory(by_label):
    def detector(frame_bytes):
        return by_label(frame_bytes.decode())
    return detector


def _synthetic_corpus(tmp_path):
    """A corpus whose 'JPEG' bytes are just the label, so a fake detector can
    return plausible sub-signals without any image decoding."""
    rows = []
    for label, count in (("attentive", 3), ("eyes_closed", 2), ("looking_away", 2), ("gaze_off", 2), ("absent", 2)):
        for i in range(count):
            name = f"{label}/{label}_{i}.jpg"
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(label, encoding="utf-8")
            rows.append({"frame": name, "label": label, "timestamp": "2026-09-26T10:00:00.000+00:00"})
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


# Each inattentive state breaks exactly ONE sub-signal and matches `attentive`
# on the other two — the same identifiability property the real capture script
# is designed to produce.
_SYNTHETIC_SIGNALS = {
    "attentive": {"eyes_open": 0.95, "head_deviation": 0.05, "gaze_centered": 0.95},
    "eyes_closed": {"eyes_open": 0.02, "head_deviation": 0.05, "gaze_centered": 0.95},
    "looking_away": {"eyes_open": 0.95, "head_deviation": 0.92, "gaze_centered": 0.95},
    "gaze_off": {"eyes_open": 0.95, "head_deviation": 0.05, "gaze_centered": 0.05},
    "absent": None,
}


def test_score_corpus_frames_excludes_absent_and_reports_coverage(tmp_path):
    records = load_corpus_manifest(_synthetic_corpus(tmp_path))
    samples, coverage = score_corpus_frames(
        records, detector=_fake_detector_factory(_SYNTHETIC_SIGNALS.get)
    )
    assert coverage["frames_total"] == 11
    assert coverage["used"] == 9  # the two absent frames are excluded
    assert len(samples) == 9
    assert "absent" not in {s["label"] for s in samples}
    assert coverage["absent_correctly_undetected"] == 2
    assert coverage["absent_falsely_detected"] == 0
    assert coverage["no_detection_on_scoring_label"] == 0
    assert coverage["per_label_total"]["attentive"] == 3


def test_score_corpus_frames_counts_a_dropped_scoring_frame(tmp_path):
    records = load_corpus_manifest(_synthetic_corpus(tmp_path))

    calls = {"n": 0}

    def flaky(frame_bytes):
        label = frame_bytes.decode()
        if label == "attentive":
            calls["n"] += 1
            if calls["n"] == 1:
                return None  # detector missed a face it should have found
        return _SYNTHETIC_SIGNALS.get(label)

    samples, coverage = score_corpus_frames(records, detector=flaky)
    assert coverage["no_detection_on_scoring_label"] == 1
    assert coverage["used"] == 8
    assert len(samples) == 8


def test_score_corpus_frames_flags_a_face_found_in_an_absent_frame(tmp_path):
    records = load_corpus_manifest(_synthetic_corpus(tmp_path))
    always = _fake_detector_factory(lambda label: _SYNTHETIC_SIGNALS["attentive"])
    _, coverage = score_corpus_frames(records, detector=always)
    assert coverage["absent_falsely_detected"] == 2
    assert coverage["absent_correctly_undetected"] == 0


# --- the sweep itself -------------------------------------------------------

def _synthetic_samples(tmp_path):
    records = load_corpus_manifest(_synthetic_corpus(tmp_path))
    samples, _ = score_corpus_frames(records, detector=_fake_detector_factory(_SYNTHETIC_SIGNALS.get))
    return samples


def test_evaluate_corpus_weights_reports_the_full_metric_set(tmp_path):
    row = evaluate_corpus_weights(_synthetic_samples(tmp_path), (1 / 3, 1 / 3, 1 / 3))
    assert {
        "weights", "auc", "balanced_accuracy", "separation", "pearson_r",
        "threshold", "mean_score_per_label", "per_split", "mean_split_auc",
        "min_split_auc", "min_split_balanced_accuracy",
        "per_split_at_pooled_threshold",
    } <= set(row)
    assert 0.0 <= row["auc"] <= 1.0
    assert row["mean_score_per_label"]["attentive"] > row["mean_score_per_label"]["eyes_closed"]


def test_sweep_corpus_weights_is_sorted_by_the_stated_objective(tmp_path):
    results = sweep_corpus_weights(_synthetic_samples(tmp_path), step=0.25)
    keys = [(-r["mean_split_auc"], -r["min_split_balanced_accuracy"], -r["separation"]) for r in results]
    assert keys == sorted(keys)
    assert all(abs(sum(r["weights"]) - 1.0) < 1e-9 for r in results)


def test_sweep_corpus_weights_needs_all_three_signals_for_perfect_separation(tmp_path):
    # Each inattentive state breaks exactly one sub-signal, so a weighting that
    # zeroes any one signal cannot separate the state that breaks it. The best
    # triple must therefore put non-zero weight on all three.
    results = sweep_corpus_weights(_synthetic_samples(tmp_path), step=0.25)
    best = results[0]
    assert best["auc"] == pytest.approx(1.0)
    assert all(w > 0.0 for w in best["weights"])

    degenerate = [r for r in results if r["weights"] == (1.0, 0.0, 0.0)][0]
    assert degenerate["auc"] < 1.0


def test_sweep_corpus_weights_rejects_an_empty_corpus():
    with pytest.raises(ValueError, match="No scored samples"):
        sweep_corpus_weights([], step=0.5)


def test_format_corpus_report_mentions_coverage_weights_and_threshold(tmp_path):
    records = load_corpus_manifest(_synthetic_corpus(tmp_path))
    samples, coverage = score_corpus_frames(records, detector=_fake_detector_factory(_SYNTHETIC_SIGNALS.get))
    report = format_corpus_report(sweep_corpus_weights(samples, step=0.25), coverage, top_n=3)
    assert "CHOSEN weights" in report
    assert "CHOSEN threshold" in report
    assert "Detector coverage per label" in report
    assert "attentive" in report


# --- multi-split evaluation, threshold reuse and plateau tie-breaking -------

def _two_split_samples():
    """Two splits, each of which can only speak to one sub-signal.

    `headish` varies head_deviation alone; `eyeish` varies eyes_open alone.
    Neither says anything about gaze_centered — exactly the situation the
    public-dataset fit is in.
    """
    samples = []
    for i in range(4):
        samples.append({
            "split": "headish", "label": "attentive", "target": 1.0, "frame": f"h{i}",
            "raw_signals": {"eyes_open": 0.9, "head_deviation": 0.05, "gaze_centered": 0.5},
        })
        samples.append({
            "split": "headish", "label": "looking_away", "target": 0.0, "frame": f"ha{i}",
            "raw_signals": {"eyes_open": 0.9, "head_deviation": 0.95, "gaze_centered": 0.5},
        })
    for i in range(4):
        samples.append({
            "split": "eyeish", "label": "attentive", "target": 1.0, "frame": f"e{i}",
            "raw_signals": {"eyes_open": 0.9, "head_deviation": 0.5, "gaze_centered": 0.5},
        })
        samples.append({
            "split": "eyeish", "label": "eyes_closed", "target": 0.0, "frame": f"ec{i}",
            "raw_signals": {"eyes_open": 0.05, "head_deviation": 0.5, "gaze_centered": 0.5},
        })
    return samples


def test_evaluate_corpus_weights_reports_each_split_separately():
    row = evaluate_corpus_weights(_two_split_samples(), (0.5, 0.5, 0.0))
    assert set(row["per_split"]) == {"headish", "eyeish"}
    assert row["per_split"]["headish"]["n"] == 8
    assert row["mean_split_auc"] == pytest.approx(1.0)


def test_evaluate_corpus_weights_a_single_signal_weighting_abandons_one_split():
    # All weight on eyes: the head split becomes unrankable (every frame scores
    # the same), so its AUC collapses to chance and drags the mean down.
    row = evaluate_corpus_weights(_two_split_samples(), (1.0, 0.0, 0.0))
    assert row["per_split"]["eyeish"]["auc"] == pytest.approx(1.0)
    assert row["per_split"]["headish"]["auc"] == pytest.approx(0.5)
    assert row["mean_split_auc"] == pytest.approx(0.75)
    assert row["min_split_balanced_accuracy"] < 1.0


def test_evaluate_corpus_weights_marks_a_single_class_split():
    samples = [s for s in _two_split_samples() if s["split"] == "headish"]
    samples += [{
        "split": "onesided", "label": "attentive", "target": 1.0, "frame": "x",
        "raw_signals": {"eyes_open": 0.9, "head_deviation": 0.1, "gaze_centered": 0.5},
    }]
    row = evaluate_corpus_weights(samples, (0.3, 0.4, 0.3))
    assert row["per_split"]["onesided"]["single_class"] is True
    # A single-class split contributes no AUC, so the objective ignores it.
    assert row["mean_split_auc"] == pytest.approx(row["per_split"]["headish"]["auc"])


def test_sweep_prefers_a_triple_that_serves_both_splits():
    results = sweep_corpus_weights(_two_split_samples(), step=0.25)
    best = results[0]
    assert best["mean_split_auc"] == pytest.approx(1.0)
    assert best["weights"][0] > 0.0 and best["weights"][1] > 0.0


def test_metrics_at_threshold_scores_a_split_at_an_external_cut():
    metrics = metrics_at_threshold([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0], 0.5)
    assert metrics["tpr"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(0.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["threshold"] == 0.5


def test_metrics_at_threshold_reports_a_bad_cut_honestly():
    metrics = metrics_at_threshold([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0], 0.05)
    assert metrics["tpr"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_choose_weights_takes_the_lowest_evidence_free_weight_on_a_plateau():
    results = sweep_corpus_weights(_two_split_samples(), step=0.25)
    chosen = choose_weights(results, tolerance=0.005, evidence_free_index=2)
    plateau = [r for r in results if r["mean_split_auc"] >= max(x["mean_split_auc"] for x in results) - 0.005]
    best_min_ba = max(r["min_split_balanced_accuracy"] for r in plateau)
    eligible = [r for r in plateau if r["min_split_balanced_accuracy"] == best_min_ba]
    assert chosen["weights"][2] == min(r["weights"][2] for r in eligible)


def test_choose_weights_never_leaves_the_plateau():
    results = sweep_corpus_weights(_two_split_samples(), step=0.25)
    best = max(r["mean_split_auc"] for r in results)
    assert choose_weights(results, tolerance=0.005)["mean_split_auc"] >= best - 0.005


def test_choose_weights_rejects_an_empty_sweep():
    with pytest.raises(ValueError, match="No sweep results"):
        choose_weights([])


def test_load_corpus_manifest_names_the_split_after_the_directory(tmp_path):
    corpus = _write_corpus(tmp_path / "head_pose", TINY_MANIFEST)
    records = load_corpus_manifest(corpus)
    assert {r["split"] for r in records} == {"head_pose"}
    assert {r["split"] for r in load_corpus_manifest(corpus, split="custom")} == {"custom"}


def test_choose_weights_floor_keeps_the_evidence_free_signal_alive():
    results = sweep_corpus_weights(_two_split_samples(), step=0.25)
    chosen = choose_weights(results, tolerance=0.005, evidence_free_index=2, evidence_free_floor=0.25)
    assert chosen["weights"][2] >= 0.25
    assert chosen["evidence_free_floor_applied"] is True


def test_choose_weights_without_a_floor_may_zero_the_evidence_free_signal():
    results = sweep_corpus_weights(_two_split_samples(), step=0.25)
    chosen = choose_weights(results, tolerance=0.005, evidence_free_index=2)
    assert chosen["weights"][2] == 0.0
    assert chosen["evidence_free_floor_applied"] is False


def test_choose_weights_ignores_a_floor_that_would_empty_the_plateau():
    results = sweep_corpus_weights(_two_split_samples(), step=0.25)
    chosen = choose_weights(results, tolerance=0.0, evidence_free_index=2, evidence_free_floor=1.0)
    assert chosen["evidence_free_floor_applied"] is False
    best = max(r["mean_split_auc"] for r in results)
    assert chosen["mean_split_auc"] == pytest.approx(best)


def test_choose_weights_reports_the_plateau_size():
    results = sweep_corpus_weights(_two_split_samples(), step=0.25)
    chosen = choose_weights(results, tolerance=0.005)
    assert 1 <= chosen["plateau_size"] <= len(results)


def test_weight_grid_has_no_negative_zero():
    # round(1.0 - 0.55 - 0.45, 4) yields -0.0, which prints as "-0.00" in the
    # report and looks like a negative weight.
    assert all(str(w) != "-0.0" for triple in _weight_grid(0.05) for w in triple)


def test_compact_result_keeps_the_headline_metrics_only():
    from weight_search import compact_result

    row = compact_result(evaluate_corpus_weights(_two_split_samples(), (0.5, 0.5, 0.0)))
    assert row["weights"] == [0.5, 0.5, 0.0]
    assert set(row["per_split_auc"]) == {"headish", "eyeish"}
    assert isinstance(row["threshold"], float)
    assert "per_split" not in row


# --- group-disjoint holdout -------------------------------------------------

def _grouped_samples(n_groups=5, per_group=4):
    samples = []
    for g in range(n_groups):
        for i in range(per_group):
            samples.append({
                "split": "headish", "group": f"person{g:02d}", "label": "attentive",
                "target": 1.0, "frame": f"a{g}{i}",
                "raw_signals": {"eyes_open": 0.9, "head_deviation": 0.05, "gaze_centered": 0.5},
            })
            samples.append({
                "split": "headish", "group": f"person{g:02d}", "label": "looking_away",
                "target": 0.0, "frame": f"b{g}{i}",
                "raw_signals": {"eyes_open": 0.9, "head_deviation": 0.95, "gaze_centered": 0.5},
            })
    return samples


def test_partition_by_group_keeps_groups_disjoint():
    from weight_search import partition_by_group

    fit, holdout = partition_by_group(_grouped_samples(), 0.4)
    assert fit and holdout
    assert not ({s["group"] for s in fit} & {s["group"] for s in holdout})
    assert len(fit) + len(holdout) == 40


def test_partition_by_group_is_deterministic():
    from weight_search import partition_by_group

    first = partition_by_group(_grouped_samples(), 0.4)[1]
    second = partition_by_group(_grouped_samples(), 0.4)[1]
    assert [s["frame"] for s in first] == [s["frame"] for s in second]


def test_partition_by_group_zero_fraction_holds_nothing_out():
    from weight_search import partition_by_group

    fit, holdout = partition_by_group(_grouped_samples(), 0.0)
    assert len(fit) == 40 and holdout == []


def test_partition_by_group_never_empties_the_fit_half():
    from weight_search import partition_by_group

    fit, holdout = partition_by_group(_grouped_samples(n_groups=2), 0.99)
    assert fit and holdout


def test_partition_by_group_leaves_a_single_group_split_in_the_fit_half():
    from weight_search import partition_by_group

    samples = _grouped_samples(n_groups=1)
    fit, holdout = partition_by_group(samples, 0.5)
    assert len(fit) == len(samples)
    assert holdout == []


def test_partition_by_group_rejects_an_impossible_fraction():
    from weight_search import partition_by_group

    with pytest.raises(ValueError, match="holdout_fraction"):
        partition_by_group(_grouped_samples(), 1.0)


def test_evaluate_holdout_uses_the_threshold_it_was_given():
    from weight_search import evaluate_holdout, partition_by_group

    _, holdout_samples = partition_by_group(_grouped_samples(), 0.4)
    report = evaluate_holdout(holdout_samples, (0.5, 0.45, 0.05), 0.5)
    assert report["n"] == len(holdout_samples)
    assert report["at_fitted_threshold"]["threshold"] == 0.5
    assert report["auc"] == pytest.approx(1.0)
    assert set(report["per_split"]) == {"headish"}


def test_evaluate_holdout_omits_auc_when_one_class_is_missing():
    from weight_search import evaluate_holdout

    samples = [s for s in _grouped_samples() if s["label"] == "attentive"][:4]
    report = evaluate_holdout(samples, (0.5, 0.45, 0.05), 0.5)
    assert "auc" not in report
    assert report["at_fitted_threshold"]["tpr"] == pytest.approx(1.0)


def test_partition_by_group_keeps_both_classes_in_the_holdout():
    """Single-class groups (the eye dataset's shards) must not produce a
    holdout of one class, where false-positive rate is undefined."""
    from weight_search import partition_by_group

    samples = []
    for g in range(4):
        for i in range(5):
            samples.append({
                "split": "eyeish", "group": f"open_{g}", "label": "attentive", "target": 1.0,
                "frame": f"o{g}{i}",
                "raw_signals": {"eyes_open": 0.9, "head_deviation": 0.2, "gaze_centered": 0.5},
            })
            samples.append({
                "split": "eyeish", "group": f"closed_{g}", "label": "eyes_closed", "target": 0.0,
                "frame": f"c{g}{i}",
                "raw_signals": {"eyes_open": 0.05, "head_deviation": 0.2, "gaze_centered": 0.5},
            })
    _, holdout = partition_by_group(samples, 0.3)
    assert {s["target"] for s in holdout} == {0.0, 1.0}
