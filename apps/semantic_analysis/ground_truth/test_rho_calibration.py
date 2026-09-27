"""Tests for the RHO calibration mechanism (ground_truth only — ships nothing).

These pin the mechanism of spec 7.1 exactly as written: R starts at UPPER, only
an at-or-above-threshold window may raise R, R never decreases and is capped at
0.80. They also pin the error-mix scoring contract (excluded windows count in
neither class), the flat-LOWER control on every grid row, the WER alignment
behaviours (by ReplayWindow.index, never positionally), and per-fold argmax
reporting.
"""

import pytest

import rho_calibration as rc
from rho_calibration import (
    ErrorMix,
    ScoredSession,
    align_wer_runs,
    evaluate,
    evaluate_flat,
    grid_search,
    running_thresholds,
)

from eval_dataset.tools.replay import ReplayWindow, TurnContribution


def window(index, seconds=25.0, text="some teacher speech here"):
    return ReplayWindow(
        index=index,
        text=text,
        raw_text=text,
        teacher_seconds_start=0.0,
        teacher_seconds_end=seconds,
        contributions=(TurnContribution(turn_index=index, seconds=seconds),),
    )


def session(similarities, gold, **kw):
    fields = dict(
        session_id=kw.get("session_id", "sess_TEST"),
        topic=kw.get("topic", "a topic"),
        scope=kw.get("scope", ""),
        category=kw.get("category", "gradual_drift"),
        author=kw.get("author", "agent:opus-5"),
        durations=kw.get("durations", "synthetic"),
        window_indices=tuple(kw.get("window_indices", range(1, len(similarities) + 1))),
        similarities=tuple(similarities),
        gold=tuple(gold),
    )
    return ScoredSession(**fields)


# --- the mechanism -------------------------------------------------------


def test_R_starts_at_upper():
    assert running_thresholds([0.05], rho=0.45)[0] == pytest.approx(0.45 * 0.36)


def test_below_threshold_window_does_not_raise_R():
    # A window below the current thr cannot become the ceiling it is judged
    # against, however high its cosine. thr = rho * R, so this clause only
    # bites when rho > 1 (see the module docstring and the results JSON
    # limitation): for rho < 1 any sim >= R is automatically >= thr. rho=3.0
    # makes the rule observable, which is the only way to pin it as written.
    thresholds = running_thresholds([0.9, 0.9, 0.9], rho=3.0)
    assert thresholds == [pytest.approx(3.0 * 0.36)] * 3

    # And the contrast: at the same R, a window that *does* reach thr raises it.
    raised = running_thresholds([0.9, 0.9], rho=0.45)
    assert raised[0] == pytest.approx(0.45 * 0.36)
    assert raised[1] == pytest.approx(0.45 * 0.80)


def test_R_never_decreases_within_a_session():
    sims = [0.70, 0.10, 0.50, 0.05, 0.62, 0.30]
    thresholds = running_thresholds(sims, rho=0.45)
    assert thresholds == sorted(thresholds), thresholds
    # and the same for a strictly descending session
    descending = running_thresholds([0.75, 0.60, 0.45, 0.30, 0.15], rho=0.50)
    assert descending == sorted(descending), descending


def test_R_is_capped_at_080():
    thresholds = running_thresholds([0.95, 0.99, 0.10], rho=0.50)
    assert thresholds[0] == pytest.approx(0.50 * 0.36)
    assert thresholds[1] == pytest.approx(0.50 * 0.80)
    assert thresholds[2] == pytest.approx(0.50 * 0.80)
    assert max(thresholds) <= 0.50 * rc.R_CAP + 1e-12


def test_R_stays_at_the_floor_when_no_window_ever_reaches_thr():
    # The depressed_baseline case. thr is constant at rho*UPPER = 0.162 for the
    # whole session, and evaluate() must score such a session normally.
    sims = [0.10, 0.08, 0.15, 0.02, 0.16]
    thresholds = running_thresholds(sims, rho=0.45)
    assert thresholds == [pytest.approx(0.162)] * 5

    depressed = session(sims, ["on_topic"] * 5, category="depressed_baseline")
    mix = evaluate([depressed], rho=0.45)
    assert mix.on_topic_total == 5
    assert mix.digression_total == 0
    assert mix.excluded == 0
    # every window sits below the floor threshold, so all five are false flags
    assert mix.false_digressions == 5


# --- scoring -------------------------------------------------------------


def test_error_mix_counts_excluded_windows_in_neither_class():
    # gold_label returns None for boundary windows; those are excluded and
    # belong to neither class.
    mixed = session(
        [0.50, 0.05, 0.40, 0.02],
        ["on_topic", "off_topic", None, None],
    )
    mix = evaluate([mixed], rho=0.45)
    assert mix.on_topic_total == 1
    assert mix.digression_total == 1
    assert mix.excluded == 2
    assert mix.on_topic_total + mix.digression_total == 4 - mix.excluded
    assert mix.caught + mix.missed == mix.digression_total
    assert mix.false_digressions <= mix.on_topic_total


def test_flat_lower_control_is_reported_alongside_every_rho():
    sessions = [
        session([0.50, 0.05, 0.30], ["on_topic", "off_topic", "on_topic"], session_id="s1"),
        session([0.12, 0.60, 0.02], ["off_topic", "on_topic", "off_topic"], session_id="s2",
                topic="another topic"),
    ]
    result = grid_search(sessions, folds="session")
    assert result.grid, "grid must not be empty"
    for rho, row in result.grid.items():
        assert "relative" in row and "flat_lower" in row, rho
        assert row["flat_lower"]["threshold"] == pytest.approx(rc.LOWER)
    # the control is the same on every row: it does not depend on rho
    controls = {json_key(row["flat_lower"]["error_mix"]) for row in result.grid.values()}
    assert len(controls) == 1
    assert evaluate_flat(sessions, threshold=rc.LOWER).as_dict() == \
        next(iter(result.grid.values()))["flat_lower"]["error_mix"]


def json_key(mix_dict):
    return tuple(sorted(mix_dict.items()))


# --- WER alignment (leg not run; behaviour unit-tested on hand-built lists) ---


def test_wer_comparison_aligns_by_window_index_not_position():
    baseline = [window(1), window(2), window(3)]
    # WER-30 dropped window 2: it fell under MIN_CONTENT_TOKENS.
    degraded = [window(1), window(3)]
    alignment = align_wer_runs(baseline, degraded)
    assert [(b.index, d.index) for b, d in alignment.matched] == [(1, 1), (3, 3)]
    # a positional zip would have paired baseline 2 with degraded 3
    assert (2, 3) not in [(b.index, d.index) for b, d in alignment.matched]
    assert alignment.baseline_only == (2,)
    assert alignment.degraded_only == ()


def test_wer_comparison_reports_window_count_mismatches():
    baseline = [window(1), window(2), window(3), window(4)]
    degraded = [window(1), window(3), window(5)]
    alignment = align_wer_runs(baseline, degraded)
    assert alignment.baseline_count == 4
    assert alignment.degraded_count == 3
    assert alignment.count_mismatch == -1
    assert alignment.baseline_only == (2, 4)
    assert alignment.degraded_only == (5,)
    assert len(alignment.matched) == 2

    same = align_wer_runs(baseline, baseline)
    assert same.count_mismatch == 0
    assert same.baseline_only == () and same.degraded_only == ()


# --- folds ---------------------------------------------------------------


def test_grid_search_reports_argmax_per_fold():
    sessions = [
        session([0.50, 0.04], ["on_topic", "off_topic"], session_id="s1", topic="t1",
                author="agent:opus-5"),
        session([0.45, 0.03], ["on_topic", "off_topic"], session_id="s2", topic="t2",
                author="agent:sonnet-5"),
        session([0.55, 0.06], ["on_topic", "off_topic"], session_id="s3", topic="t3",
                author="agent:haiku-4.5"),
        session([0.20, 0.01], ["on_topic", "off_topic"], session_id="s4", topic="t4",
                author="real:annomi"),
    ]
    for family, expected_folds in (("session", 4), ("topic", 4), ("author", 4)):
        result = grid_search(sessions, folds=family)
        assert result.folds == family
        assert len(result.per_fold) == expected_folds
        for fold in result.per_fold:
            assert fold["fold"]
            assert fold["held_out_sessions"]
            assert fold["argmax"] in rc.GRID
        assert result.overall_argmax in rc.GRID
        assert result.argmax_stability["n_folds"] == expected_folds
        assert isinstance(result.argmax_stability["identical"], bool)

    with pytest.raises(ValueError):
        grid_search(sessions, folds="window")


def test_grid_is_two_decimal_from_020_to_080():
    assert rc.GRID[0] == 0.20
    assert rc.GRID[-1] == 0.80
    assert len(rc.GRID) == 61
    assert all(rho == round(rho, 2) for rho in rc.GRID)
    assert all(repr(rho) == repr(round(rho, 2)) for rho in rc.GRID)


def test_evaluate_rejects_a_durations_mode_the_sessions_were_not_scored_with():
    s = session([0.5], ["on_topic"], durations="synthetic")
    assert isinstance(evaluate([s], rho=0.45, durations="synthetic"), ErrorMix)
    with pytest.raises(ValueError):
        evaluate([s], rho=0.45, durations="timed")
