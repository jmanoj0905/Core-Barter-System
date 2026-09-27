import pytest

from eval_dataset.tools.gold_labels import gold_label
from eval_dataset.tools.replay import ReplayWindow, TurnContribution
from eval_dataset.tools.script_parser import Script, Turn


def make_script(digression_turns=frozenset(), category="gradual_drift"):
    return Script(
        topic="Test topic",
        teacher="A",
        learner="B",
        category=category,
        turns=[Turn(speaker="A", text="placeholder")],
        digression_turns=frozenset(digression_turns),
    )


def make_window(contributions):
    return ReplayWindow(
        index=0,
        text="",
        raw_text="",
        teacher_seconds_start=0.0,
        teacher_seconds_end=sum(c.seconds for c in contributions),
        contributions=tuple(contributions),
    )


def test_wholly_on_topic_window():
    # share 0.0 -> "on_topic"
    script = make_script(digression_turns=frozenset())
    window = make_window([TurnContribution(turn_index=1, seconds=10.0)])
    assert gold_label(window, script) == "on_topic"


def test_wholly_off_topic_window():
    # share 1.0 -> "off_topic"
    script = make_script(digression_turns=frozenset({1}))
    window = make_window([TurnContribution(turn_index=1, seconds=10.0)])
    assert gold_label(window, script) == "off_topic"


def test_majority_off_topic_window():
    # share 0.7 -> "off_topic"
    script = make_script(digression_turns=frozenset({1}))
    window = make_window(
        [
            TurnContribution(turn_index=1, seconds=7.0),
            TurnContribution(turn_index=3, seconds=3.0),
        ]
    )
    assert gold_label(window, script) == "off_topic"


def test_majority_on_topic_window():
    # share 0.25 -> "on_topic"
    script = make_script(digression_turns=frozenset({1}))
    window = make_window(
        [
            TurnContribution(turn_index=1, seconds=2.5),
            TurnContribution(turn_index=3, seconds=7.5),
        ]
    )
    assert gold_label(window, script) == "on_topic"


def test_boundary_window_is_excluded():
    # share 0.5 -> None
    script = make_script(digression_turns=frozenset({1}))
    window = make_window(
        [
            TurnContribution(turn_index=1, seconds=5.0),
            TurnContribution(turn_index=3, seconds=5.0),
        ]
    )
    assert gold_label(window, script) is None


def test_band_edges_are_excluded():
    # shares of exactly 0.4 and 0.6 -> None, including when floating-point
    # division on the way there does not land on a clean value.
    script = make_script(digression_turns=frozenset({1}))

    window_low = make_window(
        [
            TurnContribution(turn_index=1, seconds=4.0),
            TurnContribution(turn_index=3, seconds=6.0),
        ]
    )
    assert gold_label(window_low, script) is None

    window_high = make_window(
        [
            TurnContribution(turn_index=1, seconds=6.0),
            TurnContribution(turn_index=3, seconds=4.0),
        ]
    )
    assert gold_label(window_high, script) is None


def test_band_edges_are_excluded_with_repeating_decimal_share():
    # Contributions of 10/3 seconds each: 2 off-topic / 5 total computes to
    # 0.39999999999999997 in float arithmetic (not exactly 0.4), and 3
    # off-topic / 5 total computes to exactly 0.6. Both must still be
    # excluded -- this is what proves the tolerance actually does something.
    script = make_script(digression_turns=frozenset({1, 2}))
    window_low = make_window(
        [
            TurnContribution(turn_index=1, seconds=10 / 3),
            TurnContribution(turn_index=2, seconds=10 / 3),
            TurnContribution(turn_index=3, seconds=10 / 3),
            TurnContribution(turn_index=4, seconds=10 / 3),
            TurnContribution(turn_index=5, seconds=10 / 3),
        ]
    )
    off_share = sum(c.seconds for c in window_low.contributions[:2]) / sum(
        c.seconds for c in window_low.contributions
    )
    assert off_share != 0.4  # confirms this genuinely exercises float noise
    assert gold_label(window_low, script) is None

    script_high = make_script(digression_turns=frozenset({1, 2, 3}))
    window_high = make_window(
        [
            TurnContribution(turn_index=1, seconds=10 / 3),
            TurnContribution(turn_index=2, seconds=10 / 3),
            TurnContribution(turn_index=3, seconds=10 / 3),
            TurnContribution(turn_index=4, seconds=10 / 3),
            TurnContribution(turn_index=5, seconds=10 / 3),
        ]
    )
    assert gold_label(window_high, script_high) is None


def test_category_with_no_digressions_is_all_on_topic():
    # abstract_on_topic: every window labels "on_topic", none excluded.
    script = make_script(digression_turns=frozenset(), category="abstract_on_topic")
    windows = [
        make_window([TurnContribution(turn_index=1, seconds=10.0)]),
        make_window(
            [
                TurnContribution(turn_index=1, seconds=5.0),
                TurnContribution(turn_index=3, seconds=5.0),
            ]
        ),
        make_window([TurnContribution(turn_index=5, seconds=25.0)]),
    ]
    labels = [gold_label(w, script) for w in windows]
    assert labels == ["on_topic", "on_topic", "on_topic"]
    assert None not in labels


def test_zero_total_seconds_returns_none_not_zerodivisionerror():
    script = make_script(digression_turns=frozenset({1}))
    window = make_window([])
    assert gold_label(window, script) is None
