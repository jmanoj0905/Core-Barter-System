import pytest

from eval_dataset.tools.replay import (
    ReplayWindow,
    TurnContribution,
    replay,
    synthetic_durations,
)
from eval_dataset.tools.script_parser import Script, Turn


def make_script(turns, teacher="A", learner="B", category="clean"):
    return Script(
        topic="Test topic",
        teacher=teacher,
        learner=learner,
        category=category,
        turns=turns,
    )


def test_only_teacher_turns_enter_windows():
    # 26 teacher-seconds closes a window; learner text must appear in no window.
    script = make_script(
        [
            Turn(speaker="A", text="word " * 65),  # 65 words * 0.4 = 26.0s
            Turn(speaker="B", text="learnerword learnerword learnerword"),
        ]
    )
    durations = synthetic_durations(script)
    windows = replay(script, durations)

    assert len(windows) == 1
    assert "learnerword" not in windows[0].raw_text
    assert "learnerword" not in windows[0].text


def test_window_closes_at_25_teacher_seconds():
    # 26s of teacher speech in one turn -> exactly one closed window.
    script = make_script([Turn(speaker="A", text="word " * 65)])  # 26.0s
    durations = synthetic_durations(script)
    windows = replay(script, durations)

    assert len(windows) == 1
    assert windows[0].teacher_seconds_start == 0.0
    assert windows[0].teacher_seconds_end == 26.0


def test_learner_turns_do_not_advance_the_buffer():
    # teacher 20s, learner 60s, teacher 10s -> one window spanning both teacher turns.
    script = make_script(
        [
            Turn(speaker="A", text="word " * 50),  # 20.0s
            Turn(speaker="B", text="word " * 150),  # 60.0s, ignored
            Turn(speaker="A", text="word " * 25),  # 10.0s -> total 30s, closes
        ]
    )
    durations = synthetic_durations(script)
    windows = replay(script, durations)

    assert len(windows) == 1
    assert windows[0].teacher_seconds_end - windows[0].teacher_seconds_start == pytest.approx(30.0)
    assert [c.turn_index for c in windows[0].contributions] == [1, 3]


def test_trailing_partial_window_is_flushed():
    # Two turns, 25s then 5s at a turn boundary -> window 1 at 25s, window 2 at 5s remainder.
    script = make_script(
        [
            Turn(speaker="A", text="word " * 63),  # 25.2s -> closes window 1
            Turn(speaker="A", text="word " * 13),  # 5.2s -> trailing flush
        ]
    )
    durations = synthetic_durations(script)
    windows = replay(script, durations)

    assert len(windows) == 2
    assert len(windows[0].contributions) == 1
    assert windows[0].contributions[0].turn_index == 1
    assert windows[0].contributions[0].seconds == pytest.approx(25.2)
    assert len(windows[1].contributions) == 1
    assert windows[1].contributions[0].turn_index == 2
    assert windows[1].contributions[0].seconds == pytest.approx(5.2)


def test_skipped_window_still_consumes_an_index():
    # A window under MIN_CONTENT_TOKENS is not emitted, but the next emitted
    # window's .index is 2, not 1 — main.py increments before the check.
    script = make_script(
        [
            # Enough seconds to close (>=25s) but under MIN_CONTENT_TOKENS (3)
            # after filler cleaning: a single filler word repeated, padded
            # with long silent-sounding tokens to hit duration without adding
            # real content tokens. Use "um" repeated many times — clean_text
            # strips it entirely, leaving zero content tokens.
            Turn(speaker="A", text="um " * 65),  # 26.0s, cleans to ""
            Turn(speaker="A", text="This is real teacher content for the window here today now."),
        ]
    )
    durations = synthetic_durations(script)
    # second turn: not long enough alone to close a window, so flush at end.
    windows = replay(script, durations)

    assert len(windows) == 1
    assert windows[0].index == 2


def test_script_with_no_teacher_turns_returns_no_windows():
    script = make_script(
        [
            Turn(speaker="B", text="Only learner speech here."),
            Turn(speaker="B", text="More learner speech."),
        ]
    )
    durations = synthetic_durations(script)
    assert replay(script, durations) == []


def test_synthetic_durations_match_the_handoff_rate():
    script = make_script([Turn(speaker="A", text="one two three four five")])
    assert synthetic_durations(script) == [2.0]


def test_contributions_record_seconds_per_turn():
    # a window fed by turns 1 (15s) and 3 (10s) carries exactly those two.
    script = make_script(
        [
            Turn(speaker="A", text="word " * 37 + "wordextra"),  # ~15.2s
            Turn(speaker="B", text="learner filler text here today."),
            Turn(speaker="A", text="word " * 25),  # 10.0s -> total closes (>=25s)
        ]
    )
    durations = synthetic_durations(script)
    windows = replay(script, durations)

    assert len(windows) == 1
    turn_indices = [c.turn_index for c in windows[0].contributions]
    assert turn_indices == [1, 3]
    assert windows[0].contributions[0].seconds == pytest.approx(durations[0])
    assert windows[0].contributions[1].seconds == pytest.approx(durations[2])


def test_replay_does_not_mutate_script():
    script = make_script([Turn(speaker="A", text="word " * 65)])
    turns_before = list(script.turns)
    durations = synthetic_durations(script)
    replay(script, durations)

    assert script.turns == turns_before


def test_window_span_equals_sum_of_contribution_seconds():
    script = make_script(
        [
            Turn(speaker="A", text="word " * 50),  # 20.0s
            Turn(speaker="A", text="word " * 20),  # 8.0s -> closes at 28.0s
            Turn(speaker="A", text="word " * 10),  # 4.0s -> trailing flush
        ]
    )
    durations = synthetic_durations(script)
    windows = replay(script, durations)

    for w in windows:
        span = w.teacher_seconds_end - w.teacher_seconds_start
        total = sum(c.seconds for c in w.contributions)
        assert span == pytest.approx(total)


def test_overshoot_is_discarded_not_carried_forward():
    # Three teacher turns at 30s / 20s / 10s, explicit durations. Discard:
    # turn 1 (30s) closes its own window; the 5s overshoot is thrown away, so
    # turns 2+3 (20s + 10s = 30s) close a second window together.
    # Carry-forward would instead credit the 5s overshoot to the next buffer,
    # letting turn 2 alone close a window (5+20=25), leaving turn 3 (10s) as
    # its own trailing-flush window -> 3 windows instead of 2.
    script = make_script(
        [
            Turn(speaker="A", text="alpha beta gamma delta"),
            Turn(speaker="A", text="epsilon zeta eta theta"),
            Turn(speaker="A", text="iota kappa lambda mu"),
        ]
    )
    durations = [30.0, 20.0, 10.0]
    windows = replay(script, durations)

    assert len(windows) == 2
    assert [c.turn_index for c in windows[0].contributions] == [1]
    assert [c.turn_index for c in windows[1].contributions] == [2, 3]


def test_filler_phrase_spanning_a_turn_boundary_is_cleaned():
    # Two teacher turns land in the same window; the multi-word filler phrase
    # "you know" straddles the turn boundary (turn 1 ends in "you", turn 2
    # begins with "know"). Only cleaning the JOIN strips it — cleaning each
    # segment first would leave both words in place, since neither "...you"
    # nor "know..." matches the phrase alone.
    script = make_script(
        [
            Turn(speaker="A", text="alpha beta gamma you"),
            Turn(speaker="A", text="know delta epsilon zeta"),
        ]
    )
    durations = [13.0, 13.0]
    windows = replay(script, durations)

    assert len(windows) == 1
    assert windows[0].raw_text == "alpha beta gamma you know delta epsilon zeta"
    assert windows[0].text == "alpha beta gamma delta epsilon zeta"
