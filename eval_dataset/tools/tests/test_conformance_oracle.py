"""Conformance gate — spec §4.3.

Pins the committed offline harness (`replay` + `synthetic_durations` +
`score_windows`) to per-window cosines recorded from two real instrumented
production runs. This is a gate, not a component: it produces no new module
and asserts against numbers transcribed in
`.superpowers/sdd/2026-09-27-analogy-corpus/conformance-oracle-reference.md`
(the authoritative oracle; cross-checked against
`apps/semantic_analysis/ground_truth/analogy_handling_findings.md`).

No mechanism code lives here: no RHO, no thr = RHO*R, no running-max ladder.
The harness runs no `warning_engine`, so warning-ladder behavior is asserted
as local arithmetic over the classification sequence (count of `incorrect`
windows, max consecutive `incorrect` run) rather than as emitted warnings.
"""

import pytest

pytest.importorskip(
    "sentence_transformers",
    reason="requires apps/semantic_analysis/venv/bin/python, not system python3",
)

from eval_dataset.tools.embed_windows import score_windows
from eval_dataset.tools.replay import replay, synthetic_durations
from eval_dataset.tools.script_parser import parse_script

# Observed max deviation across all 19 windows in both recorded production
# runs was 0.0005 (see task-5 brief, Ruling R12). The recorded oracle values
# are reported to 3 decimal places, which accounts for the residual gap up
# to 0.001. 0.001 is therefore used (tightened from the brief's draft 0.005)
# and still leaves 2x headroom over the observed worst case.
TOLERANCE = 0.001

SESS_B08 = "eval_dataset/scripts/person_B/sess_B08.txt"
SESS_C05 = "eval_dataset/scripts/person_C/sess_C05.txt"

# R4: analogy span = known teacher-turn ranges from the recorded findings,
# not a DIGRESSION header — neither script carries one, and neither should:
# both are committed code_switch scripts measured exactly as authored.
B08_ANALOGY_TURNS = frozenset(range(6, 11))  # teacher turns 6-10
C05_ANALOGY_TURNS = frozenset(range(6, 8))  # teacher turns 6-7

# Expected turn composition per window, from the oracle reference's tables.
C05_EXPECTED_TURNS = [
    (1, 3, 5),
    (6,),
    (7,),
    (9, 11),
    (13, 15),
    (17, 18),
    (20,),
]

B08_EXPECTED_TURNS = [
    (1, 3),
    (5,),
    (6,),
    (7,),
    (8,),
    (9,),
    (10,),
    (12, 14),
    (16, 18),
    (20, 22),
    (24, 26),
    (28,),
]


def _score_session(path, *, require_digression_labels=False, model=None):
    """Parse -> replay -> score a corpus script exactly as production would.

    Returns (windows, scores) so both turn composition (on ReplayWindow) and
    cosine/classification (on WindowScore) can be asserted from one call.
    """
    script = parse_script(path, require_digression_labels=require_digression_labels)
    durations = synthetic_durations(script)
    windows = replay(script, durations)
    scores = score_windows(windows, script.topic, script.scope, model=model)
    return windows, scores


def _analogy_similarities(windows, scores, analogy_turns):
    """Cosines of windows whose contributions fall entirely within the analogy span."""
    result = []
    for window, score in zip(windows, scores):
        turn_indices = {c.turn_index for c in window.contributions}
        if turn_indices and turn_indices <= analogy_turns:
            result.append(score.similarity)
    return result


def _max_consecutive_incorrect(scores) -> int:
    best = current = 0
    for score in scores:
        if score.classification == "incorrect":
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


@pytest.fixture(scope="module")
def sessions():
    """Load the model once; score both primary sessions once."""
    from eval_dataset.tools.embed_windows import _get_cached_model

    model = _get_cached_model()
    b08_windows, b08_scores = _score_session(SESS_B08, model=model)
    c05_windows, c05_scores = _score_session(SESS_C05, model=model)
    return {
        "b08": (b08_windows, b08_scores),
        "c05": (c05_windows, c05_scores),
    }


# ---------------------------------------------------------------------------
# sess_B08 — 28 dialogue turns (17 teacher), 12 windows, analogy = turns 6-10
# ---------------------------------------------------------------------------


def test_sess_B08_window_count_and_turn_composition(sessions):
    windows, _ = sessions["b08"]
    assert len(windows) == 12
    actual = [tuple(c.turn_index for c in w.contributions) for w in windows]
    assert actual == B08_EXPECTED_TURNS


def test_sess_B08_analogy_window_cosines(sessions):
    windows, scores = sessions["b08"]
    expected = [0.179, 0.130, 0.197, 0.008, 0.029]
    actual = _analogy_similarities(windows, scores, B08_ANALOGY_TURNS)
    assert actual == pytest.approx(expected, abs=TOLERANCE)


def test_sess_B08_incorrect_count_and_max_consecutive(sessions):
    _, scores = sessions["b08"]
    assert sum(s.classification == "incorrect" for s in scores) == 3
    # Max consecutive incorrect run = 2 (windows 6, 7; window 5 at 0.197
    # breaks the run between window 4 and window 6). By the live rule at
    # apps/warning_engine/main.py:216-221 (count >= 3 -> severe, == 2 ->
    # strong, == 1 -> silent), a max run of 2 yields STRONG — matching the
    # recorded run. The harness runs no warning_engine, so this is local
    # arithmetic over the classification sequence, not an emitted warning.
    assert _max_consecutive_incorrect(scores) == 2


# ---------------------------------------------------------------------------
# sess_C05 — 20 dialogue turns, 7 windows, analogy = turns 6-7
# ---------------------------------------------------------------------------


def test_sess_C05_window_count_and_turn_composition(sessions):
    windows, _ = sessions["c05"]
    assert len(windows) == 7
    actual = [tuple(c.turn_index for c in w.contributions) for w in windows]
    assert actual == C05_EXPECTED_TURNS


def test_sess_C05_analogy_window_cosines(sessions):
    windows, scores = sessions["c05"]
    expected = [0.172, 0.155]
    actual = _analogy_similarities(windows, scores, C05_ANALOGY_TURNS)
    assert actual == pytest.approx(expected, abs=TOLERANCE)


def test_sess_C05_has_no_incorrect_windows(sessions):
    _, scores = sessions["c05"]
    assert sum(s.classification == "incorrect" for s in scores) == 0
    assert _max_consecutive_incorrect(scores) == 0


# ---------------------------------------------------------------------------
# _experimental secondary checks — weaker, wider-tolerance checks per the
# plan. Neither script declares a DIGRESSION header, so the span is taken as
# the contiguous run of lowest-cosine windows of the stated length (per the
# oracle reference), and only the run's mean is asserted, at +/-0.02.
# ---------------------------------------------------------------------------


def _digression_mean(script_relpath, span_length, *, require_digression_labels=False):
    path = f"eval_dataset/scripts/{script_relpath}"
    windows, scores = _score_session(
        path, require_digression_labels=require_digression_labels
    )
    similarities = [s.similarity for s in scores]
    best_start, best_mean = 0, float("inf")
    for start in range(0, len(similarities) - span_length + 1):
        window_slice = similarities[start : start + span_length]
        mean = sum(window_slice) / span_length
        if mean < best_mean:
            best_start, best_mean = start, mean
    return best_mean


@pytest.mark.parametrize(
    "script,span_length,expected_mean",
    [
        ("_experimental/case_B2_late.txt", 11, 0.059),
        ("_experimental/adv3_long.txt", 10, 0.008),
    ],
)
def test_experimental_span_means(script, span_length, expected_mean):
    # adv3_long.txt is category `adversarial` with no DIGRESSION header, so
    # it must parse at the default require_digression_labels=False.
    actual_mean = _digression_mean(script, span_length)
    assert actual_mean == pytest.approx(expected_mean, abs=0.02)
