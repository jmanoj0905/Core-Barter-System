import textwrap
import pytest
from eval_dataset.tools.script_parser import parse_script, Script, Turn

def test_parse_script_header_and_turns(tmp_path):
    script_text = textwrap.dedent("""\
        TOPIC: Python fundamentals (variables, loops, functions)
        TEACHER: A
        LEARNER: B
        CATEGORY: gradual_drift

        A: Alright, let's start with variables.
        B: Oh nice, so it's dynamically typed?
        A: Exactly. Now let's look at loops.
        """)
    script_file = tmp_path / "sess_A01.txt"
    script_file.write_text(script_text)

    result = parse_script(str(script_file))

    assert result == Script(
        topic="Python fundamentals (variables, loops, functions)",
        teacher="A",
        learner="B",
        category="gradual_drift",
        turns=[
            Turn(speaker="A", text="Alright, let's start with variables."),
            Turn(speaker="B", text="Oh nice, so it's dynamically typed?"),
            Turn(speaker="A", text="Exactly. Now let's look at loops."),
        ],
    )

def test_parse_script_rejects_invalid_category(tmp_path):
    script_file = tmp_path / "bad.txt"
    script_file.write_text("TOPIC: X\nTEACHER: A\nLEARNER: B\nCATEGORY: not_a_category\n\nA: hi\n")

    import pytest
    with pytest.raises(ValueError, match="not_a_category"):
        parse_script(str(script_file))


def test_parse_script_rejects_missing_header_body_separator(tmp_path):
    # No blank line between the header and the first turn: every dialogue
    # line gets absorbed as a spurious header key (since e.g. "A".isupper()
    # is true), which used to silently produce turns=[].
    script_file = tmp_path / "no_separator.txt"
    script_file.write_text(
        "TOPIC: X\nTEACHER: A\nLEARNER: B\nCATEGORY: clean\n"
        "A: hi\nB: hello\n"
    )

    import pytest
    with pytest.raises(ValueError, match="missing blank line"):
        parse_script(str(script_file))


# --- Task 2 helpers -------------------------------------------------------
#
# Build a minimal valid script: TOPIC/TEACHER/LEARNER/CATEGORY header, a
# blank line, then dialogue turns. `extra` is spliced into the header
# (before the blank line separator) so tests can add SCOPE/AUTHOR/DIGRESSION
# lines. `speakers` is the per-turn speaker layout (TEACHER is "A",
# LEARNER is "B"); it defaults to 4 turns alternating A/B/A/B, so turn 2 is
# a learner turn (needed by test_digression_naming_a_learner_turn_rejected)
# and turn 9 is genuinely out of range (test_digression_turn_out_of_range_rejected).
#
# test_parses_new_header_fields needs a DIFFERENT layout: its
# "DIGRESSION: 2-3, 5" must legitimately resolve to {2, 3, 5}, which requires
# turns 2, 3 and 5 to all be teacher turns and at least 5 turns to exist. The
# default A/B/A/B layout can't satisfy that (turn 2 is the learner there, and
# there's no turn 5) — this is a real mismatch in the brief's fixture
# requirements across tests, resolved by giving that one test its own layout
# rather than weakening either unconditional validation.

_counter = 0


def _path(tmp_path, category, extra="", speakers=("A", "B", "A", "B")):
    global _counter
    _counter += 1
    header = (
        "TOPIC: Python fundamentals\n"
        "TEACHER: A\n"
        "LEARNER: B\n"
        f"CATEGORY: {category}\n"
    )
    if extra:
        header += extra
    ordinals = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ]
    lines = [f"{sp}: turn {ordinals[i]}." for i, sp in enumerate(speakers)]
    body = "\n" + "\n".join(lines) + "\n"
    script_file = tmp_path / f"script_{category}_{_counter}.txt"
    script_file.write_text(header + body)
    return str(script_file)


def _write(tmp_path, category, extra="", speakers=("A", "B", "A", "B"), **kwargs):
    return parse_script(_path(tmp_path, category, extra, speakers=speakers), **kwargs)


def test_parses_new_header_fields(tmp_path):
    # Teacher speaks turns 1, 2, 3, 5; learner speaks 4, 6 — so DIGRESSION's
    # named turns (2, 3, 5) are all legitimately teacher turns.
    s = _write(tmp_path, category="interleaved_abuse", extra=
        "SCOPE: event loop, callbacks\nAUTHOR: agent:opus-5\nDIGRESSION: 2-3, 5\n",
        speakers=("A", "A", "A", "B", "A", "B"))
    assert s.scope == "event loop, callbacks"
    assert s.author == "agent:opus-5"
    assert s.digression_turns == frozenset({2, 3, 5})

def test_defaults_keep_existing_scripts_parsing():
    s = parse_script("eval_dataset/scripts/person_C/sess_C05.txt")
    assert s.scope == "" and s.author == "" and s.digression_turns == frozenset()

def test_new_categories_accepted(tmp_path):
    for c in ("depressed_baseline", "abstract_on_topic", "interleaved_abuse"):
        assert parse_script(_path(tmp_path, category=c)).category == c

def test_digression_turn_out_of_range_rejected(tmp_path):
    # 4 dialogue turns; naming turn 9 is an authoring error, not an empty set.
    with pytest.raises(ValueError, match="turn 9"):
        parse_script(_path(tmp_path, category="adversarial", extra="DIGRESSION: 9\n"))

def test_digression_naming_a_learner_turn_rejected(tmp_path):
    # Only teacher speech reaches a window, so a learner digression is meaningless.
    with pytest.raises(ValueError, match="learner"):
        parse_script(_path(tmp_path, category="adversarial", extra="DIGRESSION: 2\n"))

def test_digression_required_for_drift_categories(tmp_path):
    # Presence rules only fire when require_digression_labels=True (Ruling
    # R1) — the default keeps existing DIGRESSION-less scripts parsing.
    with pytest.raises(ValueError, match="requires DIGRESSION"):
        parse_script(_path(tmp_path, category="gradual_drift"), require_digression_labels=True)

def test_digression_forbidden_for_on_topic_categories(tmp_path):
    with pytest.raises(ValueError, match="must not declare DIGRESSION"):
        parse_script(
            _path(tmp_path, category="abstract_on_topic", extra="DIGRESSION: 1\n"),
            require_digression_labels=True,
        )
