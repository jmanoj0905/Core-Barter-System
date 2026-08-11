import textwrap
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
