import re
from pathlib import Path

import pytest

from eval_dataset.tools.authoring import CATEGORY_DESCRIPTIONS
from eval_dataset.tools.script_parser import parse_script

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "authoring" / "prompt_template.md"
)

FORBIDDEN = ["cosine", "similarity", "embedding", "threshold", "rho", "RHO",
             "0.36", "0.14", "classifier", "sentence-bert", "sbert",
             "window", "warning", "escrow", "drift score"]


def test_template_leaks_no_mechanism_vocabulary():
    text = TEMPLATE_PATH.read_text().lower()
    for term in FORBIDDEN:
        assert term.lower() not in text, f"blinding leak: {term}"


def test_category_descriptions_leak_nothing():
    for name, desc in CATEGORY_DESCRIPTIONS.items():
        for term in FORBIDDEN:
            assert term.lower() not in desc.lower(), f"{name} leaks {term}"


def test_every_corpus_category_has_a_description():
    assert set(CATEGORY_DESCRIPTIONS) == {
        "gradual_drift", "adversarial", "depressed_baseline",
        "abstract_on_topic", "interleaved_abuse"}


def test_template_has_all_placeholders():
    text = TEMPLATE_PATH.read_text()
    placeholders = [
        "{topic}", "{scope}", "{category_description}",
        "{teacher_word_target}", "{teacher_letter}", "{learner_letter}",
    ]
    for placeholder in placeholders:
        assert text.count(placeholder) == 1, (
            f"{placeholder} appears {text.count(placeholder)} times, expected exactly 1"
        )


def test_depressed_baseline_and_abstract_on_topic_forbid_digression():
    for name in ("depressed_baseline", "abstract_on_topic"):
        desc = CATEGORY_DESCRIPTIONS[name].lower()
        assert "never" in desc or "no " in desc or "without" in desc, (
            f"{name} description must explicitly instruct staying on topic throughout"
        )


def test_template_does_not_instruct_a_placeholder_digression_value():
    text = TEMPLATE_PATH.read_text()
    assert "(none)" not in text
    # No line telling the author to write "DIGRESSION:" followed by a
    # non-numeric placeholder value (none/n/a/empty/etc). Any DIGRESSION:
    # example value shown in the template must be parseable turn numbers.
    for match in re.finditer(r"^DIGRESSION:\s*(\S.*)$", text, re.MULTILINE):
        value = match.group(1).strip()
        if value.startswith("<"):
            # A literal placeholder token like "<turn numbers>" describing
            # the field, not a concrete example value — not what an author
            # would actually write.
            continue
        for part in value.split(","):
            part = part.strip()
            for piece in part.split("-"):
                assert piece.strip().isdigit(), (
                    f"template shows an unparseable DIGRESSION value: {value!r}"
                )


def _write_script(tmp_path, *, category, digression_line, turns):
    lines = [
        "TOPIC: Test topic",
        "TEACHER: A",
        "LEARNER: B",
        f"CATEGORY: {category}",
    ]
    if digression_line is not None:
        lines.append(digression_line)
    lines.append("")
    lines.extend(turns)
    path = tmp_path / "script.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_omitted_digression_line_parses_cleanly_for_no_drift_category(tmp_path):
    path = _write_script(
        tmp_path,
        category="depressed_baseline",
        digression_line=None,
        turns=[
            "A: Let's talk about the topic.",
            "B: Sounds good.",
            "A: Here is more about the topic.",
        ],
    )
    script = parse_script(str(path), require_digression_labels=True)
    assert script.digression_turns == frozenset()


def test_digression_none_placeholder_is_rejected_by_the_parser(tmp_path):
    path = _write_script(
        tmp_path,
        category="depressed_baseline",
        digression_line="DIGRESSION: (none)",
        turns=[
            "A: Let's talk about the topic.",
            "B: Sounds good.",
            "A: Here is more about the topic.",
        ],
    )
    with pytest.raises(ValueError):
        parse_script(str(path), require_digression_labels=True)
