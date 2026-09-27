"""Tests for provenance verification (Task 10).

These tests never touch the real (gitignored) AnnoMI corpus. Every test
builds a synthetic script + provenance file under tmp_path and injects a
source_loader that serves genuine "source" text out of an in-memory dict,
so `verify()` is fully exercised without the fetched raw/ data.

The tests are deliberately adversarial about the module's central claim:
every failure path in verify() must raise ProvenanceError, never return
cleanly (Review Focus #5). Several tests assert not just that an error was
raised, but that it was raised *for the stated reason* — a check that
happens to fail for an unrelated reason is not the check we want.
"""

import json

import pytest

from eval_dataset.tools.verify_provenance import ProvenanceError, verify

# A snippet of genuine-looking source dialogue, used as the AnnoMI "record"
# text in most tests. Deliberately has irregular spacing to exercise the
# whitespace-normalisation policy (collapse runs of whitespace only).
SOURCE_TEXT = (
    "Okay, today I want to get you past the syntax and into the thing "
    "that actually  makes Node different.\n"
    "I keep hearing non-blocking but honestly I don't know what it's not "
    "blocking.\n"
    "The short version is that Node runs your JavaScript on a single "
    "thread."
)

SOURCE_LINE_1 = SOURCE_TEXT.split("\n")[0]  # has the double space, verbatim


SECOND_SOURCE_TEXT = (
    "Let's switch gears and talk about closures for a minute.\n"
    "A closure is a function that remembers the variables from where it "
    "was defined."
)


def _write_script(tmp_path, turns, category="clean"):
    """Write a minimal valid script file and return its path."""
    lines = [
        "TOPIC: Node.js basics (event loop, non-blocking I/O, callbacks)",
        "TEACHER: C",
        "LEARNER: D",
        f"CATEGORY: {category}",
        "",
    ]
    lines.extend(f"{speaker}: {text}" for speaker, text in turns)
    script_path = tmp_path / "sess_TEST01.txt"
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(script_path)


def _write_provenance(tmp_path, *, dataset, records, spans, seams, name="prov.json"):
    provenance_path = tmp_path / name
    provenance_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "records": list(records),
                "spans": [list(s) for s in spans],
                "seams": list(seams),
            }
        ),
        encoding="utf-8",
    )
    return str(provenance_path)


def _find(source_text, needle):
    """Locate the exact char span of `needle` inside `source_text`."""
    start = source_text.index(needle)
    return start, start + len(needle)


def _make_loader(records: dict):
    """An injectable source_loader over an in-memory {record_id: text} map.

    Mirrors the real loader's contract: unknown record ids return None
    (missing), a present-but-empty string means empty.
    """

    def loader(record_id):
        return records.get(record_id)

    return loader


def test_verbatim_script_passes(tmp_path):
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually  makes Node different."
    line2 = "I keep hearing non-blocking but honestly I don't know what it's not blocking."
    script_path = _write_script(
        tmp_path,
        [("C", "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."),
         ("D", "I keep hearing non-blocking but honestly I don't know what it's not blocking.")],
    )
    span1 = _find(SOURCE_TEXT, line1)
    span2 = _find(SOURCE_TEXT, line2)
    provenance_path = _write_provenance(
        tmp_path,
        dataset="annomi",
        records=["rec_1"],
        spans=[("rec_1", *span1), ("rec_1", *span2)],
        seams=[],
    )
    loader = _make_loader({"rec_1": SOURCE_TEXT})

    assert verify(script_path, provenance_path, source_loader=loader) is None


def test_a_single_altered_word_fails(tmp_path):
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually  makes Node different."
    line2 = "I keep hearing non-blocking but honestly I don't know what it's not blocking."
    # Change exactly one word in the middle of the second line: "honestly"
    # -> "actually" — same length (8 chars), so a length-only comparison
    # would NOT catch this. Only genuine content comparison does.
    script_path = _write_script(
        tmp_path,
        [("C", "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."),
         ("D", "I keep hearing non-blocking but actually I don't know what it's not blocking.")],
    )
    span1 = _find(SOURCE_TEXT, line1)
    span2 = _find(SOURCE_TEXT, line2)
    provenance_path = _write_provenance(
        tmp_path,
        dataset="annomi",
        records=["rec_1"],
        spans=[("rec_1", *span1), ("rec_1", *span2)],
        seams=[],
    )
    loader = _make_loader({"rec_1": SOURCE_TEXT})

    with pytest.raises(ProvenanceError, match="not a verbatim excerpt"):
        verify(script_path, provenance_path, source_loader=loader)


def test_missing_source_file_raises_not_passes(tmp_path):
    """Review Focus #5: citing a record the loader has never heard of must
    raise, never silently pass. A substring test against absent text
    ("x" in None, or treating None as "") is trivially satisfied/skippable,
    which would admit an unverified script labelled as verified."""
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."
    script_path = _write_script(tmp_path, [("C", line1)])
    provenance_path = _write_provenance(
        tmp_path,
        dataset="annomi",
        records=["rec_missing"],
        spans=[("rec_missing", 0, len(line1))],
        seams=[],
    )
    # Loader that has no knowledge of "rec_missing" at all -> returns None.
    loader = _make_loader({"rec_1": SOURCE_TEXT})

    with pytest.raises(ProvenanceError, match="rec_missing.*missing"):
        verify(script_path, provenance_path, source_loader=loader)


def test_empty_source_text_raises(tmp_path):
    """The record exists but its text is empty. Must raise, and the error
    must name the empty source specifically -- distinguishing "raised for
    the right reason" from a naive `"anything" in ""` check that would
    also raise here, but only by accident (e.g. it would also incorrectly
    pass a missing record if that path happened to yield "" too)."""
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."
    script_path = _write_script(tmp_path, [("C", line1)])
    provenance_path = _write_provenance(
        tmp_path,
        dataset="annomi",
        records=["rec_empty"],
        spans=[("rec_empty", 0, len(line1))],
        seams=[],
    )
    loader = _make_loader({"rec_empty": ""})

    with pytest.raises(ProvenanceError, match="rec_empty.*empty"):
        verify(script_path, provenance_path, source_loader=loader)


def test_seams_must_fall_on_turn_boundaries(tmp_path):
    """Both turns are cited from the same record (no real splice anywhere),
    but the provenance declares a seam at turn index 1 anyway. A seam must
    mark a genuine transition between two different cited records; one
    declared where no such transition exists does not fall on a valid
    splice boundary and must be rejected."""
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."
    line2 = "I keep hearing non-blocking but honestly I don't know what it's not blocking."
    script_path = _write_script(tmp_path, [("C", line1), ("D", line2)])
    span1 = _find(SOURCE_TEXT, SOURCE_LINE_1)
    span2 = _find(SOURCE_TEXT, line2)
    provenance_path = _write_provenance(
        tmp_path,
        dataset="annomi",
        records=["rec_1"],
        spans=[("rec_1", *span1), ("rec_1", *span2)],
        seams=[1],  # bogus: both spans cite the same record
    )
    loader = _make_loader({"rec_1": SOURCE_TEXT})

    with pytest.raises(ProvenanceError, match="seam"):
        verify(script_path, provenance_path, source_loader=loader)


def test_seam_out_of_range_also_raises(tmp_path):
    """A seam index that isn't even a valid interior turn boundary (e.g.
    equal to the turn count) must raise too."""
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."
    line2 = "I keep hearing non-blocking but honestly I don't know what it's not blocking."
    script_path = _write_script(tmp_path, [("C", line1), ("D", line2)])
    span1 = _find(SOURCE_TEXT, SOURCE_LINE_1)
    span2 = _find(SECOND_SOURCE_TEXT, "Let's switch gears and talk about closures for a minute.")
    provenance_path = _write_provenance(
        tmp_path,
        dataset="annomi",
        records=["rec_1", "rec_2"],
        spans=[("rec_1", *span1), ("rec_2", *span2)],
        seams=[2],  # out of range: only one interior boundary exists (index 1)
    )
    loader = _make_loader({"rec_1": SOURCE_TEXT, "rec_2": SECOND_SOURCE_TEXT})

    with pytest.raises(ProvenanceError, match="out of range"):
        verify(script_path, provenance_path, source_loader=loader)


def test_every_dialogue_line_must_be_covered_by_a_span(tmp_path):
    """A script with an extra, uncited line (the cleaner wrote one original
    sentence and cited nothing for it) must be rejected -- catches an
    author smuggling free prose into a corpus labelled real:annomi."""
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."
    uncited_line = "And honestly, I think you'll really enjoy this next part."
    script_path = _write_script(tmp_path, [("C", line1), ("D", uncited_line)])
    span1 = _find(SOURCE_TEXT, SOURCE_LINE_1)
    # Only one span is cited even though the script has two turns.
    provenance_path = _write_provenance(
        tmp_path,
        dataset="annomi",
        records=["rec_1"],
        spans=[("rec_1", *span1)],
        seams=[],
    )
    loader = _make_loader({"rec_1": SOURCE_TEXT})

    with pytest.raises(ProvenanceError, match="uncited"):
        verify(script_path, provenance_path, source_loader=loader)


def test_provenance_file_missing_raises(tmp_path):
    """The provenance file itself being absent must also raise, not pass."""
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."
    script_path = _write_script(tmp_path, [("C", line1)])
    missing_provenance_path = str(tmp_path / "does_not_exist.json")

    with pytest.raises(ProvenanceError, match="not found"):
        verify(script_path, missing_provenance_path)


def test_malformed_script_raises_provenance_error_not_value_error(tmp_path):
    """A script that fails to parse (bad CATEGORY, missing header field, no
    dialogue turns, etc.) must surface as ProvenanceError, not let the
    underlying ValueError from parse_script escape uncaught. Task 11 drives
    verify() over 10 scripts catching ProvenanceError to decide which to
    re-extract; an escaping ValueError would crash that loop instead of
    giving a clean per-script diagnostic."""
    bad_script_path = tmp_path / "sess_bad.txt"
    bad_script_path.write_text(
        "TOPIC: X\nTEACHER: C\nLEARNER: D\nCATEGORY: not_a_real_category\n\n"
        "C: hi\n",
        encoding="utf-8",
    )
    provenance_path = _write_provenance(
        tmp_path, dataset="annomi", records=["rec_1"], spans=[("rec_1", 0, 2)], seams=[]
    )
    loader = _make_loader({"rec_1": "hi"})

    with pytest.raises(ProvenanceError, match="could not be parsed"):
        verify(str(bad_script_path), provenance_path, source_loader=loader)


def test_malformed_span_arity_raises_provenance_error_not_type_error(tmp_path):
    """A span tuple with the wrong arity (e.g. missing char_end) must raise
    ProvenanceError, not a bare TypeError/ValueError from unpacking it deep
    inside verify()."""
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."
    script_path = _write_script(tmp_path, [("C", line1)])
    provenance_path = tmp_path / "prov.json"
    provenance_path.write_text(
        json.dumps(
            {
                "dataset": "annomi",
                "records": ["rec_1"],
                "spans": [["rec_1", 0]],  # missing char_end
                "seams": [],
            }
        ),
        encoding="utf-8",
    )
    loader = _make_loader({"rec_1": SOURCE_TEXT})

    with pytest.raises(ProvenanceError, match="malformed"):
        verify(script_path, str(provenance_path), source_loader=loader)


def test_case_only_divergence_fails(tmp_path):
    """A line differing from its source ONLY by capitalisation must raise.
    Pins normalisation as collapse-whitespace-only: a future maintainer
    adding .lower() would let a reworded line pass, defeating the module's
    purpose, and this test would catch that."""
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."
    script_path = _write_script(tmp_path, [("C", line1.upper())])
    span1 = _find(SOURCE_TEXT, SOURCE_LINE_1)
    provenance_path = _write_provenance(
        tmp_path,
        dataset="annomi",
        records=["rec_1"],
        spans=[("rec_1", *span1)],
        seams=[],
    )
    loader = _make_loader({"rec_1": SOURCE_TEXT})

    with pytest.raises(ProvenanceError, match="not a verbatim excerpt"):
        verify(script_path, provenance_path, source_loader=loader)


def test_punctuation_only_divergence_fails(tmp_path):
    """A line differing from its source ONLY by punctuation must raise.
    Pins normalisation as collapse-whitespace-only: a future maintainer
    stripping punctuation "for robustness" would let a reworded line pass,
    and this test would catch that."""
    line1 = "Okay, today I want to get you past the syntax and into the thing that actually makes Node different."
    stripped = line1.replace(",", "").replace(".", "")
    script_path = _write_script(tmp_path, [("C", stripped)])
    span1 = _find(SOURCE_TEXT, SOURCE_LINE_1)
    provenance_path = _write_provenance(
        tmp_path,
        dataset="annomi",
        records=["rec_1"],
        spans=[("rec_1", *span1)],
        seams=[],
    )
    loader = _make_loader({"rec_1": SOURCE_TEXT})

    with pytest.raises(ProvenanceError, match="not a verbatim excerpt"):
        verify(script_path, provenance_path, source_loader=loader)
