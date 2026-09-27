"""Corpus-integrity checks for the calibration corpus (Task 8, extended by
Task 11 to cover the full 25-session corpus).

Guards the calibration corpus's own stricter contract: every script must
parse with declared gold DIGRESSION labels, there must be exactly 5 scripts
per category with topics drawn from the shared topic pool (for the blinded
categories) or from AnnoMI's own third-party labels (for the real-data
categories), teacher word counts and window counts must fall in the
expected ranges, and authorship must be rotated across the three agent
models plus the real:annomi provenance author.

No mechanism code here (no cosine/embedding/RHO) — this only exercises the
script parser, the offline replay windower, and the provenance verifier,
all of which are dependency-light and runnable under plain system python3.
"""

import json
import re
from pathlib import Path

import pytest

from eval_dataset.tools.replay import replay, synthetic_durations
from eval_dataset.tools.script_parser import parse_script
from eval_dataset.tools.verify_provenance import verify

REPO_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_ROOT = REPO_ROOT / "eval_dataset" / "scripts" / "calibration"
PROVENANCE_ROOT = REPO_ROOT / "eval_dataset" / "provenance"
TOPIC_POOL_PATH = REPO_ROOT / "eval_dataset" / "topics" / "topic_pool.md"
ANNOMI_RAW_DIR = REPO_ROOT / "eval_dataset" / "curated" / "annomi" / "raw"

# The three original blinded drift categories. Deliberately kept narrow —
# widening this to all five categories would break the drift-specific
# assertions below (e.g. test_drift_scripts_declare_digressions), which
# only hold for these three. Do not add the two real-data categories here.
CATEGORIES = ("gradual_drift", "adversarial", "interleaved_abuse")

# The two categories sourced from real AnnoMI dialogue (Task 11).
REAL_DATA_CATEGORIES = ("abstract_on_topic", "depressed_baseline")

# All five categories, corpus-wide.
ALL_CATEGORIES = CATEGORIES + REAL_DATA_CATEGORIES

AUTHORS = {"agent:opus-5", "agent:sonnet-5", "agent:haiku-4.5"}
REAL_DATA_AUTHOR = "real:annomi"
ALL_AUTHORS = AUTHORS | {REAL_DATA_AUTHOR}

EXPECTED_TOTAL = 15
EXPECTED_PER_CATEGORY = 5

EXPECTED_CORPUS_TOTAL = 25
EXPECTED_REAL_DATA_TOTAL = 10


def _category_scripts(category: str) -> list[Path]:
    return sorted((CALIBRATION_ROOT / category).glob("sess_CAL*.txt"))


def _all_scripts() -> list[Path]:
    paths: list[Path] = []
    for category in CATEGORIES:
        paths.extend(_category_scripts(category))
    return paths


def _real_data_scripts() -> list[Path]:
    paths: list[Path] = []
    for category in REAL_DATA_CATEGORIES:
        paths.extend(_category_scripts(category))
    return paths


def _corpus_wide_scripts() -> list[Path]:
    paths: list[Path] = []
    for category in ALL_CATEGORIES:
        paths.extend(_category_scripts(category))
    return paths


def _parse_topic_pool() -> set[str]:
    """Parse eval_dataset/topics/topic_pool.md into the set of 40 topics.

    The pool is Markdown with `## Person A`-`## Person D` sections, each
    holding a numbered list of lines shaped `N. <teaches> ↔ <wants to
    learn>`. A valid topic is the left side of the arrow, stripped.
    """
    text = TOPIC_POOL_PATH.read_text(encoding="utf-8")
    topics: set[str] = set()
    line_re = re.compile(r"^\d+\.\s*(.+?)\s*↔\s*(.+?)\s*$")
    for line in text.splitlines():
        match = line_re.match(line.strip())
        if match:
            topics.add(match.group(1).strip())
    return topics


def test_every_calibration_script_parses():
    paths = _all_scripts()
    assert len(paths) == EXPECTED_TOTAL, (
        f"expected {EXPECTED_TOTAL} calibration scripts to exist, found "
        f"{len(paths)}: {[str(p) for p in paths]}"
    )
    for path in paths:
        # require_digression_labels=True: the calibration corpus mandates
        # declared gold labels, unlike older committed scripts elsewhere.
        parse_script(str(path), require_digression_labels=True)


def test_five_sessions_per_category():
    for category in CATEGORIES:
        paths = _category_scripts(category)
        assert len(paths) == EXPECTED_PER_CATEGORY, (
            f"expected {EXPECTED_PER_CATEGORY} scripts in category "
            f"{category!r}, found {len(paths)}: {[str(p) for p in paths]}"
        )
        for path in paths:
            script = parse_script(str(path), require_digression_labels=True)
            assert script.category == category, (
                f"{path} sits under {category!r} but declares CATEGORY "
                f"{script.category!r}"
            )


def test_all_topics_distinct_and_from_the_pool():
    paths = _all_scripts()
    assert len(paths) == EXPECTED_TOTAL

    pool = _parse_topic_pool()
    assert len(pool) == 40, f"expected 40 topics in the topic pool, found {len(pool)}"

    topics = []
    for path in paths:
        script = parse_script(str(path), require_digression_labels=True)
        topics.append(script.topic)

    assert len(topics) == len(set(topics)), (
        f"topics are not pairwise distinct across the corpus: {topics}"
    )
    for topic in topics:
        assert topic in pool, f"topic {topic!r} is not present in the topic pool"


def test_teacher_word_count_in_range():
    paths = _all_scripts()
    assert len(paths) == EXPECTED_TOTAL

    for path in paths:
        script = parse_script(str(path), require_digression_labels=True)
        word_count = sum(
            len(turn.text.split()) for turn in script.turns if turn.speaker == script.teacher
        )
        assert 1000 <= word_count <= 1400, (
            f"{path}: teacher word count {word_count} is outside [1000, 1400]"
        )


def test_each_script_yields_at_least_ten_windows():
    paths = _all_scripts()
    assert len(paths) == EXPECTED_TOTAL

    for path in paths:
        script = parse_script(str(path), require_digression_labels=True)
        windows = replay(script, synthetic_durations(script))
        assert len(windows) >= 10, (
            f"{path}: replay produced only {len(windows)} windows, expected >= 10"
        )


def test_authors_are_rotated():
    author_counts_by_category = {category: {} for category in CATEGORIES}
    author_counts_total = {}

    for category in CATEGORIES:
        paths = _category_scripts(category)
        assert len(paths) == EXPECTED_PER_CATEGORY
        for path in paths:
            script = parse_script(str(path), require_digression_labels=True)
            author = script.author
            author_counts_by_category[category][author] = (
                author_counts_by_category[category].get(author, 0) + 1
            )
            author_counts_total[author] = author_counts_total.get(author, 0) + 1

    for category in CATEGORIES:
        for author in AUTHORS:
            count = author_counts_by_category[category].get(author, 0)
            assert count >= 1, (
                f"author {author!r} does not appear in category {category!r}"
            )

    for author in AUTHORS:
        count = author_counts_total.get(author, 0)
        assert count == 5, (
            f"author {author!r} appears {count} times corpus-wide, expected 5"
        )


def test_drift_scripts_declare_digressions():
    paths = _all_scripts()
    assert len(paths) == EXPECTED_TOTAL

    for path in paths:
        script = parse_script(str(path), require_digression_labels=True)
        assert script.digression_turns, (
            f"{path}: category {script.category!r} is a drift category and "
            "must declare a non-empty DIGRESSION set"
        )


# --- Task 11: corpus-wide tests covering the full 25-session corpus ---


def test_real_data_scripts_pass_provenance_verification():
    """verify() must raise for none of the 10 real-data scripts (CAL16-25).

    This is the load-bearing provenance test: it re-derives every dialogue
    line of each real-data script from its cited AnnoMI source span and
    confirms it is genuinely verbatim. It needs the fetched (gitignored)
    AnnoMI corpus under eval_dataset/curated/annomi/raw/ — when that is
    absent we must skip loudly (never silently pass) naming the missing
    directory and how to populate it.
    """
    if not ANNOMI_RAW_DIR.is_dir():
        pytest.skip(
            f"AnnoMI raw corpus not found at {ANNOMI_RAW_DIR} — this test "
            "needs the fetched (gitignored) AnnoMI dataset to verify "
            "provenance. Populate it by running "
            "eval_dataset/curated/annomi/build_dataset.py (see "
            "eval_dataset/curated/annomi/SOURCE_README.md) before running "
            "this test; skipping is NOT a pass."
        )

    paths = _real_data_scripts()
    assert len(paths) == EXPECTED_REAL_DATA_TOTAL, (
        f"expected {EXPECTED_REAL_DATA_TOTAL} real-data scripts, found "
        f"{len(paths)}: {[str(p) for p in paths]}"
    )

    for path in paths:
        provenance_path = PROVENANCE_ROOT / f"{path.stem}.provenance.json"
        assert provenance_path.is_file(), (
            f"{path}: expected a provenance sidecar at {provenance_path}"
        )
        verify(str(path), str(provenance_path))


def test_real_data_scripts_declare_no_digressions():
    paths = _real_data_scripts()
    assert len(paths) == EXPECTED_REAL_DATA_TOTAL

    for path in paths:
        script = parse_script(str(path), require_digression_labels=True)
        assert script.digression_turns == frozenset(), (
            f"{path}: real-data scripts must declare no digressions, found "
            f"{script.digression_turns}"
        )


def test_real_data_topics_are_annomi_declared_and_distinct():
    paths = _real_data_scripts()
    assert len(paths) == EXPECTED_REAL_DATA_TOTAL

    pool = _parse_topic_pool()
    assert len(pool) == 40, f"expected 40 topics in the topic pool, found {len(pool)}"

    topics = []
    for path in paths:
        script = parse_script(str(path), require_digression_labels=True)
        topics.append(script.topic)

    assert len(topics) == len(set(topics)), (
        f"real-data topics are not pairwise distinct: {topics}"
    )

    for topic in topics:
        assert topic not in pool, (
            f"real-data topic {topic!r} unexpectedly appears in the "
            "topic pool — AnnoMI topics are third-party labels and must "
            "NOT be drawn from the shared blinded pool"
        )


def test_corpus_has_twenty_five_distinct_topics():
    paths = _corpus_wide_scripts()
    assert len(paths) == EXPECTED_CORPUS_TOTAL, (
        f"expected {EXPECTED_CORPUS_TOTAL} scripts corpus-wide, found "
        f"{len(paths)}: {[str(p) for p in paths]}"
    )

    topics = []
    for path in paths:
        script = parse_script(str(path), require_digression_labels=True)
        topics.append(script.topic)

    assert len(topics) == EXPECTED_CORPUS_TOTAL
    assert len(set(topics)) == EXPECTED_CORPUS_TOTAL, (
        f"topics are not pairwise distinct across the 25-session corpus: {topics}"
    )


def test_corpus_is_twenty_five_sessions_across_five_categories():
    paths = _corpus_wide_scripts()
    assert len(paths) == EXPECTED_CORPUS_TOTAL

    for category in ALL_CATEGORIES:
        cat_paths = _category_scripts(category)
        assert len(cat_paths) == EXPECTED_PER_CATEGORY, (
            f"expected {EXPECTED_PER_CATEGORY} scripts in category "
            f"{category!r}, found {len(cat_paths)}: "
            f"{[str(p) for p in cat_paths]}"
        )
        for path in cat_paths:
            script = parse_script(str(path), require_digression_labels=True)
            assert script.category == category, (
                f"{path} sits under {category!r} but declares CATEGORY "
                f"{script.category!r}"
            )


def test_author_folds_have_at_least_four_values():
    paths = _corpus_wide_scripts()
    assert len(paths) == EXPECTED_CORPUS_TOTAL

    author_counts = {}
    for path in paths:
        script = parse_script(str(path), require_digression_labels=True)
        author_counts[script.author] = author_counts.get(script.author, 0) + 1

    assert len(author_counts) >= 4, (
        "expected at least 4 distinct AUTHOR values corpus-wide (the three "
        f"agent models plus {REAL_DATA_AUTHOR!r}) to make a leave-one-"
        f"author-out fold possible, found only {sorted(author_counts)}"
    )
    assert ALL_AUTHORS <= set(author_counts), (
        f"expected authors {sorted(ALL_AUTHORS)} to all appear corpus-wide, "
        f"found {sorted(author_counts)}"
    )
    assert author_counts.get(REAL_DATA_AUTHOR) == EXPECTED_REAL_DATA_TOTAL, (
        f"expected {EXPECTED_REAL_DATA_TOTAL} scripts authored "
        f"{REAL_DATA_AUTHOR!r}, found {author_counts.get(REAL_DATA_AUTHOR)}"
    )


def test_corpus_wide_scripts_yield_at_least_ten_windows():
    """Same >= 10 windows floor as the drift-15 test, extended to the 10
    real-data scripts (which range 14-18 windows per the brief)."""
    paths = _real_data_scripts()
    assert len(paths) == EXPECTED_REAL_DATA_TOTAL

    for path in paths:
        script = parse_script(str(path), require_digression_labels=True)
        windows = replay(script, synthetic_durations(script))
        assert len(windows) >= 10, (
            f"{path}: replay produced only {len(windows)} windows, expected >= 10"
        )


def test_depressed_baseline_scripts_are_unspliced():
    """depressed_baseline is defined as a fully clean, unspliced session:
    exactly one cited source record and an empty seams tuple. A splice
    would introduce a topic change, which the category forbids."""
    paths = _category_scripts("depressed_baseline")
    assert len(paths) == EXPECTED_PER_CATEGORY, (
        f"expected {EXPECTED_PER_CATEGORY} depressed_baseline scripts, "
        f"found {len(paths)}: {[str(p) for p in paths]}"
    )

    for path in paths:
        provenance_path = PROVENANCE_ROOT / f"{path.stem}.provenance.json"
        assert provenance_path.is_file(), (
            f"{path}: expected a provenance sidecar at {provenance_path}"
        )
        raw = json.loads(provenance_path.read_text(encoding="utf-8"))
        assert len(raw["records"]) == 1, (
            f"{provenance_path}: depressed_baseline must cite exactly one "
            f"source record, found {raw['records']}"
        )
        assert raw["seams"] == [], (
            f"{provenance_path}: depressed_baseline must have an empty "
            f"seams list (unspliced), found {raw['seams']}"
        )
