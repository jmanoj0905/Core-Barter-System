"""Corpus-integrity checks for the 15-script calibration corpus (Task 8).

Guards the calibration corpus's own stricter contract: every script must
parse with declared gold DIGRESSION labels, there must be exactly 5 scripts
per category with topics drawn from the shared topic pool, teacher word
counts and window counts must fall in the expected ranges, and authorship
must be rotated across the three agent models.

No mechanism code here (no cosine/embedding/RHO) — this only exercises the
script parser and the offline replay windower, both of which are dependency-
light and runnable under plain system python3.
"""

import re
from pathlib import Path

from eval_dataset.tools.replay import replay, synthetic_durations
from eval_dataset.tools.script_parser import parse_script

REPO_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_ROOT = REPO_ROOT / "eval_dataset" / "scripts" / "calibration"
TOPIC_POOL_PATH = REPO_ROOT / "eval_dataset" / "topics" / "topic_pool.md"

CATEGORIES = ("gradual_drift", "adversarial", "interleaved_abuse")

AUTHORS = {"agent:opus-5", "agent:sonnet-5", "agent:haiku-4.5"}

EXPECTED_TOTAL = 15
EXPECTED_PER_CATEGORY = 5


def _category_scripts(category: str) -> list[Path]:
    return sorted((CALIBRATION_ROOT / category).glob("sess_CAL*.txt"))


def _all_scripts() -> list[Path]:
    paths: list[Path] = []
    for category in CATEGORIES:
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
