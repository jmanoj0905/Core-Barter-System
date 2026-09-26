"""Cleaning behaviour that threshold calibration depends on.

Thresholds in main.py were fitted on cosine scores produced by this exact
cleaning (ground_truth/generate_synthetic.py reuses clean_text). A silent change
here invalidates them, so the guarantees are pinned.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture(scope="module")
def sa():
    spec = importlib.util.spec_from_file_location("sa_main_clean", APP_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filler_adjacent_to_punctuation_is_stripped(sa):
    """The old version compared raw tokens, so "um," survived and "um" didn't."""
    assert "um" not in sa.clean_text("Um, a comprehension builds a list.").lower()
    assert "well" not in sa.clean_text("Hope it turns out well.").lower()


def test_multi_word_fillers_are_stripped(sa):
    cleaned = sa.clean_text("You know, hydration matters, i mean a lot.")
    assert "you know" not in cleaned.lower()
    assert "i mean" not in cleaned.lower()
    assert "hydration" in cleaned


def test_multi_word_removal_is_word_bounded(sa):
    """"i mean" must not eat the middle of an unrelated word."""
    assert "dopamine" in sa.clean_text("The dopamine response is immediate.")


def test_no_dangling_punctuation_after_removal(sa):
    cleaned = sa.clean_text("You know, hydration matters.")
    assert not cleaned.startswith(",")
    assert " ," not in cleaned


def test_content_words_that_double_as_fillers_are_still_removed(sa):
    """Deliberate: an ablation showed keeping these costs 5/60 held-out
    predictions. See ground_truth/design_decisions.md (D5)."""
    for word in ("like", "so", "basically", "actually", "literally", "okay"):
        assert word not in sa.clean_text(f"A comprehension is {word} a for loop.").lower()


def test_filler_only_window_cleans_to_too_few_tokens(sa):
    """Must fall below MIN_CONTENT_TOKENS so process_window skips it rather than
    scoring noise as `incorrect` and escalating a warning (D6)."""
    cleaned = sa.clean_text("So basically, like, well, okay.")
    assert len(cleaned.split()) < sa.MIN_CONTENT_TOKENS


def test_real_content_survives_above_the_skip_threshold(sa):
    cleaned = sa.clean_text("Um, so a list comprehension builds a new list in one line.")
    assert len(cleaned.split()) >= sa.MIN_CONTENT_TOKENS
    assert "comprehension" in cleaned


def test_calibrated_thresholds_are_ordered_and_in_fitted_range(sa):
    assert sa.LOWER < sa.UPPER
    assert (sa.UPPER, sa.LOWER) == (0.36, 0.14), (
        "thresholds changed without updating design_decisions.md (D1, D2)"
    )
    assert sa.classify(sa.UPPER) == "correct"
    assert sa.classify(sa.LOWER) == "weakly_correct"
    assert sa.classify(sa.LOWER - 0.01) == "incorrect"
