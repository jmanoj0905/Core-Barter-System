import pytest

from reversal_detection import detect_meaning_reversal


def test_reversal_detected_simple_negation():
    assert detect_meaning_reversal("I will not pay the agreed amount.") is True


def test_commitment_alone_not_flagged():
    assert detect_meaning_reversal("I will pay the agreed amount.") is False


def test_negation_far_from_commitment_not_flagged():
    text = "I am not sure about the weather, but I will pay you the full amount."
    assert detect_meaning_reversal(text) is False


def test_negation_in_different_sentence_not_flagged():
    text = "I won't be free tomorrow. I will pay you the full amount."
    assert detect_meaning_reversal(text) is False


def test_multiword_negation_and_commitment_phrase():
    assert detect_meaning_reversal("I refuse to honor the agreement.") is True


def test_no_commitment_and_no_negation_not_flagged():
    assert detect_meaning_reversal("The weather has been nice this week.") is False


def test_double_negation_known_limitation_misfires():
    # Documented known limitation (see design spec): "not ... refuse to pay"
    # actually affirms the commitment, but proximity-based detection cannot
    # tell double negation from single negation and flags it anyway. This
    # test pins that CURRENT (imperfect) behavior so a future change to it
    # is a deliberate, visible decision rather than a silent regression.
    assert detect_meaning_reversal("I will definitely not refuse to pay.") is True


def test_empty_and_whitespace_string():
    assert detect_meaning_reversal("") is False
    assert detect_meaning_reversal("   ") is False
    assert detect_meaning_reversal("...") is False


@pytest.mark.parametrize("text", [
    "I won't pay you.",
    "I won’t pay you.",
    "I can’t deliver the goods.",
    "I can't deliver the goods.",
    "I cannot honor this.",
    "I can not honor this.",
    "I will never give it back.",
    "I no longer agree to the deal.",
])
def test_contractions_and_curly_apostrophes(text):
    assert detect_meaning_reversal(text) is True


@pytest.mark.parametrize("text", [
    "I won't be free tomorrow! I will pay you the full amount.",
    "Won’t you be there? I will pay you the full amount.",
])
def test_scoping_respects_exclamation_and_question_marks(text):
    assert detect_meaning_reversal(text) is False


def test_negation_and_commitment_within_same_sentence_after_question_mark_split():
    assert detect_meaning_reversal("Are you ready? I will not pay you.") is True


def test_negation_after_commitment_within_proximity():
    assert detect_meaning_reversal("Pay you? Never.") is False  # split by "?"
    assert detect_meaning_reversal("I pay you never again") is True


def test_proximity_boundary():
    # gap of exactly 4 tokens is linked; 5 is not.
    assert detect_meaning_reversal("not a b c d pay") is True
    assert detect_meaning_reversal("not a b c d e pay") is False


def test_inflected_forms_not_matched_documented_limitation():
    # Lexicon is exact tokens per the spec; inflections do not match. This
    # pins the current (imperfect) behavior; expanding the lexicon is a
    # deliberate decision, not something to do silently.
    assert detect_meaning_reversal("I will not be paying you.") is False
    assert detect_meaning_reversal("I have not paid you.") is False
