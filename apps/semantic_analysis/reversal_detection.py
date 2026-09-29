"""Proximity-based meaning-reversal detection.

Flags text where a negation cue ("not", "refuse to", "won't", ...) sits within
NEGATION_PROXIMITY_TOKENS tokens of a commitment phrase ("pay", "agree to",
...) inside the same sentence, e.g. "I will not pay the agreed amount."

Matching is token-span based rather than a single regex (unlike windowing.py's
_FILLER_PHRASE_RE): a regex can match a phrase but cannot measure the token
distance between two matches, which the proximity rule needs. The lexicons
follow windowing.py's constant style and reuse its _PUNCTUATION trimming set.

Known limitations (documented, pinned by tests):
- Double negation ("not refuse to pay") is flagged like a single negation.
- Lexicon entries are exact tokens, so inflections ("paying", "paid") do not
  match.
"""
import re

from windowing import _PUNCTUATION

COMMITMENT_PHRASES = {
    "pay", "give", "provide", "deliver", "send", "transfer", "return",
    "refund", "owe", "accept", "honor", "complete", "fulfill",
    "commit to", "promise", "agree to", "guarantee",
}

NEGATION_CUES = {
    "not", "never", "no longer", "won't", "can't", "cannot", "wouldn't",
    "shouldn't", "couldn't", "refuse to", "refused to", "unable to",
}

# A negation cue and a commitment phrase are linked if they occur within this
# many tokens of each other in the same sentence.
NEGATION_PROXIMITY_TOKENS = 4

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

# Transcription can emit typographic apostrophes; fold them to ASCII so
# "won’t" matches the "won't" lexicon entry.
_APOSTROPHE_TRANSLATION = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


def _tokenize(sentence: str) -> list[str]:
    sentence = sentence.translate(_APOSTROPHE_TRANSLATION)
    return [t.strip(_PUNCTUATION).lower() for t in sentence.split() if t.strip(_PUNCTUATION)]


def _phrase_spans(tokens: list[str], phrases: set[str]) -> list[tuple[int, int]]:
    spans = []
    for phrase in phrases:
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == phrase_tokens:
                spans.append((i, i + n))
    return spans


def detect_meaning_reversal(cleaned_text: str) -> bool:
    for sentence in _SENTENCE_SPLIT_RE.split(cleaned_text):
        tokens = _tokenize(sentence)
        if not tokens:
            continue
        negation_spans = _phrase_spans(tokens, NEGATION_CUES)
        if not negation_spans:
            continue
        commitment_spans = _phrase_spans(tokens, COMMITMENT_PHRASES)
        if not commitment_spans:
            continue
        for neg_start, neg_end in negation_spans:
            for com_start, com_end in commitment_spans:
                gap = max(neg_start - com_end, com_start - neg_end, 0)
                if gap <= NEGATION_PROXIMITY_TOKENS:
                    return True
    return False
