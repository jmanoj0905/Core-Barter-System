import re

# Cosine similarity thresholds, calibrated on ground_truth/synthetic_dataset.csv.
# NOT hand-picked: UPPER is the grid-search optimum, stable in 60/60 leave-one-out
# and 6/6 leave-one-topic-out folds. LOWER is set below its accuracy optimum on
# purpose — see design_decisions.md (D1, D2) for the false-accusation tradeoff.
# The previous 0.55 / 0.35 scored worst of every option tested (28/60).
UPPER = 0.36  # >= UPPER → correct
LOWER = 0.14  # LOWER..UPPER → weakly_correct, < LOWER → incorrect

# Filler words stripped before embedding. Kept deliberately aggressive: an
# ablation over the 60-row fixture (ground_truth/design_decisions.md, D5) showed
# that narrowing this to pure vocalizations — leaving "like", "so", "well",
# "basically", "literally", "actually", "okay" in the text — cost 5 of 60
# held-out predictions (45/60 -> 40/60), exactly matching no cleaning at all.
# These markers dilute a window toward generic conversational language; removing
# them concentrates the topical content the cosine has to work with.
FILLER_WORDS = {
    "uh", "um", "er", "ah", "like", "you know", "i mean",
    "basically", "literally", "actually", "so", "well", "okay",
}

# Multi-word entries above can never match a whitespace token, so they are
# removed from the raw string first. Before this, "you know" and "i mean" sat
# in the set but were dead weight — split() never produces them.
FILLER_PHRASES = tuple(w for w in FILLER_WORDS if " " in w)

# Word-bounded so "i mean" is stripped but "dopamine" survives.
_FILLER_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in sorted(FILLER_PHRASES)) + r")\b",
    re.IGNORECASE,
)

# Trimmed from a token before filler matching. Hyphen excluded so hyphenated
# words stay intact.
_PUNCTUATION = ".,!?;:\"'()[]{}…"

# A window with fewer content tokens than this after cleaning carries too
# little topical evidence to classify, and is skipped rather than scored. See
# design_decisions.md (D6): scoring it would almost certainly yield a low
# cosine and thus a spurious `incorrect`, which is what escalates warnings.
MIN_CONTENT_TOKENS = 3

# Window triggers when accumulated audio >= this many seconds
WINDOW_DURATION_THRESHOLD = 25.0


def clean_text(text: str) -> str:
    """Strip filler words and extra whitespace.

    Matching ignores surrounding punctuation, so "um," is dropped the same way
    "um" is — the previous version compared raw tokens and silently kept every
    filler that happened to sit beside a comma. Casing of kept tokens is now
    preserved: the encoder's tokenizer is uncased, so lowercasing changed no
    embedding (verified on the fixture) and only made logged text harder to read.
    """
    stripped = _FILLER_PHRASE_RE.sub(" ", text)
    kept = [t for t in stripped.split() if t.strip(_PUNCTUATION).lower() not in FILLER_WORDS]
    return _tidy(" ".join(kept))


def _tidy(text: str) -> str:
    """Repair punctuation left dangling by filler removal.

    Dropping a leading "You know," leaves ", hydration matters" — a stray comma
    the encoder would tokenize. Close the gap and trim punctuation that no
    longer follows a word.
    """
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"^[\s,.!?;:]+", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def classify(similarity: float) -> str:
    if similarity >= UPPER:
        return "correct"
    if similarity >= LOWER:
        return "weakly_correct"
    return "incorrect"


def topic_text(topic: str, scope: str) -> str:
    return f"{topic}. {scope}"


def window_is_ready(accumulated_seconds: float) -> bool:
    return accumulated_seconds >= WINDOW_DURATION_THRESHOLD


def has_enough_content(cleaned: str) -> bool:
    return len(cleaned.split()) >= MIN_CONTENT_TOKENS
