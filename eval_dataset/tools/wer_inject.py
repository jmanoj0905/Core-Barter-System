import random

CONFUSION_PAIRS = {
    "there": "their", "their": "there", "to": "too", "too": "to",
    "for": "four", "four": "for", "hear": "here", "here": "hear",
    "write": "right", "right": "write", "know": "no", "no": "know",
    "sea": "see", "see": "sea", "flour": "flower", "flower": "flour",
    "buy": "by", "by": "buy", "wait": "weight", "weight": "wait",
}
FILLER_WORDS = ["um", "uh", "like", "so"]
FALLBACK_VOCAB = ["thing", "stuff", "okay", "yeah", "well", "actually"]


def compute_wer(reference_words: list[str], hypothesis_words: list[str]) -> float:
    """Levenshtein edit distance / len(reference), standard WER definition."""
    n, m = len(reference_words), len(hypothesis_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference_words[i - 1] == hypothesis_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n if n else 0.0


def _substitute(word: str, rng: random.Random) -> str:
    return CONFUSION_PAIRS.get(word.lower(), rng.choice(FALLBACK_VOCAB))


_MAX_ATTEMPTS = 50
_TOLERANCE = 0.02


def _generate_candidate(words: list[dict], target_wer: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n = len(words)
    num_errors = max(1, round(target_wer * n)) if target_wer > 0 else 0
    error_indices = rng.sample(range(n), min(num_errors, n))

    result = []
    for i, w in enumerate(words):
        if i not in error_indices:
            result.append(dict(w))
            continue
        op = rng.choice(["substitute", "delete", "insert"])
        if op == "substitute":
            new_word = dict(w)
            new_word["text"] = _substitute(w["text"], rng)
            result.append(new_word)
        elif op == "delete":
            continue  # word dropped entirely
        else:  # insert: keep original word, add a filler word after it
            result.append(dict(w))
            filler_start = w["end"]
            result.append({"text": rng.choice(FILLER_WORDS), "start": filler_start, "end": filler_start + 0.2})

    return result


def inject_wer(transcript: dict, target_wer: float, seed: int = 0) -> dict:
    words = list(transcript["words"])
    if target_wer <= 0 or not words:
        return {"words": [dict(w) for w in words]}

    original_words = [w["text"] for w in words]

    for attempt in range(_MAX_ATTEMPTS):
        candidate = _generate_candidate(words, target_wer, seed + attempt)
        candidate_words = [w["text"] for w in candidate]
        wer = compute_wer(original_words, candidate_words)
        if abs(wer - target_wer) <= _TOLERANCE:
            return {"words": candidate}

    raise RuntimeError(
        f"inject_wer: failed to reach target_wer={target_wer} within "
        f"±{_TOLERANCE} after {_MAX_ATTEMPTS} attempts (seed={seed})"
    )
