# Eval Dataset Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling that turns 40 screenplay-style scripts into a labeled, WER-varied audio+transcript eval dataset: script parsing, TTS synthesis, AWS Transcribe integration, synthetic WER injection, 5s-window chunking, an annotation web app, and inter-annotator kappa computation.

**Architecture:** A `eval_dataset/tools/` package of small, independently testable Python modules (parsing, chunking, WER injection, kappa — all pure logic, unit tested with no AWS calls) plus two thin AWS-calling wrappers (`synthesize.py` for Polly, `transcribe.py` for S3+Transcribe, both tested against mocked boto3 clients) and a small FastAPI + single-file React annotation app.

**Tech Stack:** Python 3.11, boto3 (AWS Polly/Transcribe/S3), pydub + ffmpeg (audio concat — ffmpeg already a project dependency per CLAUDE.md), pytest, FastAPI + uvicorn, React (via CDN, no build step) for the annotation UI.

## Global Constraints

- Session categories, exactly these five string values: `clean`, `gradual_drift`, `adversarial`, `code_switch`, `silence` (per design doc §Team split).
- Window size: 5 seconds, matching the production `Window Result` unit (per design doc §Annotation tool).
- Labels: exactly `correct`, `weakly_correct`, `incorrect` (per CLAUDE.md Key Domain Concepts — matches production window classification).
- WER targets: 0 (real), 10, 20, 30 percent, tolerance ±2% (per design doc §WER injection).
- Kappa target: ≥ 0.7 overall (per design doc §Deliverables & success criteria).
- Audio files (`.wav`) tracked via Git LFS; everything else in the dataset is plain text/JSON in normal git (per design doc §Repo layout).
- AWS credentials come from the existing project env vars: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (default `ap-south-1`) — same convention as `apps/audio_pipeline` (per CLAUDE.md STT Configuration).

---

## File Structure

```
eval_dataset/
├── .gitattributes                    # Git LFS rule for *.wav
├── requirements.txt                  # boto3, pydub, fastapi, uvicorn, pytest
├── README.md                         # how to run each tool
├── topics/topic_pool.md              # 40 topics, 4x10 (already specced — copied verbatim)
├── scripts/person_{A,B,C,D}/*.txt    # written by team, not by this plan
├── audio/                            # .wav output of synthesize.py
├── transcripts/raw/, transcripts/synthetic/
├── annotations/
└── tools/
    ├── __init__.py
    ├── script_parser.py              # Task 2
    ├── window_chunker.py             # Task 3
    ├── wer_inject.py                 # Task 4
    ├── kappa.py                      # Task 5
    ├── synthesize.py                 # Task 6 (AWS Polly)
    ├── transcribe.py                 # Task 7 (AWS S3 + Transcribe)
    ├── annotation_app/
    │   ├── main.py                   # Task 8 (FastAPI backend)
    │   └── static/index.html         # Task 9 (single-file React frontend)
    └── tests/
        ├── test_script_parser.py
        ├── test_window_chunker.py
        ├── test_wer_inject.py
        ├── test_kappa.py
        ├── test_synthesize.py
        ├── test_transcribe.py
        └── test_annotation_app.py
```

---

### Task 1: Scaffold directories, topic pool, LFS config, requirements

**Files:**
- Create: `eval_dataset/.gitattributes`
- Create: `eval_dataset/requirements.txt`
- Create: `eval_dataset/topics/topic_pool.md`
- Create: `eval_dataset/README.md`
- Create: `eval_dataset/tools/__init__.py`
- Create: `eval_dataset/scripts/person_A/.gitkeep`, `person_B/.gitkeep`, `person_C/.gitkeep`, `person_D/.gitkeep`
- Create: `eval_dataset/audio/.gitkeep`
- Create: `eval_dataset/transcripts/raw/.gitkeep`, `eval_dataset/transcripts/synthetic/.gitkeep`
- Create: `eval_dataset/annotations/.gitkeep`

**Interfaces:**
- Produces: the directory layout every later task writes into. No code interfaces.

- [ ] **Step 1: Create the directory tree and placeholder files**

```bash
cd /Users/manojj/Documents/CSE-Projects/core-barter-system
mkdir -p eval_dataset/topics eval_dataset/scripts/person_A eval_dataset/scripts/person_B \
  eval_dataset/scripts/person_C eval_dataset/scripts/person_D eval_dataset/audio \
  eval_dataset/transcripts/raw eval_dataset/transcripts/synthetic eval_dataset/annotations \
  eval_dataset/tools/annotation_app/static eval_dataset/tools/tests
touch eval_dataset/scripts/person_A/.gitkeep eval_dataset/scripts/person_B/.gitkeep \
  eval_dataset/scripts/person_C/.gitkeep eval_dataset/scripts/person_D/.gitkeep \
  eval_dataset/audio/.gitkeep eval_dataset/transcripts/raw/.gitkeep \
  eval_dataset/transcripts/synthetic/.gitkeep eval_dataset/annotations/.gitkeep
touch eval_dataset/tools/__init__.py eval_dataset/tools/tests/__init__.py
```

- [ ] **Step 2: Write `.gitattributes` for Git LFS on audio**

```
*.wav filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 3: Write `requirements.txt`**

```
boto3==1.34.*
pydub==0.25.*
fastapi==0.115.*
uvicorn==0.30.*
pytest==8.*
```

- [ ] **Step 4: Write `topics/topic_pool.md`**

Copy the 40-topic pool verbatim from `docs/superpowers/specs/2026-08-11-eval-dataset-design.md` §Topic pool (Person A/B/C/D lists, 10 topics each) into this file, unchanged.

- [ ] **Step 5: Write `README.md`**

```markdown
# Eval Dataset Tools

Build order: write a script in `scripts/person_X/sess_XNN.txt` (format in
`docs/superpowers/specs/2026-08-11-eval-dataset-design.md`), then run:

    python -m eval_dataset.tools.synthesize scripts/person_A/sess_A01.txt
    python -m eval_dataset.tools.transcribe audio/sess_A01.wav
    python -m eval_dataset.tools.wer_inject transcripts/raw/sess_A01_wer0.json

Then annotate via the web app:

    uvicorn eval_dataset.tools.annotation_app.main:app --reload
    # open http://localhost:8000

Requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` env vars
(same as `apps/audio_pipeline`), plus Polly + Transcribe + S3 IAM permissions.
```

- [ ] **Step 6: Verify structure exists**

Run: `find eval_dataset -type d | sort`
Expected: all 11 directories listed (topics, scripts/person_{A,B,C,D}, audio, transcripts/{raw,synthetic}, annotations, tools, tools/annotation_app/static, tools/tests).

- [ ] **Step 7: Commit**

```bash
git add eval_dataset/
git commit -m "chore: scaffold eval_dataset directory structure and topic pool"
```

---

### Task 2: Script parser

Parses the screenplay-style `.txt` format from the design doc into a
structured object every downstream tool (synthesize, chunker) consumes.

**Files:**
- Create: `eval_dataset/tools/script_parser.py`
- Test: `eval_dataset/tools/tests/test_script_parser.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class Turn:
      speaker: str      # "A" or "B"
      text: str

  @dataclass
  class Script:
      topic: str
      teacher: str       # "A" or "B"
      learner: str        # "A" or "B"
      category: str        # one of clean/gradual_drift/adversarial/code_switch/silence
      turns: list[Turn]

  def parse_script(path: str) -> Script: ...
  ```

- [ ] **Step 1: Write the failing test**

```python
# eval_dataset/tools/tests/test_script_parser.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest eval_dataset/tools/tests/test_script_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_dataset.tools.script_parser'`

- [ ] **Step 3: Write the implementation**

```python
# eval_dataset/tools/script_parser.py
from dataclasses import dataclass

VALID_CATEGORIES = {"clean", "gradual_drift", "adversarial", "code_switch", "silence"}


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class Script:
    topic: str
    teacher: str
    learner: str
    category: str
    turns: list[Turn]


def parse_script(path: str) -> Script:
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    header = {}
    turn_lines = []
    in_turns = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            in_turns = True
            continue
        if not in_turns and ":" in stripped and stripped.split(":", 1)[0].isupper():
            key, value = stripped.split(":", 1)
            header[key.strip()] = value.strip()
        elif stripped:
            turn_lines.append(stripped)

    category = header["CATEGORY"]
    if category not in VALID_CATEGORIES:
        raise ValueError(f"invalid category: {category!r}, must be one of {VALID_CATEGORIES}")

    turns = []
    for line in turn_lines:
        speaker, text = line.split(":", 1)
        turns.append(Turn(speaker=speaker.strip(), text=text.strip()))

    return Script(
        topic=header["TOPIC"],
        teacher=header["TEACHER"],
        learner=header["LEARNER"],
        category=category,
        turns=turns,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest eval_dataset/tools/tests/test_script_parser.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/script_parser.py eval_dataset/tools/tests/test_script_parser.py
git commit -m "feat: add screenplay script parser for eval dataset scripts"
```

---

### Task 3: Window chunker

Chunks a transcript (list of words with timestamps) into 5-second windows —
the same unit used by both the annotation app and later baseline scoring.

**Files:**
- Create: `eval_dataset/tools/window_chunker.py`
- Test: `eval_dataset/tools/tests/test_window_chunker.py`

**Interfaces:**
- Consumes: transcripts are `dict` shaped `{"words": [{"text": str, "start": float, "end": float}, ...]}` — this is the shape both `transcribe.py` (Task 7) and `wer_inject.py` (Task 4) produce.
- Produces:
  ```python
  @dataclass
  class Window:
      index: int
      start: float
      end: float
      text: str

  def chunk_into_windows(transcript: dict, window_seconds: float = 5.0) -> list[Window]: ...
  ```

- [ ] **Step 1: Write the failing test**

```python
# eval_dataset/tools/tests/test_window_chunker.py
from eval_dataset.tools.window_chunker import chunk_into_windows, Window

def test_chunk_into_windows_splits_by_word_start_time():
    transcript = {
        "words": [
            {"text": "Alright,", "start": 0.0, "end": 0.4},
            {"text": "let's", "start": 0.4, "end": 0.7},
            {"text": "start", "start": 4.9, "end": 5.2},
            {"text": "with", "start": 5.3, "end": 5.5},
            {"text": "variables.", "start": 9.8, "end": 10.3},
        ]
    }

    windows = chunk_into_windows(transcript, window_seconds=5.0)

    assert windows == [
        Window(index=0, start=0.0, end=5.0, text="Alright, let's start"),
        Window(index=1, start=5.0, end=10.0, text="with variables."),
    ]

def test_chunk_into_windows_empty_transcript_returns_empty_list():
    assert chunk_into_windows({"words": []}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest eval_dataset/tools/tests/test_window_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# eval_dataset/tools/window_chunker.py
from dataclasses import dataclass


@dataclass
class Window:
    index: int
    start: float
    end: float
    text: str


def chunk_into_windows(transcript: dict, window_seconds: float = 5.0) -> list[Window]:
    words = transcript["words"]
    if not words:
        return []

    max_end = max(w["end"] for w in words)
    num_windows = int(max_end // window_seconds) + 1

    windows = []
    for i in range(num_windows):
        w_start = i * window_seconds
        w_end = w_start + window_seconds
        words_in_window = [w["text"] for w in words if w_start <= w["start"] < w_end]
        if words_in_window:
            windows.append(Window(index=i, start=w_start, end=w_end, text=" ".join(words_in_window)))

    return windows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest eval_dataset/tools/tests/test_window_chunker.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/window_chunker.py eval_dataset/tools/tests/test_window_chunker.py
git commit -m "feat: add 5s window chunker for transcripts"
```

---

### Task 4: WER injection

Pure algorithm: takes a real (WER-0) transcript and produces a corrupted
variant hitting a target WER within tolerance.

**Files:**
- Create: `eval_dataset/tools/wer_inject.py`
- Test: `eval_dataset/tools/tests/test_wer_inject.py`

**Interfaces:**
- Consumes: same transcript shape as Task 3 (`{"words": [{"text", "start", "end"}, ...]}`).
- Produces:
  ```python
  def compute_wer(reference_words: list[str], hypothesis_words: list[str]) -> float: ...
  def inject_wer(transcript: dict, target_wer: float, seed: int = 0) -> dict: ...
  ```
  `inject_wer` returns a transcript dict of the same shape, corrupted so that
  `compute_wer(original_words, corrupted_words)` is within ±0.02 of `target_wer`
  (target_wer expressed as a fraction, e.g. `0.10` for 10%).

- [ ] **Step 1: Write the failing test**

```python
# eval_dataset/tools/tests/test_wer_inject.py
from eval_dataset.tools.wer_inject import compute_wer, inject_wer

def test_compute_wer_identical_sequences_is_zero():
    assert compute_wer(["the", "cat", "sat"], ["the", "cat", "sat"]) == 0.0

def test_compute_wer_one_substitution_out_of_three():
    assert compute_wer(["the", "cat", "sat"], ["the", "dog", "sat"]) == 1 / 3

def test_inject_wer_hits_target_within_tolerance():
    transcript = {
        "words": [
            {"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
            for i, w in enumerate(
                ("the quick brown fox jumps over the lazy dog while "
                 "the cat sat quietly on the warm windowsill today").split()
            )
        ]
    }
    original_words = [w["text"] for w in transcript["words"]]

    corrupted = inject_wer(transcript, target_wer=0.20, seed=42)
    corrupted_words = [w["text"] for w in corrupted["words"]]

    wer = compute_wer(original_words, corrupted_words)
    assert abs(wer - 0.20) <= 0.02

def test_inject_wer_zero_target_returns_unchanged_words():
    transcript = {"words": [{"text": "hi", "start": 0.0, "end": 0.3}]}
    corrupted = inject_wer(transcript, target_wer=0.0, seed=1)
    assert [w["text"] for w in corrupted["words"]] == ["hi"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest eval_dataset/tools/tests/test_wer_inject.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# eval_dataset/tools/wer_inject.py
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


def inject_wer(transcript: dict, target_wer: float, seed: int = 0) -> dict:
    rng = random.Random(seed)
    words = list(transcript["words"])
    if target_wer <= 0 or not words:
        return {"words": [dict(w) for w in words]}

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

    return {"words": result}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest eval_dataset/tools/tests/test_wer_inject.py -v`
Expected: PASS (4 passed). If the 20%-target test is flaky near the ±0.02 boundary, that's expected variance from random substitute/delete/insert mix — rerun with the fixed seed; it's deterministic per seed, so a real failure means the algorithm is off, not the test.

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/wer_inject.py eval_dataset/tools/tests/test_wer_inject.py
git commit -m "feat: add synthetic WER injection for transcript robustness testing"
```

---

### Task 5: Inter-annotator kappa

**Files:**
- Create: `eval_dataset/tools/kappa.py`
- Test: `eval_dataset/tools/tests/test_kappa.py`

**Interfaces:**
- Consumes: annotation files shaped `list[{"window_index": int, "label": str}]` (this is the export shape Task 8's `/sessions/{id}/annotate` endpoint writes).
- Produces:
  ```python
  def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float: ...
  def fleiss_kappa(labels_by_annotator: list[list[str]]) -> float: ...
  def session_kappa(annotator_label_lists: list[list[str]]) -> float: ...
  ```
  `session_kappa` dispatches: 2 annotators → `cohens_kappa`, 3+ → `fleiss_kappa`.

- [ ] **Step 1: Write the failing test**

```python
# eval_dataset/tools/tests/test_kappa.py
from eval_dataset.tools.kappa import cohens_kappa, fleiss_kappa, session_kappa

def test_cohens_kappa_perfect_agreement_is_one():
    labels = ["correct", "weakly_correct", "incorrect", "correct"]
    assert cohens_kappa(labels, labels) == 1.0

def test_cohens_kappa_no_agreement_beyond_chance_is_near_zero():
    a = ["correct"] * 10
    b = ["incorrect"] * 10
    # constant labels -> undefined chance agreement; kappa defined as 0.0 by convention here
    assert cohens_kappa(a, b) == 0.0

def test_fleiss_kappa_perfect_agreement_is_one():
    labels_by_annotator = [
        ["correct", "incorrect", "weakly_correct"],
        ["correct", "incorrect", "weakly_correct"],
        ["correct", "incorrect", "weakly_correct"],
    ]
    assert fleiss_kappa(labels_by_annotator) == 1.0

def test_session_kappa_dispatches_by_annotator_count():
    two = [["correct", "incorrect"], ["correct", "incorrect"]]
    three = [["correct"], ["correct"], ["correct"]]
    assert session_kappa(two) == cohens_kappa(*two)
    assert session_kappa(three) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest eval_dataset/tools/tests/test_kappa.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# eval_dataset/tools/kappa.py
from collections import Counter

LABELS = ("correct", "weakly_correct", "incorrect")


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    observed_agreement = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    chance_agreement = sum((count_a[l] / n) * (count_b[l] / n) for l in LABELS)

    if chance_agreement >= 1.0:
        return 0.0
    return (observed_agreement - chance_agreement) / (1 - chance_agreement)


def fleiss_kappa(labels_by_annotator: list[list[str]]) -> float:
    num_annotators = len(labels_by_annotator)
    num_items = len(labels_by_annotator[0])

    # counts[item][label] = number of annotators assigning that label
    counts = []
    for item_idx in range(num_items):
        item_labels = [labels_by_annotator[a][item_idx] for a in range(num_annotators)]
        counts.append(Counter(item_labels))

    p_item = []
    for item_counts in counts:
        agreements = sum(c * (c - 1) for c in item_counts.values())
        p_item.append(agreements / (num_annotators * (num_annotators - 1)))
    p_bar = sum(p_item) / num_items

    label_totals = Counter()
    for item_counts in counts:
        label_totals.update(item_counts)
    total_ratings = num_items * num_annotators
    p_e = sum((label_totals[l] / total_ratings) ** 2 for l in LABELS)

    if p_e >= 1.0:
        return 0.0
    return (p_bar - p_e) / (1 - p_e)


def session_kappa(annotator_label_lists: list[list[str]]) -> float:
    if len(annotator_label_lists) == 2:
        return cohens_kappa(*annotator_label_lists)
    return fleiss_kappa(annotator_label_lists)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest eval_dataset/tools/tests/test_kappa.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/kappa.py eval_dataset/tools/tests/test_kappa.py
git commit -m "feat: add Cohen's/Fleiss' kappa for inter-annotator agreement"
```

---

### Task 6: TTS synthesis (AWS Polly)

**Files:**
- Create: `eval_dataset/tools/synthesize.py`
- Test: `eval_dataset/tools/tests/test_synthesize.py`

**Interfaces:**
- Consumes: `Script`/`Turn` from Task 2 (`eval_dataset.tools.script_parser`).
- Produces:
  ```python
  VOICE_MAP = {"A": "Joanna", "B": "Matthew"}  # AWS Polly neural voice IDs

  def synthesize_turn(polly_client, text: str, voice_id: str) -> bytes: ...
  def synthesize_script(script: Script, polly_client, output_wav_path: str, timing_json_path: str) -> None: ...
  ```
  `synthesize_script` writes the concatenated `.wav` to `output_wav_path` and a
  `turns_timing.json` (`list[{"speaker", "text", "start", "end"}]`) to `timing_json_path`.

- [ ] **Step 1: Write the failing test**

```python
# eval_dataset/tools/tests/test_synthesize.py
import json
import wave
from unittest.mock import MagicMock
import io

from eval_dataset.tools.script_parser import Script, Turn
from eval_dataset.tools.synthesize import synthesize_script, VOICE_MAP


def _fake_wav_bytes(duration_seconds=0.5, framerate=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.set_nchannels_or_defaults = None
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(framerate)
        f.writeframes(b"\x00\x00" * int(framerate * duration_seconds))
    return buf.getvalue()


def test_synthesize_script_calls_polly_per_turn_with_correct_voice(tmp_path):
    script = Script(
        topic="Python", teacher="A", learner="B", category="clean",
        turns=[Turn(speaker="A", text="Hello there."), Turn(speaker="B", text="Hi!")],
    )
    mock_client = MagicMock()
    mock_client.synthesize_speech.return_value = {"AudioStream": io.BytesIO(_fake_wav_bytes())}

    wav_path = tmp_path / "out.wav"
    timing_path = tmp_path / "timing.json"
    synthesize_script(script, mock_client, str(wav_path), str(timing_path))

    calls = mock_client.synthesize_speech.call_args_list
    assert calls[0].kwargs["VoiceId"] == VOICE_MAP["A"]
    assert calls[0].kwargs["Text"] == "Hello there."
    assert calls[1].kwargs["VoiceId"] == VOICE_MAP["B"]
    assert calls[1].kwargs["Text"] == "Hi!"
    assert calls[0].kwargs["OutputFormat"] == "pcm"

    assert wav_path.exists()
    timing = json.loads(timing_path.read_text())
    assert len(timing) == 2
    assert timing[0]["speaker"] == "A"
    assert timing[1]["start"] > timing[0]["start"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest eval_dataset/tools/tests/test_synthesize.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# eval_dataset/tools/synthesize.py
import json
import wave
import io

from eval_dataset.tools.script_parser import Script

VOICE_MAP = {"A": "Joanna", "B": "Matthew"}
PAUSE_SECONDS = 0.6
SAMPLE_RATE = 8000  # Polly "pcm" output format is 16-bit signed LE, 8kHz or 16kHz


def synthesize_turn(polly_client, text: str, voice_id: str) -> bytes:
    response = polly_client.synthesize_speech(
        Text=text, VoiceId=voice_id, OutputFormat="pcm", Engine="neural",
    )
    return response["AudioStream"].read()


def synthesize_script(script: Script, polly_client, output_wav_path: str, timing_json_path: str) -> None:
    pcm_chunks = []
    timing = []
    cursor = 0.0
    silence_frame = b"\x00\x00" * int(SAMPLE_RATE * PAUSE_SECONDS)

    for turn in script.turns:
        voice_id = VOICE_MAP[turn.speaker]
        pcm = synthesize_turn(polly_client, turn.text, voice_id)
        duration = len(pcm) / 2 / SAMPLE_RATE  # 16-bit samples
        timing.append({"speaker": turn.speaker, "text": turn.text, "start": cursor, "end": cursor + duration})
        cursor += duration + PAUSE_SECONDS
        pcm_chunks.append(pcm)
        pcm_chunks.append(silence_frame)

    with wave.open(output_wav_path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(b"".join(pcm_chunks))

    with open(timing_json_path, "w") as f:
        json.dump(timing, f, indent=2)


if __name__ == "__main__":
    import sys
    import boto3
    from eval_dataset.tools.script_parser import parse_script

    script_path = sys.argv[1]
    script = parse_script(script_path)
    session_id = script_path.split("/")[-1].removesuffix(".txt")
    client = boto3.client("polly")
    synthesize_script(
        script, client,
        f"eval_dataset/audio/{session_id}.wav",
        f"eval_dataset/audio/{session_id}_timing.json",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest eval_dataset/tools/tests/test_synthesize.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/synthesize.py eval_dataset/tools/tests/test_synthesize.py
git commit -m "feat: add AWS Polly TTS synthesis for scripted sessions"
```

---

### Task 7: AWS Transcribe integration

**Files:**
- Create: `eval_dataset/tools/transcribe.py`
- Test: `eval_dataset/tools/tests/test_transcribe.py`

**Interfaces:**
- Produces:
  ```python
  def transcribe_audio(s3_client, transcribe_client, wav_path: str, bucket: str, job_name: str) -> dict: ...
  ```
  Returns a transcript in the same shape Task 3/4 consume: `{"words": [{"text", "start", "end"}, ...]}`.
  Uploads to S3, starts a Transcribe job, polls until `COMPLETED`/`FAILED`, downloads
  and reshapes the result, then deletes the S3 object (transient scratch use only,
  per design doc §Cloud services required).

- [ ] **Step 1: Write the failing test**

```python
# eval_dataset/tools/tests/test_transcribe.py
import json
from unittest.mock import MagicMock, patch
from eval_dataset.tools.transcribe import transcribe_audio


def test_transcribe_audio_uploads_polls_downloads_and_cleans_up(tmp_path):
    wav_path = tmp_path / "sess_A01.wav"
    wav_path.write_bytes(b"fake-wav-bytes")

    s3_client = MagicMock()
    transcribe_client = MagicMock()
    transcribe_client.get_transcription_job.side_effect = [
        {"TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}},
        {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "COMPLETED",
                "Transcript": {"TranscriptFileUri": "https://fake-uri/transcript.json"},
            }
        },
    ]

    aws_transcript_payload = {
        "results": {
            "items": [
                {"type": "pronunciation", "alternatives": [{"content": "Hello"}], "start_time": "0.0", "end_time": "0.4"},
                {"type": "punctuation", "alternatives": [{"content": "."}]},
                {"type": "pronunciation", "alternatives": [{"content": "there"}], "start_time": "0.5", "end_time": "0.9"},
            ]
        }
    }

    with patch("eval_dataset.tools.transcribe._download_json", return_value=aws_transcript_payload), \
         patch("eval_dataset.tools.transcribe.time.sleep"):
        result = transcribe_audio(s3_client, transcribe_client, str(wav_path), bucket="test-bucket", job_name="job-1")

    s3_client.upload_file.assert_called_once_with(str(wav_path), "test-bucket", "job-1.wav")
    transcribe_client.start_transcription_job.assert_called_once()
    s3_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key="job-1.wav")

    assert result == {"words": [{"text": "Hello", "start": 0.0, "end": 0.4}, {"text": "there", "start": 0.5, "end": 0.9}]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest eval_dataset/tools/tests/test_transcribe.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# eval_dataset/tools/transcribe.py
import time
import urllib.request
import json


def _download_json(uri: str) -> dict:
    with urllib.request.urlopen(uri) as response:
        return json.loads(response.read())


def _reshape_transcript(aws_payload: dict) -> dict:
    words = []
    for item in aws_payload["results"]["items"]:
        if item["type"] != "pronunciation":
            continue
        words.append({
            "text": item["alternatives"][0]["content"],
            "start": float(item["start_time"]),
            "end": float(item["end_time"]),
        })
    return {"words": words}


def transcribe_audio(s3_client, transcribe_client, wav_path: str, bucket: str, job_name: str) -> dict:
    s3_key = f"{job_name}.wav"
    s3_client.upload_file(wav_path, bucket, s3_key)

    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": f"s3://{bucket}/{s3_key}"},
        MediaFormat="wav",
        LanguageCode="en-US",
    )

    while True:
        response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = response["TranscriptionJob"]["TranscriptionJobStatus"]
        if status in ("COMPLETED", "FAILED"):
            break
        time.sleep(5)

    if status == "FAILED":
        s3_client.delete_object(Bucket=bucket, Key=s3_key)
        raise RuntimeError(f"Transcribe job {job_name} failed")

    uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    aws_payload = _download_json(uri)
    s3_client.delete_object(Bucket=bucket, Key=s3_key)

    return _reshape_transcript(aws_payload)


if __name__ == "__main__":
    import sys
    import boto3
    import os

    wav_path = sys.argv[1]
    job_name = wav_path.split("/")[-1].removesuffix(".wav")
    bucket = os.environ.get("AWS_S3_BUCKET", "core-barter-audio-tmp")
    s3 = boto3.client("s3")
    transcribe = boto3.client("transcribe")
    result = transcribe_audio(s3, transcribe, wav_path, bucket, job_name)
    with open(f"eval_dataset/transcripts/raw/{job_name}_wer0.json", "w") as f:
        json.dump(result, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest eval_dataset/tools/tests/test_transcribe.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/transcribe.py eval_dataset/tools/tests/test_transcribe.py
git commit -m "feat: add AWS Transcribe integration for real WER-0 transcripts"
```

---

### Task 8: Annotation backend (FastAPI)

**Files:**
- Create: `eval_dataset/tools/annotation_app/__init__.py`
- Create: `eval_dataset/tools/annotation_app/main.py`
- Test: `eval_dataset/tools/tests/test_annotation_app.py`

**Interfaces:**
- Consumes: `chunk_into_windows` (Task 3), transcripts from `eval_dataset/transcripts/raw/*_wer0.json`, audio from `eval_dataset/audio/*.wav`.
- Produces three HTTP endpoints:
  - `GET /sessions` → `list[{"session_id": str, "topic": str}]`
  - `GET /sessions/{session_id}/windows` → `list[{"index": int, "start": float, "end": float, "text": str}]`
  - `POST /sessions/{session_id}/annotate?annotator=<name>` body `list[{"window_index": int, "label": str}]` → writes `eval_dataset/annotations/{session_id}_{annotator}.json`, returns `{"status": "saved"}`
  - Static file serving of `eval_dataset/audio/` under `/audio/`.

- [ ] **Step 1: Write the failing test**

```python
# eval_dataset/tools/tests/test_annotation_app.py
import json
import wave
from fastapi.testclient import TestClient


def _write_test_fixtures(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    raw_dir = tmp_path / "transcripts" / "raw"
    ann_dir = tmp_path / "annotations"
    audio_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    with wave.open(str(audio_dir / "sess_A01.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\x00\x00" * 8000)

    transcript = {"words": [{"text": "Hello", "start": 0.0, "end": 0.4}, {"text": "there.", "start": 0.5, "end": 0.9}]}
    (raw_dir / "sess_A01_wer0.json").write_text(json.dumps(transcript))

    import eval_dataset.tools.annotation_app.main as main_module
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(audio_dir))
    monkeypatch.setattr(main_module, "TRANSCRIPTS_DIR", str(raw_dir))
    monkeypatch.setattr(main_module, "ANNOTATIONS_DIR", str(ann_dir))
    return main_module, ann_dir


def test_list_sessions_returns_sessions_with_transcripts(tmp_path, monkeypatch):
    main_module, _ = _write_test_fixtures(tmp_path, monkeypatch)
    client = TestClient(main_module.app)

    response = client.get("/sessions")

    assert response.status_code == 200
    assert response.json() == [{"session_id": "sess_A01"}]


def test_get_windows_returns_chunked_transcript(tmp_path, monkeypatch):
    main_module, _ = _write_test_fixtures(tmp_path, monkeypatch)
    client = TestClient(main_module.app)

    response = client.get("/sessions/sess_A01/windows")

    assert response.status_code == 200
    assert response.json() == [{"index": 0, "start": 0.0, "end": 5.0, "text": "Hello there."}]


def test_post_annotation_saves_file(tmp_path, monkeypatch):
    main_module, ann_dir = _write_test_fixtures(tmp_path, monkeypatch)
    client = TestClient(main_module.app)

    response = client.post(
        "/sessions/sess_A01/annotate?annotator=bob",
        json=[{"window_index": 0, "label": "correct"}],
    )

    assert response.status_code == 200
    assert response.json() == {"status": "saved"}
    saved = json.loads((ann_dir / "sess_A01_bob.json").read_text())
    assert saved == [{"window_index": 0, "label": "correct"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest eval_dataset/tools/tests/test_annotation_app.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# eval_dataset/tools/annotation_app/__init__.py
```

```python
# eval_dataset/tools/annotation_app/main.py
import json
import os
import glob

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from eval_dataset.tools.window_chunker import chunk_into_windows

AUDIO_DIR = "eval_dataset/audio"
TRANSCRIPTS_DIR = "eval_dataset/transcripts/raw"
ANNOTATIONS_DIR = "eval_dataset/annotations"

app = FastAPI()


@app.get("/sessions")
def list_sessions():
    files = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*_wer0.json")))
    session_ids = [os.path.basename(f).removesuffix("_wer0.json") for f in files]
    return [{"session_id": sid} for sid in session_ids]


@app.get("/sessions/{session_id}/windows")
def get_windows(session_id: str):
    with open(os.path.join(TRANSCRIPTS_DIR, f"{session_id}_wer0.json")) as f:
        transcript = json.load(f)
    windows = chunk_into_windows(transcript)
    return [{"index": w.index, "start": w.start, "end": w.end, "text": w.text} for w in windows]


@app.post("/sessions/{session_id}/annotate")
def save_annotation(session_id: str, annotator: str, labels: list[dict]):
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    out_path = os.path.join(ANNOTATIONS_DIR, f"{session_id}_{annotator}.json")
    with open(out_path, "w") as f:
        json.dump(labels, f, indent=2)
    return {"status": "saved"}


if os.path.isdir(AUDIO_DIR):
    app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
```

Note: `save_annotation`'s `labels` parameter is a FastAPI body parameter
(`list[dict]`), and `annotator` is a query parameter — matches the test's
`client.post("/sessions/sess_A01/annotate?annotator=bob", json=[...])` call shape.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest eval_dataset/tools/tests/test_annotation_app.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/annotation_app/__init__.py eval_dataset/tools/annotation_app/main.py \
  eval_dataset/tools/tests/test_annotation_app.py
git commit -m "feat: add FastAPI backend for window annotation app"
```

---

### Task 9: Annotation frontend (single-file React)

**Files:**
- Create: `eval_dataset/tools/annotation_app/static/index.html`

**Interfaces:**
- Consumes: the three endpoints from Task 8 (`GET /sessions`, `GET /sessions/{id}/windows`, `POST /sessions/{id}/annotate?annotator=<name>`) and static audio at `/audio/{session_id}.wav`.
- No further consumers — this is the UI leaf.

- [ ] **Step 1: Write the frontend**

```html
<!-- eval_dataset/tools/annotation_app/static/index.html -->
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Eval Dataset Annotation</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 2rem auto; }
    .window-row { display: flex; align-items: center; gap: 1rem; padding: 0.5rem 0; border-bottom: 1px solid #ddd; }
    .window-text { flex: 1; }
    button { cursor: pointer; }
    button.selected { font-weight: bold; outline: 2px solid #333; }
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
    const { useState, useEffect } = React;
    const LABELS = ["correct", "weakly_correct", "incorrect"];

    function App() {
      const [annotator, setAnnotator] = useState("");
      const [sessions, setSessions] = useState([]);
      const [sessionId, setSessionId] = useState(null);
      const [windows, setWindows] = useState([]);
      const [labels, setLabels] = useState({});

      useEffect(() => {
        fetch("/sessions").then(r => r.json()).then(setSessions);
      }, []);

      useEffect(() => {
        if (!sessionId) return;
        fetch(`/sessions/${sessionId}/windows`).then(r => r.json()).then(ws => {
          setWindows(ws);
          setLabels({});
        });
      }, [sessionId]);

      function setLabel(index, label) {
        setLabels(prev => ({ ...prev, [index]: label }));
      }

      function submit() {
        const payload = windows.map(w => ({ window_index: w.index, label: labels[w.index] || null }));
        fetch(`/sessions/${sessionId}/annotate?annotator=${encodeURIComponent(annotator)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }).then(() => alert("Saved."));
      }

      if (!annotator) {
        return (
          <div>
            <h2>Enter your name</h2>
            <input onKeyDown={e => e.key === "Enter" && setAnnotator(e.target.value)} placeholder="e.g. bob" />
          </div>
        );
      }

      if (!sessionId) {
        return (
          <div>
            <h2>Sessions</h2>
            <ul>
              {sessions.map(s => (
                <li key={s.session_id}>
                  <button onClick={() => setSessionId(s.session_id)}>{s.session_id}</button>
                </li>
              ))}
            </ul>
          </div>
        );
      }

      return (
        <div>
          <h2>{sessionId} — annotator: {annotator}</h2>
          <audio controls src={`/audio/${sessionId}.wav`} style={{ width: "100%" }} />
          {windows.map(w => (
            <div className="window-row" key={w.index}>
              <span>{w.start.toFixed(1)}s–{w.end.toFixed(1)}s</span>
              <span className="window-text">{w.text}</span>
              {LABELS.map(l => (
                <button
                  key={l}
                  className={labels[w.index] === l ? "selected" : ""}
                  onClick={() => setLabel(w.index, l)}
                >{l}</button>
              ))}
            </div>
          ))}
          <button onClick={submit}>Submit annotations</button>
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById("root")).render(<App />);
  </script>
</body>
</html>
```

- [ ] **Step 2: Manual test**

Run: `uvicorn eval_dataset.tools.annotation_app.main:app --reload` from the repo root, then open `http://localhost:8000` in a browser.
Expected: name prompt → session list → clicking a session shows an audio player and a list of 5s windows, each with three label buttons; clicking a label highlights it; "Submit annotations" writes `eval_dataset/annotations/{session}_{name}.json` (verify by checking the file appears with the clicked labels).

- [ ] **Step 3: Commit**

```bash
git add eval_dataset/tools/annotation_app/static/index.html
git commit -m "feat: add single-file React frontend for window annotation"
```

---

### Task 10: Kappa report CLI

Aggregates all annotation files for a session and prints the kappa report,
flagging sessions below the 0.7 target (per design doc §Deliverables & success criteria).

**Files:**
- Create: `eval_dataset/tools/compute_kappa.py`
- Test: `eval_dataset/tools/tests/test_compute_kappa.py`

**Interfaces:**
- Consumes: `session_kappa` from Task 5 (`eval_dataset.tools.kappa`), reads `eval_dataset/annotations/{session_id}_{annotator}.json` files (shape: `list[{"window_index": int, "label": str}]`, matching Task 8's write format).
- Produces:
  ```python
  def load_session_annotations(annotations_dir: str, session_id: str) -> list[list[str]]: ...
  def report_all_sessions(annotations_dir: str) -> dict[str, float]: ...
  ```
  `report_all_sessions` returns `{session_id: kappa_value}` for every session
  with 2+ annotators found in `annotations_dir`.

- [ ] **Step 1: Write the failing test**

```python
# eval_dataset/tools/tests/test_compute_kappa.py
import json
from eval_dataset.tools.compute_kappa import load_session_annotations, report_all_sessions


def test_load_session_annotations_orders_by_window_index(tmp_path):
    (tmp_path / "sess_A01_bob.json").write_text(json.dumps([
        {"window_index": 1, "label": "incorrect"},
        {"window_index": 0, "label": "correct"},
    ]))
    (tmp_path / "sess_A01_amy.json").write_text(json.dumps([
        {"window_index": 0, "label": "correct"},
        {"window_index": 1, "label": "incorrect"},
    ]))

    result = load_session_annotations(str(tmp_path), "sess_A01")

    assert sorted(result) == sorted([["correct", "incorrect"], ["correct", "incorrect"]])


def test_report_all_sessions_computes_kappa_per_session(tmp_path):
    (tmp_path / "sess_A01_bob.json").write_text(json.dumps([{"window_index": 0, "label": "correct"}]))
    (tmp_path / "sess_A01_amy.json").write_text(json.dumps([{"window_index": 0, "label": "correct"}]))
    (tmp_path / "sess_B01_bob.json").write_text(json.dumps([{"window_index": 0, "label": "correct"}]))

    result = report_all_sessions(str(tmp_path))

    assert result == {"sess_A01": 1.0}  # sess_B01 skipped: only 1 annotator
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest eval_dataset/tools/tests/test_compute_kappa.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# eval_dataset/tools/compute_kappa.py
import glob
import json
import os
from collections import defaultdict

from eval_dataset.tools.kappa import session_kappa


def load_session_annotations(annotations_dir: str, session_id: str) -> list[list[str]]:
    files = sorted(glob.glob(os.path.join(annotations_dir, f"{session_id}_*.json")))
    annotator_labels = []
    for path in files:
        with open(path) as f:
            entries = json.load(f)
        ordered = sorted(entries, key=lambda e: e["window_index"])
        annotator_labels.append([e["label"] for e in ordered])
    return annotator_labels


def report_all_sessions(annotations_dir: str) -> dict[str, float]:
    sessions = defaultdict(int)
    for path in glob.glob(os.path.join(annotations_dir, "*.json")):
        basename = os.path.basename(path).removesuffix(".json")
        session_id = basename.rsplit("_", 1)[0]
        sessions[session_id] += 1

    report = {}
    for session_id, count in sessions.items():
        if count < 2:
            continue
        annotator_labels = load_session_annotations(annotations_dir, session_id)
        report[session_id] = session_kappa(annotator_labels)
    return report


if __name__ == "__main__":
    report = report_all_sessions("eval_dataset/annotations")
    for session_id, kappa in sorted(report.items()):
        flag = "  <-- BELOW 0.7 TARGET" if kappa < 0.7 else ""
        print(f"{session_id}: kappa={kappa:.3f}{flag}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest eval_dataset/tools/tests/test_compute_kappa.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/compute_kappa.py eval_dataset/tools/tests/test_compute_kappa.py
git commit -m "feat: add kappa report CLI aggregating all session annotations"
```

---

## Self-Review Notes

- **Spec coverage:** topic pool (Task 1), script format (Task 2), TTS synthesis (Task 6), cloud services AWS Polly/Transcribe (Tasks 6-7), WER injection (Task 4), annotation tool (Tasks 8-9), kappa/success criteria (Tasks 5, 10), Git LFS storage (Task 1) — all covered.
- **Type consistency:** transcript shape `{"words": [{"text","start","end"}]}` is produced by Task 7 and Task 4, consumed by Task 3 and Task 8 — consistent throughout. Annotation file shape `list[{"window_index","label"}]` is written by Task 8, read by Task 10 — consistent. Label vocabulary (`correct`/`weakly_correct`/`incorrect`) matches CLAUDE.md's domain concepts throughout.
- **No placeholders:** all steps contain complete, runnable code.
