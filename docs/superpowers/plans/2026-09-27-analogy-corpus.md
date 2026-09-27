# Analogy Corpus & `RHO` Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 25-session evaluation corpus and a production-faithful measurement harness, then calibrate or reject `RHO = 0.45` in `thr = RHO · R`.

**Architecture:** Harness first. Pure windowing logic is extracted out of `apps/semantic_analysis/main.py` into an import-light module that both production and an offline replay harness share, so they cannot diverge. The harness is validated against per-window cosines already recorded in the handoff before any corpus work begins. Corpus scripts come from blinded fresh-context subagents (three drift categories) and from verbatim excerpts of AnnoMI (two on-topic categories). Everything then goes through Polly/Transcribe/WER injection, and the calibration runs a grid search with three fold families against four pre-registered rejection criteria.

**Tech Stack:** Python 3.14, pytest, sentence-transformers (`all-MiniLM-L6-v2`), AWS Polly + Transcribe, boto3.

**Spec:** `docs/superpowers/specs/2026-09-27-analogy-corpus-design.md`

## Global Constraints

- **No mechanism code ships.** `classify()` keeps its current behaviour, `thr = RHO·R` does not enter production, the 1/2/3 escalation ladder stands, `severe` keeps its payout veto.
- `apps/semantic_analysis/windowing.py` is the **only** production file this plan creates; `main.py` is modified to import from it with **zero behaviour change**. `warning_engine`, `backend`, escrow and `eval_dataset/tools/window_chunker.py` are untouched.
- Production constants, copied verbatim: `UPPER = 0.36`, `LOWER = 0.14`, `MIN_CONTENT_TOKENS = 3`, `WINDOW_DURATION_THRESHOLD = 25.0`. Topic text is `f"{topic}. {scope}"`. Model is `all-MiniLM-L6-v2`.
- `windowing.py` must import **no** `sentence_transformers`, `torch`, `fastapi` or `httpx` at module scope — the harness imports it from outside the service venv.
- Cross-directory imports follow the repo's existing convention: `importlib.util.spec_from_file_location`, as in `apps/semantic_analysis/tests/conftest.py:16`.
- Corpus: 25 sessions, 5 per category, 25 **distinct** topics, teacher-word target 1,000–1,400 each. The 15 blinded scripts draw topics from `eval_dataset/topics/topic_pool.md`; the 10 AnnoMI scripts carry AnnoMI's **own declared topic** for the source conversation — a third-party label, which is the point (spec §8). Spec §5.4 says "25 distinct topics from the 40-topic pool"; that is accurate only for the blinded 15, and this plan holds the weaker, correct invariant: 25 distinct topics corpus-wide.
- Real-data scripts are **excerpt and splice only**. No word is ever rewritten. Every dialogue line must be a verbatim substring of its cited source.
- Third-party raw data is **never committed**: `raw/` gitignored, fetched by a `build_dataset.py` against sha256-pinned URLs, per `eval_dataset/curated/qatd/`.
- `RHO` grid: 0.20–0.80 at 0.01. Control on every table: flat `LOWER = 0.14`.

## Review Focus

1. **A script whose teacher never speaks** (all turns are the learner, or `TEACHER:` names a speaker absent from the dialogue) — replay must return zero windows and the caller must not divide by zero when computing rates. Test in Task 3.
2. **A `DIGRESSION` range naming a nonexistent turn or a learner turn** — the parser must reject it loudly rather than silently dropping it, or gold labels become quietly wrong. Test in Task 2.
3. **A session where no window ever reaches `thr`**, so `R` never rises off its `UPPER` floor — this is exactly the `depressed_baseline` case the whole project turns on, and the calibration must handle it without treating an unmoved `R` as missing data. Test in Task 13.
4. **WER deletion pushing a window under `MIN_CONTENT_TOKENS`** — a window present at WER-0 is skipped at WER-30, so the two runs have different window counts and comparing them positionally silently misaligns. The robustness table must align by window index and report count mismatches. Test in Task 13.
5. **A provenance check run when the source file is missing or empty** — a substring check against absent text must error, never vacuously pass, or unverified scripts enter the corpus labelled as verified. Test in Task 10.

---

### Task 1: Extract pure windowing logic

**Files:**
- Create: `apps/semantic_analysis/windowing.py`
- Modify: `apps/semantic_analysis/main.py` (remove the moved definitions, import them back)
- Test: `apps/semantic_analysis/tests/test_windowing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `clean_text(text: str) -> str`, `classify(similarity: float) -> str`, `topic_text(topic: str, scope: str) -> str`, `window_is_ready(accumulated_seconds: float) -> bool`, `has_enough_content(cleaned: str) -> bool`, and constants `UPPER`, `LOWER`, `MIN_CONTENT_TOKENS`, `WINDOW_DURATION_THRESHOLD`.

- [ ] **Step 1: Write the failing test**

```python
# apps/semantic_analysis/tests/test_windowing.py
import subprocess, sys
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent.parent

def test_windowing_imports_without_heavy_dependencies():
    # The harness imports this module outside the service venv.
    code = (
        "import sys; sys.path.insert(0, %r); import windowing; "
        "assert 'sentence_transformers' not in sys.modules; "
        "assert 'torch' not in sys.modules; assert 'fastapi' not in sys.modules"
        % str(APP_DIR)
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

def test_classify_boundaries():
    import windowing as w
    assert w.classify(0.36) == "correct"
    assert w.classify(0.359) == "weakly_correct"
    assert w.classify(0.14) == "weakly_correct"
    assert w.classify(0.139) == "incorrect"

def test_topic_text_matches_production_format():
    import windowing as w
    assert w.topic_text("Node.js basics", "event loop") == "Node.js basics. event loop"
    assert w.topic_text("Node.js basics", "") == "Node.js basics. "

def test_window_is_ready_at_threshold():
    import windowing as w
    assert w.window_is_ready(25.0) is True
    assert w.window_is_ready(24.9) is False

def test_has_enough_content_at_min_tokens():
    import windowing as w
    assert w.has_enough_content("one two three") is True
    assert w.has_enough_content("one two") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd apps/semantic_analysis && ./venv/bin/pytest tests/test_windowing.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'windowing'`

- [ ] **Step 3: Create `apps/semantic_analysis/windowing.py`**

Move verbatim from `main.py`: `UPPER` (55), `LOWER` (56), `FILLER_WORDS` (65), `_FILLER_PHRASE_RE` (76), `_PUNCTUATION` (83), `MIN_CONTENT_TOKENS` (89), `WINDOW_DURATION_THRESHOLD` (92), `clean_text`, `_tidy`, `classify`. Add the three new predicates, each a one-line expression over the constants. Only stdlib `re` is imported.

- [ ] **Step 4: Rewrite those definitions in `main.py` as an import**

Replace the moved blocks with `from windowing import (...)`, importing every moved name so `from main import clean_text` at `ground_truth/generate_synthetic.py:20` keeps working. Replace the local `topic_text = f"{req.topic}. {req.scope}"` at `main.py:441` and the equivalent at `:466` with the `topic_text(...)` call, and the `>= WINDOW_DURATION_THRESHOLD` test at `:524` and the `< MIN_CONTENT_TOKENS` test at `:396` with the predicates. No other line changes.

- [ ] **Step 5: Run the new and the pre-existing tests**

Run: `cd apps/semantic_analysis && ./venv/bin/pytest tests/ -v`
Expected: PASS, including `test_clean_text.py` unchanged — it pins the cleaning behaviour the current thresholds were fitted on, so it is the regression guard for this move.

- [ ] **Step 6: Commit**

```bash
git add apps/semantic_analysis/windowing.py apps/semantic_analysis/main.py apps/semantic_analysis/tests/test_windowing.py
git commit -m "refactor: extract pure windowing logic so the harness can share it"
```

---

### Task 2: Extend the script parser

**Files:**
- Modify: `eval_dataset/tools/script_parser.py`
- Test: `eval_dataset/tools/tests/test_script_parser.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `VALID_CATEGORIES` gains `depressed_baseline`, `abstract_on_topic`, `interleaved_abuse`. `Script` gains `scope: str = ""`, `author: str = ""`, `digression_turns: frozenset[int] = frozenset()`, `source: str | None = None`. Turn numbering for `DIGRESSION` is **1-indexed over all dialogue turns**, inclusive ranges, comma-separated: `DIGRESSION: 5-9, 14`.

- [ ] **Step 1: Write the failing tests**

```python
def test_parses_new_header_fields(tmp_path):
    s = _write(tmp_path, category="interleaved_abuse", extra=
        "SCOPE: event loop, callbacks\nAUTHOR: agent:opus-5\nDIGRESSION: 2-3, 5\n")
    assert s.scope == "event loop, callbacks"
    assert s.author == "agent:opus-5"
    assert s.digression_turns == frozenset({2, 3, 5})

def test_defaults_keep_existing_scripts_parsing():
    s = parse_script("eval_dataset/scripts/person_C/sess_C05.txt")
    assert s.scope == "" and s.author == "" and s.digression_turns == frozenset()

def test_new_categories_accepted(tmp_path):
    for c in ("depressed_baseline", "abstract_on_topic", "interleaved_abuse"):
        assert parse_script(_path(tmp_path, category=c)).category == c

def test_digression_turn_out_of_range_rejected(tmp_path):
    # 4 dialogue turns; naming turn 9 is an authoring error, not an empty set.
    with pytest.raises(ValueError, match="turn 9"):
        parse_script(_path(tmp_path, category="adversarial", extra="DIGRESSION: 9\n"))

def test_digression_naming_a_learner_turn_rejected(tmp_path):
    # Only teacher speech reaches a window, so a learner digression is meaningless.
    with pytest.raises(ValueError, match="learner"):
        parse_script(_path(tmp_path, category="adversarial", extra="DIGRESSION: 2\n"))

def test_digression_required_for_drift_categories(tmp_path):
    with pytest.raises(ValueError, match="requires DIGRESSION"):
        parse_script(_path(tmp_path, category="gradual_drift"))

def test_digression_forbidden_for_on_topic_categories(tmp_path):
    with pytest.raises(ValueError, match="must not declare DIGRESSION"):
        parse_script(_path(tmp_path, category="abstract_on_topic", extra="DIGRESSION: 1\n"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest eval_dataset/tools/tests/test_script_parser.py -v`
Expected: FAIL on the new tests, PASS on the existing ones.

- [ ] **Step 3: Implement the parse and validation**

Add `_parse_digression(value: str, turns: list[Turn], teacher: str) -> frozenset[int]` to `script_parser.py`. It expands inclusive ranges, rejects an index outside `1..len(turns)`, and rejects any index whose turn speaker is not `teacher`. Category-level validation: the three drift categories require a non-empty set; `clean`, `silence`, `depressed_baseline` and `abstract_on_topic` require an empty one. `code_switch` is left unconstrained — existing scripts use it and this plan does not re-label them.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest eval_dataset/tools/tests/test_script_parser.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/script_parser.py eval_dataset/tools/tests/test_script_parser.py
git commit -m "feat: add corpus categories and structural-label header fields"
```

---

### Task 3: The replay windower

**Files:**
- Create: `eval_dataset/tools/replay.py`
- Test: `eval_dataset/tools/tests/test_replay.py`

**Interfaces:**
- Consumes: `Script` from Task 2; `clean_text`, `window_is_ready`, `has_enough_content` from Task 1.
- Produces:

```python
@dataclass(frozen=True)
class TurnContribution:
    turn_index: int      # 1-indexed, matching DIGRESSION
    seconds: float

@dataclass(frozen=True)
class ReplayWindow:
    index: int           # production window_counter value
    text: str            # cleaned
    raw_text: str
    teacher_seconds_start: float
    teacher_seconds_end: float
    contributions: tuple[TurnContribution, ...]

def synthetic_durations(script: Script) -> list[float]     # len(text.split()) * 0.4 per turn
def replay(script: Script, durations: list[float]) -> list[ReplayWindow]
```

- [ ] **Step 1: Write the failing tests**

```python
def test_only_teacher_turns_enter_windows():
    # learner text must appear in no window
def test_window_closes_at_25_teacher_seconds():
    # 26s of teacher speech in one turn -> exactly one closed window
def test_learner_turns_do_not_advance_the_buffer():
    # teacher 20s, learner 60s, teacher 10s -> one window spanning both teacher turns
def test_trailing_partial_window_is_flushed():
    # 30s total -> window 1 at 25s, window 2 with the 5s remainder
def test_skipped_window_still_consumes_an_index():
    # A window under MIN_CONTENT_TOKENS is not emitted, but the next emitted
    # window's .index is 2, not 1 — main.py:385 increments before the check.
def test_script_with_no_teacher_turns_returns_no_windows():
    assert replay(script_with_only_learner_turns, durations) == []
def test_synthetic_durations_match_the_handoff_rate():
    assert synthetic_durations(five_word_turn_script) == [2.0]
def test_contributions_record_seconds_per_turn():
    # a window fed by turns 1 (15s) and 3 (10s) carries exactly those two
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest eval_dataset/tools/tests/test_replay.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'eval_dataset.tools.replay'`

- [ ] **Step 3: Implement `replay.py`**

Import `windowing` by `importlib.util.spec_from_file_location` against `apps/semantic_analysis/windowing.py`. Walk turns in order, accumulating only those whose speaker equals `script.teacher`; close on `window_is_ready`; increment the counter **before** the `has_enough_content` check and emit nothing when it fails; flush any remainder after the last turn, subject to the same check. `raw_text` is `" ".join` of the raw segment texts; `text` is `clean_text` of that.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest eval_dataset/tools/tests/test_replay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/replay.py eval_dataset/tools/tests/test_replay.py
git commit -m "feat: production-faithful offline windower"
```

---

### Task 4: Window embedding and scoring

**Files:**
- Create: `eval_dataset/tools/embed_windows.py`
- Test: `eval_dataset/tools/tests/test_embed_windows.py`

**Interfaces:**
- Consumes: `ReplayWindow` from Task 3; `classify`, `topic_text` from Task 1.
- Produces:

```python
@dataclass(frozen=True)
class WindowScore:
    index: int
    similarity: float
    classification: str

def score_windows(windows, topic: str, scope: str, model=None) -> list[WindowScore]
```

`model=None` loads `SentenceTransformer("all-MiniLM-L6-v2")` once and caches it at module level.

- [ ] **Step 1: Write the failing tests**

```python
def test_identical_text_scores_near_one():
    # window text == topic text -> similarity > 0.99
def test_classification_uses_production_thresholds():
    # a scored window's .classification equals classify(.similarity)
def test_scope_is_included_in_the_topic_embedding():
    # same windows, different scope -> different similarities
def test_empty_window_list_returns_empty():
    assert score_windows([], "t", "s") == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest eval_dataset/tools/tests/test_embed_windows.py -v`
Expected: FAIL, module not found

- [ ] **Step 3: Implement `embed_windows.py`**

Encode `topic_text(topic, scope)` once, encode each window's cleaned `text`, cosine via `sentence_transformers.util.cos_sim`, classify with the shared `classify`.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest eval_dataset/tools/tests/test_embed_windows.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/embed_windows.py eval_dataset/tools/tests/test_embed_windows.py
git commit -m "feat: score replayed windows against the contract topic"
```

---

### Task 5: The conformance gate — HARD STOP

**Files:**
- Test: `eval_dataset/tools/tests/test_conformance_oracle.py`

**Interfaces:**
- Consumes: `replay`, `synthetic_durations`, `score_windows`.
- Produces: nothing. This task is a gate, not a component.

Spec §4.3. **If this task fails, stop and report — do not proceed to Task 6.** A failure means the handoff's recorded numbers are not reproducible, which is a more important finding than anything downstream.

- [ ] **Step 1: Write the test**

```python
TOLERANCE = 0.005

def test_sess_B08_analogy_window_cosines():
    scores = _run("eval_dataset/scripts/person_B/sess_B08.txt", scope="")
    expected = [0.179, 0.130, 0.197, 0.008, 0.029]
    actual = _analogy_window_similarities(scores)
    assert actual == pytest.approx(expected, abs=TOLERANCE)

def test_sess_B08_incorrect_count_and_max_consecutive():
    scores = _run("eval_dataset/scripts/person_B/sess_B08.txt", scope="")
    assert sum(s.classification == "incorrect" for s in scores) == 3
    assert _max_consecutive_incorrect(scores) == 2   # -> STRONG by warning_engine/main.py:178

def test_sess_C05_analogy_window_cosines():
    scores = _run("eval_dataset/scripts/person_C/sess_C05.txt", scope="")
    assert _analogy_window_similarities(scores) == pytest.approx([0.172, 0.155], abs=TOLERANCE)

def test_sess_C05_has_no_incorrect_windows():
    scores = _run("eval_dataset/scripts/person_C/sess_C05.txt", scope="")
    assert sum(s.classification == "incorrect" for s in scores) == 0
    assert _max_consecutive_incorrect(scores) == 0

@pytest.mark.parametrize("script,expected_mean", [
    ("_experimental/case_B2_late.txt", 0.059),
    ("_experimental/adv3_long.txt", 0.008),
])
def test_experimental_span_means(script, expected_mean):
    assert _digression_mean(script) == pytest.approx(expected_mean, abs=0.02)
```

`_max_consecutive_incorrect` is local arithmetic over the classification sequence; the harness runs no `warning_engine`.

- [ ] **Step 2: Run the gate**

Run: `python -m pytest eval_dataset/tools/tests/test_conformance_oracle.py -v`
Expected: PASS. If the analogy cosines land within ±0.001, tighten `TOLERANCE` to 0.001 and re-run.

- [ ] **Step 3: On failure, stop**

Record actual vs expected per window in `apps/semantic_analysis/ground_truth/rho_calibration_findings.md` under "Conformance gate failure", commit, and report. Do not start Task 6.

- [ ] **Step 4: Commit**

```bash
git add eval_dataset/tools/tests/test_conformance_oracle.py
git commit -m "test: pin the harness to the handoff's recorded per-window cosines"
```

---

### Task 6: Structural gold labels

**Files:**
- Create: `eval_dataset/tools/gold_labels.py`
- Test: `eval_dataset/tools/tests/test_gold_labels.py`

**Interfaces:**
- Consumes: `ReplayWindow`, `TurnContribution`, `Script`.
- Produces: `gold_label(window: ReplayWindow, script: Script) -> str | None` returning `"on_topic"`, `"off_topic"`, or `None` for excluded. Off-topic share is `sum(seconds of contributions whose turn_index is in script.digression_turns) / total seconds`; `None` when that share is in the closed band `[0.4, 0.6]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_wholly_on_topic_window():          # share 0.0 -> "on_topic"
def test_wholly_off_topic_window():         # share 1.0 -> "off_topic"
def test_majority_off_topic_window():       # share 0.7 -> "off_topic"
def test_majority_on_topic_window():        # share 0.25 -> "on_topic"
def test_boundary_window_is_excluded():     # share 0.5 -> None
def test_band_edges_are_excluded():         # shares 0.4 and 0.6 -> None
def test_category_with_no_digressions_is_all_on_topic():
    # abstract_on_topic: every window labels "on_topic", none excluded
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest eval_dataset/tools/tests/test_gold_labels.py -v`
Expected: FAIL, module not found

- [ ] **Step 3: Implement `gold_labels.py`**

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest eval_dataset/tools/tests/test_gold_labels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/gold_labels.py eval_dataset/tools/tests/test_gold_labels.py
git commit -m "feat: derive per-window gold labels from declared digressions"
```

---

### Task 7: The blinding prompt template

**Files:**
- Create: `eval_dataset/tools/authoring/prompt_template.md`
- Create: `eval_dataset/tools/authoring/__init__.py`
- Test: `eval_dataset/tools/tests/test_prompt_template.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a template with `{topic}`, `{scope}`, `{category_description}`, `{teacher_word_target}`, `{teacher_letter}`, `{learner_letter}` placeholders, and `CATEGORY_DESCRIPTIONS: dict[str, str]` in `eval_dataset/tools/authoring/__init__.py` — one purely behavioural description per category, naming no measurement.

Spec §5.3: the template is committed so the blinding is auditable rather than asserted.

- [ ] **Step 1: Write the failing test**

```python
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
    # every placeholder above appears exactly once
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest eval_dataset/tools/tests/test_prompt_template.py -v`
Expected: FAIL, file not found

- [ ] **Step 3: Write the template and the descriptions**

The template instructs: write a two-person teaching dialogue in the script format, hit the teacher-word target, and declare `DIGRESSION` turn ranges for any stretch where the teacher is not teaching the stated topic. Describe the *word budget* in words, never in windows or seconds. Each `CATEGORY_DESCRIPTIONS` entry is behavioural: e.g. `gradual_drift` = "the teacher starts on the topic and progressively stops teaching it, never returning". `depressed_baseline` and `abstract_on_topic` must additionally instruct: never leave the topic at all.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest eval_dataset/tools/tests/test_prompt_template.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/authoring/ eval_dataset/tools/tests/test_prompt_template.py
git commit -m "feat: committed, auditable blinding prompt for corpus authors"
```

---

### Task 8: Author the 15 blinded drift scripts

**Files:**
- Create: `eval_dataset/scripts/calibration/gradual_drift/sess_CAL{01..05}.txt`
- Create: `eval_dataset/scripts/calibration/adversarial/sess_CAL{06..10}.txt`
- Create: `eval_dataset/scripts/calibration/interleaved_abuse/sess_CAL{11..15}.txt`
- Test: `eval_dataset/tools/tests/test_corpus_integrity.py`

**Interfaces:**
- Consumes: the Task 7 template; `parse_script`; `replay`; `synthetic_durations`.
- Produces: 15 scripts. `AUTHOR` cycles `agent:opus-5`, `agent:sonnet-5`, `agent:haiku-4.5` so each model writes 5, at least one per category.

- [ ] **Step 1: Write the failing corpus-integrity test**

```python
def test_every_calibration_script_parses():
def test_five_sessions_per_category():
def test_all_topics_distinct_and_from_the_pool():
def test_teacher_word_count_in_range():          # 1000..1400 inclusive
def test_each_script_yields_at_least_ten_windows():
    # replay at synthetic durations -> len(windows) >= 10
def test_authors_are_rotated():
    # each of the three agent authors appears >= 1 time in every category
def test_drift_scripts_declare_digressions():
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest eval_dataset/tools/tests/test_corpus_integrity.py -v`
Expected: FAIL — no scripts exist yet.

- [ ] **Step 3: Dispatch one fresh-context subagent per script**

15 dispatches. Each receives **only** the rendered template for its topic and category, and returns the script body. Rotate the subagent model per the `AUTHOR` assignment. Do not pass the spec, the handoff, this plan, or any other script into a dispatch. Write each returned body to its path, prepending the header with `TOPIC`, `SCOPE`, `TEACHER`, `LEARNER`, `CATEGORY`, `AUTHOR`, and the `DIGRESSION` line the author declared.

- [ ] **Step 4: Run the integrity test; re-dispatch any script that fails**

Run: `python -m pytest eval_dataset/tools/tests/test_corpus_integrity.py -v`
Expected: PASS. A script short of 10 windows or outside the word range is re-dispatched with the same blinded template, not hand-edited — hand-editing would make this session a co-author.

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/scripts/calibration/ eval_dataset/tools/tests/test_corpus_integrity.py
git commit -m "feat: 15 blinded drift-category corpus scripts"
```

---

### Task 9: Vendor the AnnoMI source

**Files:**
- Create: `eval_dataset/curated/annomi/build_dataset.py`
- Create: `eval_dataset/curated/annomi/source_manifest.json`
- Create: `eval_dataset/curated/annomi/.gitignore`
- Create: `eval_dataset/curated/annomi/SOURCE_README.md`
- Test: `eval_dataset/tools/tests/test_annomi_build.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `build_dataset.py` fetching each manifest entry to `raw/` and verifying sha256; `load_conversations() -> list[Conversation]` with `Conversation(id, topic, utterances)` and `Utterance(speaker_role, text)` where `speaker_role` is `"therapist"` or `"client"`.

Mirror `eval_dataset/curated/qatd/` exactly. `.gitignore` contains `raw/`, `*.csv`, `__pycache__/`.

- [ ] **Step 1: Write the failing tests**

```python
def test_gitignore_excludes_raw_data():
def test_manifest_pins_revision_and_sha256_per_file():
def test_build_rejects_a_file_whose_sha256_mismatches(tmp_path):
def test_load_conversations_yields_topic_and_two_roles():
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest eval_dataset/tools/tests/test_annomi_build.py -v`
Expected: FAIL

- [ ] **Step 3: Write the manifest, fetcher and loader**

Pin the `uccollab/AnnoMI` commit and per-file sha256 at fetch time. `SOURCE_README.md` records, verbatim and prominently: **the upstream repository ships no LICENSE file**, the CC0 claim is third-party, transcripts derive from third-party video, and contacting the authors is an open action item (spec §8).

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest eval_dataset/tools/tests/test_annomi_build.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/curated/annomi/ eval_dataset/tools/tests/test_annomi_build.py
git commit -m "feat: vendor AnnoMI by pinned manifest, raw data ungitted"
```

---

### Task 10: Provenance verification

**Files:**
- Create: `eval_dataset/tools/verify_provenance.py`
- Test: `eval_dataset/tools/tests/test_verify_provenance.py`

**Interfaces:**
- Consumes: `parse_script`; `load_conversations` from Task 9.
- Produces:

```python
@dataclass(frozen=True)
class Provenance:
    dataset: str
    records: tuple[str, ...]        # source conversation ids, in splice order
    spans: tuple[tuple[str, int, int], ...]   # (record_id, char_start, char_end)
    seams: tuple[int, ...]          # turn indices where a splice occurs

def verify(script_path: str, provenance_path: str) -> None   # raises ProvenanceError
```

- [ ] **Step 1: Write the failing tests**

```python
def test_verbatim_script_passes():
def test_a_single_altered_word_fails():
def test_missing_source_file_raises_not_passes():
    # vacuous pass here would admit unverified scripts labelled as verified
def test_empty_source_text_raises():
def test_seams_must_fall_on_turn_boundaries():
def test_every_dialogue_line_must_be_covered_by_a_span():
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest eval_dataset/tools/tests/test_verify_provenance.py -v`
Expected: FAIL, module not found

- [ ] **Step 3: Implement `verify_provenance.py`**

For each dialogue line, assert it is a substring of the cited record's text. Raise `ProvenanceError` — never return cleanly — when a source record is absent, empty, or uncited.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest eval_dataset/tools/tests/test_verify_provenance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/tools/verify_provenance.py eval_dataset/tools/tests/test_verify_provenance.py
git commit -m "feat: enforce verbatim provenance for excerpted corpus scripts"
```

---

### Task 11: Extract the 10 AnnoMI scripts

**Files:**
- Create: `eval_dataset/scripts/calibration/abstract_on_topic/sess_CAL{16..20}.txt`
- Create: `eval_dataset/scripts/calibration/depressed_baseline/sess_CAL{21..25}.txt`
- Create: `eval_dataset/provenance/sess_CAL{16..25}.provenance.json`
- Modify: `eval_dataset/tools/tests/test_corpus_integrity.py`

**Interfaces:**
- Consumes: `load_conversations`, `verify`, `parse_script`, `replay`.
- Produces: 10 scripts with `AUTHOR: real:annomi` and a `SOURCE:` pointing at the sidecar.

- [ ] **Step 1: Extend the integrity test**

```python
def test_real_data_scripts_pass_provenance_verification():
    # verify() raises for none of sess_CAL16..25
def test_real_data_scripts_declare_no_digressions():
def test_real_data_topics_are_annomi_declared_and_distinct():
    # AnnoMI topics are third-party labels and are NOT in topic_pool.md
def test_corpus_has_twenty_five_distinct_topics():
def test_corpus_is_twenty_five_sessions_across_five_categories():
def test_author_folds_have_at_least_four_values():
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest eval_dataset/tools/tests/test_corpus_integrity.py -v`
Expected: FAIL

- [ ] **Step 3: Run the cleaner agent**

One dispatch per script. It selects spans from AnnoMI conversations, maps `therapist`->teacher and `client`->learner, and splices only at speaker-turn boundaries to reach the teacher-word target. **It selects; it never writes words.** Each dispatch returns the script plus its provenance sidecar.

`depressed_baseline` scripts must come from a single conversation with no splice (seams empty) — the category is defined as a *fully clean* session, and a splice would introduce a topic change. `abstract_on_topic` may splice across conversations sharing a topic label.

- [ ] **Step 4: Run integrity and provenance over the whole corpus**

Run: `python -m pytest eval_dataset/tools/tests/ -v`
Expected: PASS. Any script failing `verify` is re-extracted, never patched by hand.

- [ ] **Step 5: Commit**

```bash
git add eval_dataset/scripts/calibration/ eval_dataset/provenance/ eval_dataset/tools/tests/test_corpus_integrity.py
git commit -m "feat: 10 verbatim AnnoMI-derived on-topic corpus scripts"
```

---

### Task 12: The STT sweep

**Files:**
- Create: `eval_dataset/tools/run_stt_sweep.py`
- Modify: `eval_dataset/README.md`
- Test: `eval_dataset/tools/tests/test_run_stt_sweep.py`

**Interfaces:**
- Consumes: `synthesize_script`, `transcribe`, `inject_wer`.
- Produces:

```python
def check_polly_access(polly_client) -> None        # raises on AccessDenied
def timed_durations(script: Script, transcript: dict, timing: list[dict]) -> list[float]
def sweep(session_ids: list[str], seed: int = 0) -> None
```

`timed_durations` joins transcript words to turns by timestamp containment against `{session_id}_timing.json` (spec §6.1), since `transcribe.py:16` emits no speaker labels.

- [ ] **Step 1: Write the failing tests**

```python
def test_check_polly_access_raises_on_access_denied():   # mocked client
def test_timed_durations_assigns_each_word_to_its_turn():
def test_timed_durations_ignores_words_outside_every_turn():   # inter-turn silence
def test_wer_variants_are_deterministic_for_a_fixed_seed():
def test_sweep_skips_a_session_whose_wav_already_exists():     # reruns are cheap
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest eval_dataset/tools/tests/test_run_stt_sweep.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `run_stt_sweep.py`**

`check_polly_access` runs first and aborts the whole sweep on failure — `2026-08-11-eval-dataset-design.md` lists the Polly IAM permission as never added, so this is the gate, not a surprise mid-run.

- [ ] **Step 4: Run the tests, then the real sweep over all 25 sessions**

Run: `python -m pytest eval_dataset/tools/tests/test_run_stt_sweep.py -v`
Then: `python -m eval_dataset.tools.run_stt_sweep --all`
Expected: 25 `.wav`, 25 `_timing.json`, 25 `_wer0.json`, 75 WER variants.

- [ ] **Step 5: Commit**

WER-0 transcripts are committed as artifacts — AWS Transcribe is not reproducible across runs (spec §6.4). The 10/20/30 variants are regenerable from the committed WER-0 at the recorded seed, so they are not stored.

```bash
git add eval_dataset/tools/run_stt_sweep.py eval_dataset/tools/tests/test_run_stt_sweep.py \
        eval_dataset/audio/ eval_dataset/transcripts/raw/ eval_dataset/README.md
git commit -m "feat: STT sweep with speaker attribution by turn timing"
```

---

### Task 13: Calibrate `RHO`

**Files:**
- Create: `apps/semantic_analysis/ground_truth/rho_calibration.py`
- Create: `apps/semantic_analysis/ground_truth/rho_calibration_results.json`
- Test: `apps/semantic_analysis/ground_truth/test_rho_calibration.py`

**Interfaces:**
- Consumes: `replay`, `synthetic_durations`, `timed_durations`, `score_windows`, `gold_label`.
- Produces:

```python
def running_thresholds(similarities: list[float], rho: float) -> list[float]
    # R starts at UPPER (0.36); a window updates R only if sim >= rho * R_current;
    # R is capped at 0.80 and never decreases. Returns thr per window.

def evaluate(sessions, rho: float, durations: str = "synthetic") -> ErrorMix
    # ErrorMix(caught, missed, false_digressions, on_topic_total, digression_total)

def grid_search(sessions, folds: str) -> GridResult
    # folds in {"session", "topic", "author"}; returns per-fold argmax and the full grid
```

- [ ] **Step 1: Write the failing tests**

```python
def test_R_starts_at_upper():
    assert running_thresholds([0.05], rho=0.45)[0] == pytest.approx(0.45 * 0.36)

def test_below_threshold_window_does_not_raise_R():
    # a 0.9 window that sits below thr cannot become the ceiling it is judged by
def test_R_never_decreases_within_a_session():
def test_R_is_capped_at_080():
def test_R_stays_at_the_floor_when_no_window_ever_reaches_thr():
    # the depressed_baseline case: thr constant at 0.162 for the whole session,
    # and evaluate() must score it, not treat it as missing data
def test_flat_lower_control_is_reported_alongside_every_rho():
def test_error_mix_counts_excluded_windows_in_neither_class():
def test_wer_comparison_aligns_by_window_index_not_position():
    # WER-30 may skip a window that WER-0 emitted; a positional zip misaligns
def test_wer_comparison_reports_window_count_mismatches():
def test_grid_search_reports_argmax_per_fold():
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd apps/semantic_analysis && ./venv/bin/pytest ground_truth/test_rho_calibration.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `rho_calibration.py`**

Grid 0.20–0.80 at 0.01. Three fold families: leave-one-session-out (25), leave-one-topic-out (25), leave-one-author-out (4). Control on every table: flat `LOWER = 0.14`. Write the full grid, per-fold argmaxes and the WER 0/10/20/30 robustness table to `rho_calibration_results.json`, matching the shape of the existing `threshold_fixture_results.json`.

- [ ] **Step 4: Run the tests, then the real calibration**

Run: `cd apps/semantic_analysis && ./venv/bin/pytest ground_truth/test_rho_calibration.py -v`
Then: `./venv/bin/python -m ground_truth.rho_calibration --all-wer`
Expected: PASS, and `rho_calibration_results.json` written.

- [ ] **Step 5: Commit**

```bash
git add apps/semantic_analysis/ground_truth/rho_calibration.py \
        apps/semantic_analysis/ground_truth/rho_calibration_results.json \
        apps/semantic_analysis/ground_truth/test_rho_calibration.py
git commit -m "feat: RHO grid search with session, topic and author folds"
```

---

### Task 14: Findings and verdict

**Files:**
- Create: `apps/semantic_analysis/ground_truth/rho_calibration_findings.md`
- Modify: `apps/semantic_analysis/ground_truth/design_decisions.md`

**Interfaces:**
- Consumes: `rho_calibration_results.json`.
- Produces: a written verdict on `RHO`.

- [ ] **Step 1: Write the findings document**

Structure it as `threshold_experiment_findings.md` is structured. Required contents:

- The D2-shaped error-mix table: `RHO` | digressions caught | **false digressions** | on-topic windows affected, with the flat-`LOWER` control row.
- Per-fold argmax for all three fold families, and whether the argmax was *identical* across folds — D1's bar.
- The WER 0/10/20/30 robustness table and how far the argmax moved.
- An explicit **verdict against each of spec §7.5's four rejection criteria**, one subsection each, stating pass or fail.
- The count of windows excluded as boundary cases (Task 6), per category.
- A limits section reproducing spec §8 in full: structural labels with no kappa; synthetic prosody throughout; AnnoMI is counselling, not teaching, and carries no licence grant; three of five categories LLM-authored; 25 sessions is small.

- [ ] **Step 2: Record the decision**

Add a `D3 — RHO` section to `design_decisions.md` in the style of D1/D2: the decision, why that value (or why rejected), the held-out result, and limits stated plainly.

- [ ] **Step 3: State the consequence for handoff §6 step 1**

One paragraph: whether `thr = RHO·R` is cleared to ship, blocked, or rejected. **A rejection is a successful completion of this plan** (spec §11) — it closes §6 step 1 and saves building three services on a constant that does not hold.

- [ ] **Step 4: Commit**

```bash
git add apps/semantic_analysis/ground_truth/rho_calibration_findings.md \
        apps/semantic_analysis/ground_truth/design_decisions.md
git commit -m "docs: RHO calibration findings and verdict"
```
