# Meaning-Reversal Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect meaning-reversed commitments (e.g. "I will not pay X") that pass the existing cosine topic-gate because they're topically similar, force those windows to `classification = "incorrect"`, and thread a `meaning_reversal_detected` audit flag through `semantic_analysis` → `warning_engine` → `backend` so a reversal-forced `incorrect` is distinguishable from a genuine low-cosine one.

**Architecture:** A new pure-function module in `semantic_analysis` detects negation-cue-near-commitment-phrase patterns per sentence; `process_window` calls it right after `classify()` and overrides the classification on a hit. The existing `consecutive_incorrect` escalation ladder in `warning_engine` is untouched — the override reuses that path as-is. A boolean flag rides alongside `classification` through every existing payload/schema/table in the chain, purely for audit visibility.

**Tech Stack:** Python/FastAPI (`semantic_analysis`, `warning_engine`, `backend`), SQLAlchemy async ORM, pytest/pytest-asyncio, `sentence-transformers` (already a `semantic_analysis` dependency, used only by the existing integration-test fixture, not by this feature's own logic).

**Spec:** `docs/superpowers/specs/2026-09-29-meaning-reversal-check-design.md`

## Global Constraints

- `NEGATION_PROXIMITY_TOKENS = 4` — a negation cue and a commitment phrase are linked if they occur within this many tokens of each other in the same sentence.
- `COMMITMENT_PHRASES` = `{"pay", "give", "provide", "deliver", "send", "transfer", "return", "refund", "owe", "accept", "honor", "complete", "fulfill", "commit to", "promise", "agree to", "guarantee"}`.
- `NEGATION_CUES` = `{"not", "never", "no longer", "won't", "can't", "cannot", "wouldn't", "shouldn't", "couldn't", "refuse to", "refused to", "unable to"}`.
- No changes to `UPPER`/`LOWER` thresholds, `classify()`'s cosine logic, or `warning_engine`'s escalation ladder (`consecutive_incorrect`, severity tiers).
- `meaning_reversal_detected` is a passthrough field everywhere except `semantic_analysis`, where it is computed — it must never change `warning_engine`'s severity decision.
- Double-negation sentences are a **documented known limitation**, not a bug to fix in this plan — the test for that case asserts the current (imperfect) behavior so a future change to it is a visible, deliberate decision, not a silent regression.

## Review Focus

- **A commitment phrase appears with no negation nearby** (the common case — most of a session's speech): must never be flagged. Tested directly on the pure function.
- **A negation appears far from any commitment phrase in the same sentence** (beyond `NEGATION_PROXIMITY_TOKENS`): must not be flagged — this is the false-positive case the design explicitly calls out.
- **A negation and commitment phrase are in different sentences** of the same window (e.g., "I won't be free tomorrow. I will pay you the full amount."): must not be flagged — proximity is scoped per-sentence, not per-window.
- **A window already classified `incorrect` by cosine alone** also contains a reversal: the override must be idempotent (still `"incorrect"`, `meaning_reversal_detected = True`), never raise or double-count in `consecutive_incorrect`.
- **`meaning_reversal_detected` defaults to `False`/absent-safe** at every hop (a warning_engine or backend instance running old code against a new payload, or vice versa) — every schema field must have a default so a missing field never 500s the request.

---

## Task 1: Reversal-detection pure function (`semantic_analysis`)

**Files:**
- Create: `apps/semantic_analysis/reversal_detection.py`
- Test: `apps/semantic_analysis/tests/test_reversal_detection.py`

**Interfaces:**
- Produces: `COMMITMENT_PHRASES: set[str]`, `NEGATION_CUES: set[str]`, `NEGATION_PROXIMITY_TOKENS: int` (module-level constants, exact values from Global Constraints).
- Produces: `detect_meaning_reversal(cleaned_text: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# apps/semantic_analysis/tests/test_reversal_detection.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/semantic_analysis && venv/bin/pytest tests/test_reversal_detection.py -v`
Expected: FAIL — `reversal_detection` module does not exist.

- [ ] **Step 3: Implement `apps/semantic_analysis/reversal_detection.py`**

```python
import re

COMMITMENT_PHRASES = {
    "pay", "give", "provide", "deliver", "send", "transfer", "return",
    "refund", "owe", "accept", "honor", "complete", "fulfill",
    "commit to", "promise", "agree to", "guarantee",
}

NEGATION_CUES = {
    "not", "never", "no longer", "won't", "can't", "cannot", "wouldn't",
    "shouldn't", "couldn't", "refuse to", "refused to", "unable to",
}

NEGATION_PROXIMITY_TOKENS = 4

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_PUNCTUATION = ".,!?;:\"'()[]{}…"


def _tokenize(sentence: str) -> list[str]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/semantic_analysis && venv/bin/pytest tests/test_reversal_detection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/semantic_analysis/reversal_detection.py apps/semantic_analysis/tests/test_reversal_detection.py
git commit -m "feat: add proximity-based meaning-reversal detection"
```

---

## Task 2: Integrate into `process_window` (`semantic_analysis/main.py`)

**Files:**
- Modify: `apps/semantic_analysis/main.py`
- Test: `apps/semantic_analysis/tests/test_reversal_integration.py`

**Interfaces:**
- Consumes: `detect_meaning_reversal(cleaned_text: str) -> bool` from Task 1.
- Modifies: `process_window` (currently `apps/semantic_analysis/main.py:323-359`) — classification override.
- Modifies: `post_window_result(barter_id, window_id, classification, similarity, ts_start, ts_end, text_preview, meaning_reversal_detected)` — new parameter, forwarded in the POST payload as `"meaning_reversal_detected"`.

- [ ] **Step 1: Write the failing test**

Reuse the `semantic_client` fixture already defined in `apps/semantic_analysis/tests/conftest.py` (loads the real app + real MiniLM model, mocks only `http_client`).

```python
# apps/semantic_analysis/tests/test_reversal_integration.py
import pytest


@pytest.mark.asyncio
async def test_reversal_forces_incorrect_and_sets_flag(semantic_client):
    client, mock_http, sa_main = semantic_client

    resp = await client.post("/ingest/segment", json={
        "barter_id": 1,
        "user_id": 1,  # teacher
        "text": "I will not pay you the money.",
        "duration_seconds": 30.0,  # >= WINDOW_DURATION_THRESHOLD, triggers window immediately
        "timestamp_start": 0.0,
        "timestamp_end": 30.0,
    })
    assert resp.status_code == 200

    result_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/window/result")]
    assert len(result_calls) == 1
    payload = result_calls[0].kwargs["json"]
    assert payload["classification"] == "incorrect"
    assert payload["meaning_reversal_detected"] is True


@pytest.mark.asyncio
async def test_no_reversal_leaves_flag_false(semantic_client):
    client, mock_http, sa_main = semantic_client

    resp = await client.post("/ingest/segment", json={
        "barter_id": 1,
        "user_id": 1,
        "text": "I will pay you the money as agreed.",
        "duration_seconds": 30.0,
        "timestamp_start": 0.0,
        "timestamp_end": 30.0,
    })
    assert resp.status_code == 200

    result_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/window/result")]
    assert len(result_calls) == 1
    payload = result_calls[0].kwargs["json"]
    assert payload["meaning_reversal_detected"] is False


@pytest.mark.asyncio
async def test_reversal_override_idempotent_when_already_incorrect_by_cosine(semantic_client):
    # A window that is already "incorrect" on cosine grounds AND contains a
    # detected reversal must stay "incorrect" (not raise, not double-apply)
    # — the override is a no-op assignment, not an increment.
    client, mock_http, sa_main = semantic_client
    sa_main.contracts[1]["topic_embedding"] = sa_main.embed("a completely unrelated topic about gardening")

    resp = await client.post("/ingest/segment", json={
        "barter_id": 1,
        "user_id": 1,
        "text": "I will not pay you the money for the car repair.",
        "duration_seconds": 30.0,
        "timestamp_start": 0.0,
        "timestamp_end": 30.0,
    })
    assert resp.status_code == 200

    result_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/window/result")]
    assert len(result_calls) == 1
    payload = result_calls[0].kwargs["json"]
    assert payload["classification"] == "incorrect"
    assert payload["meaning_reversal_detected"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/semantic_analysis && venv/bin/pytest tests/test_reversal_integration.py -v`
Expected: FAIL — `payload["meaning_reversal_detected"]` KeyError.

- [ ] **Step 3: Implement in `apps/semantic_analysis/main.py`**

Import `detect_meaning_reversal` from `reversal_detection`. In `process_window`, immediately after `classification = classify(similarity)` (line 348):

```python
meaning_reversal_detected = detect_meaning_reversal(cleaned)
if meaning_reversal_detected:
    classification = "incorrect"
```

Pass `meaning_reversal_detected` through to `_window(...)` (extend that debug-print helper's signature if it should log it — optional, not required by any test) and to `post_window_result(...)`. Add the new parameter to `post_window_result`'s signature and its `payload` dict (key `"meaning_reversal_detected"`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/semantic_analysis && venv/bin/pytest tests/test_reversal_integration.py -v`
Expected: PASS

- [ ] **Step 5: Run the full semantic_analysis test suite**

Run: `cd apps/semantic_analysis && venv/bin/pytest -v`
Expected: PASS — no regression in `test_clean_text.py`, `test_windowing.py`, `test_engagement_update_post.py`.

- [ ] **Step 6: Commit**

```bash
git add apps/semantic_analysis/main.py apps/semantic_analysis/tests/test_reversal_integration.py
git commit -m "feat: force incorrect classification on detected meaning reversal"
```

---

## Task 3: Passthrough field (`warning_engine`)

**Files:**
- Modify: `apps/warning_engine/main.py`
- Test: `apps/warning_engine/tests/test_reversal_passthrough.py`

**Interfaces:**
- Consumes: `"meaning_reversal_detected"` key in the JSON POSTed to `/window/result` from Task 2.
- Modifies: `WindowResultRequest` (currently `apps/warning_engine/main.py:59-66`) — add `meaning_reversal_detected: bool = False`.
- Modifies: `run_warning_decision`'s `window_payload` dict (currently lines 201-209) — add `"meaning_reversal_detected": request.meaning_reversal_detected`.

- [ ] **Step 1: Write the failing test**

Reuse the `warning_client` fixture from `apps/warning_engine/tests/conftest.py`.

```python
# apps/warning_engine/tests/test_reversal_passthrough.py
import pytest


@pytest.mark.asyncio
async def test_reversal_flag_forwarded_to_backend(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    resp = await client.post("/window/result", json={
        "barter_id": 1, "window_id": 1, "classification": "incorrect",
        "similarity_score": 0.5, "text_preview": "I will not pay you.",
        "timestamp_start": 0.0, "timestamp_end": 25.0,
        "meaning_reversal_detected": True,
    })
    assert resp.status_code == 200

    result_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/window/result")]
    assert len(result_calls) == 1
    assert result_calls[0].kwargs["json"]["meaning_reversal_detected"] is True


@pytest.mark.asyncio
async def test_reversal_flag_defaults_false_when_omitted(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    resp = await client.post("/window/result", json={
        "barter_id": 1, "window_id": 1, "classification": "correct",
        "similarity_score": 0.5, "text_preview": "on topic",
        "timestamp_start": 0.0, "timestamp_end": 25.0,
    })
    assert resp.status_code == 200

    result_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/window/result")]
    assert result_calls[0].kwargs["json"]["meaning_reversal_detected"] is False


@pytest.mark.asyncio
async def test_reversal_flag_does_not_change_escalation_severity(warning_client):
    # A reversal-forced "incorrect" must escalate exactly like any other
    # incorrect window — no special-casing in the severity ladder.
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    for window_id in (1, 2, 3):
        await client.post("/window/result", json={
            "barter_id": 1, "window_id": window_id, "classification": "incorrect",
            "similarity_score": 0.5, "text_preview": "I will not pay.",
            "timestamp_start": 0.0, "timestamp_end": 25.0,
            "meaning_reversal_detected": True,
        })

    warning_calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/warnings/log")]
    assert len(warning_calls) == 2  # identical to the existing strong+severe test in test_advisory_flag.py
    assert warning_calls[-1].kwargs["json"]["severity"] == "severe"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/warning_engine && venv/bin/pytest tests/test_reversal_passthrough.py -v`
Expected: FAIL — `meaning_reversal_detected` absent from forwarded payload / rejected as unexpected field is not the failure mode here (Pydantic ignores unknown fields by default only if not `extra="forbid"`; the field must exist for it to appear in `window_payload`), so failure is a `KeyError`/assertion mismatch on the forwarded payload.

- [ ] **Step 3: Implement in `apps/warning_engine/main.py`**

Add `meaning_reversal_detected: bool = False` to `WindowResultRequest`. Add `"meaning_reversal_detected": request.meaning_reversal_detected` to the `window_payload` dict in `run_warning_decision`. Do not touch the severity/`consecutive_incorrect` logic below it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/warning_engine && venv/bin/pytest tests/test_reversal_passthrough.py -v`
Expected: PASS

- [ ] **Step 5: Run the full warning_engine test suite**

Run: `cd apps/warning_engine && venv/bin/pytest -v`
Expected: PASS — `test_advisory_flag.py` and `test_engagement_fusion.py` unaffected.

- [ ] **Step 6: Commit**

```bash
git add apps/warning_engine/main.py apps/warning_engine/tests/test_reversal_passthrough.py
git commit -m "feat: forward meaning_reversal_detected flag through warning engine"
```

---

## Task 4: Storage & retrieval (`backend`)

**Files:**
- Modify: `apps/backend/app/models.py`
- Modify: `apps/backend/app/schemas.py`
- Modify: `apps/backend/app/database.py`
- Modify: `apps/backend/app/routes.py`
- Test: `apps/backend/tests/test_reversal_column.py`

**Interfaces:**
- Consumes: `"meaning_reversal_detected"` key in the JSON POSTed to `/window/result` from Task 3.
- Modifies: `WindowResult` model (`apps/backend/app/models.py:58-74`) — new column `meaning_reversal_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)`.
- Modifies: `WindowResultRequest` schema (`apps/backend/app/schemas.py:61-68`) — add `meaning_reversal_detected: bool = False`.
- Modifies: `_EXPECTED_COLUMNS` (`apps/backend/app/database.py:28-33`) — add `("window_results", "meaning_reversal_detected", "BOOLEAN DEFAULT 0")`.
- Modifies: `log_window_result` (`apps/backend/app/routes.py:636-659`) and `get_windows` (`apps/backend/app/routes.py:662-680`).

- [ ] **Step 1: Write the failing tests**

```python
# apps/backend/tests/test_reversal_column.py
import pytest
from app.models import WindowResult


@pytest.mark.asyncio
async def test_window_result_model_roundtrip_with_reversal_flag(backend_client, db_session):
    row = WindowResult(
        barter_session_id=1, window_number=1, classification="incorrect",
        cosine_similarity=0.5, text_content="I will not pay.",
        meaning_reversal_detected=True,
    )
    db_session.add(row)
    await db_session.commit()
    assert row.meaning_reversal_detected is True


@pytest.mark.asyncio
async def test_post_window_result_stores_and_returns_reversal_flag(backend_client):
    resp = await backend_client.post("/window/result", json={
        "barter_id": 1, "window_id": 1, "classification": "incorrect",
        "similarity_score": 0.5, "text_preview": "I will not pay.",
        "timestamp_start": 0.0, "timestamp_end": 25.0,
        "meaning_reversal_detected": True,
    })
    assert resp.status_code == 200

    resp = await backend_client.get("/session/1/windows")
    assert resp.status_code == 200
    windows = resp.json()
    assert len(windows) == 1
    assert windows[0]["meaning_reversal_detected"] is True


@pytest.mark.asyncio
async def test_post_window_result_defaults_reversal_flag_false(backend_client):
    resp = await backend_client.post("/window/result", json={
        "barter_id": 1, "window_id": 2, "classification": "correct",
        "similarity_score": 0.5, "text_preview": "on topic",
        "timestamp_start": 0.0, "timestamp_end": 25.0,
    })
    assert resp.status_code == 200

    resp = await backend_client.get("/session/1/windows")
    windows = resp.json()
    matching = [w for w in windows if w["window_id"] == 2]
    assert matching[0]["meaning_reversal_detected"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/backend && venv/bin/pytest tests/test_reversal_column.py -v`
Expected: FAIL — `TypeError` (unexpected keyword `meaning_reversal_detected`) and missing response key.

- [ ] **Step 3: Add the column to `apps/backend/app/models.py`**

Add `meaning_reversal_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)` to `WindowResult`, directly after the existing `labeled_at` column, following the same style as the `human_label` columns added there.

- [ ] **Step 4: Add the field to `apps/backend/app/schemas.py`**

Add `meaning_reversal_detected: bool = False` to `WindowResultRequest` (`apps/backend/app/schemas.py:61-68`).

- [ ] **Step 5: Add the migration entry to `apps/backend/app/database.py`**

Add `("window_results", "meaning_reversal_detected", "BOOLEAN DEFAULT 0")` to the `_EXPECTED_COLUMNS` tuple.

- [ ] **Step 6: Update the routes in `apps/backend/app/routes.py`**

In `log_window_result`, pass `meaning_reversal_detected=req.meaning_reversal_detected` into the `WindowResult(...)` constructor. In `get_windows`, add `"meaning_reversal_detected": w.meaning_reversal_detected` to the returned dict.

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd apps/backend && venv/bin/pytest tests/test_reversal_column.py -v`
Expected: PASS

- [ ] **Step 8: Run the full backend test suite**

Run: `cd apps/backend && venv/bin/pytest -v`
Expected: PASS — `test_advisory_column.py` (the precedent for this exact migration pattern) and all other tests unaffected.

- [ ] **Step 9: Commit**

```bash
git add apps/backend/app/models.py apps/backend/app/schemas.py apps/backend/app/database.py apps/backend/app/routes.py apps/backend/tests/test_reversal_column.py
git commit -m "feat: store and expose meaning_reversal_detected on window results"
```

---

## Task 5: Synthesized evaluation corpus

**Files:**
- Create: `apps/semantic_analysis/ground_truth/reversal_eval_corpus.json`
- Create: `apps/semantic_analysis/ground_truth/evaluate_reversal_detection.py`
- Test: `apps/semantic_analysis/tests/test_evaluate_reversal_detection.py`

**Interfaces:**
- Consumes: `detect_meaning_reversal` from Task 1.
- Produces: `reversal_eval_corpus.json` — a list of `{"text": str, "category": "commitment" | "reversal" | "distractor" | "double_negation", "expected_reversal": bool}` records.
- Produces: `load_corpus(path: Path) -> list[dict]`, `evaluate_corpus(records: list[dict]) -> dict` — returns `{"precision": float, "recall": float, "by_category": dict[str, dict]}` (per-category hit/miss counts, so the double-negation category's known failures are visible in the report rather than folded into an aggregate that hides them).
- Produces: CLI `main(argv=None) -> int`, run as `python evaluate_reversal_detection.py`, printing the metrics table (mirrors `docs/threshold-calibration.md`'s reporting style — this is not a new format).

Unlike `generate_synthetic.py`, this corpus needs no SentenceTransformer embedding step — `detect_meaning_reversal` is a pure text function, independent of topic/cosine similarity. This is a deliberate scope reduction from the spec's "mirrors the generate_synthetic.py pattern" framing: the *labeled-corpus-plus-standalone-evaluation-script* shape is reused; the embedding step is not, because nothing in this feature depends on it.

- [ ] **Step 1: Write the failing tests**

```python
# apps/semantic_analysis/tests/test_evaluate_reversal_detection.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ground_truth"))
from evaluate_reversal_detection import load_corpus, evaluate_corpus


def test_load_corpus_reads_json(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps([
        {"text": "I will pay you.", "category": "commitment", "expected_reversal": False},
    ]))
    records = load_corpus(corpus_path)
    assert records == [{"text": "I will pay you.", "category": "commitment", "expected_reversal": False}]


def test_evaluate_corpus_computes_precision_recall_and_by_category():
    records = [
        {"text": "I will not pay you.", "category": "reversal", "expected_reversal": True},
        {"text": "I will pay you.", "category": "commitment", "expected_reversal": False},
        {"text": "I am not sure about the weather, but I will pay you the full amount.",
         "category": "distractor", "expected_reversal": False},
    ]
    result = evaluate_corpus(records)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["by_category"]["reversal"]["correct"] == 1
    assert result["by_category"]["commitment"]["correct"] == 1
    assert result["by_category"]["distractor"]["correct"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/semantic_analysis && venv/bin/pytest tests/test_evaluate_reversal_detection.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `apps/semantic_analysis/ground_truth/evaluate_reversal_detection.py`**

`load_corpus(path)`: `json.loads(path.read_text())`.

`evaluate_corpus(records)`: for each record, compute `detect_meaning_reversal(record["text"])` (import from `reversal_detection` — add `sys.path.insert` for the parent `semantic_analysis` dir, matching `generate_synthetic.py`'s existing `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` line) and compare to `record["expected_reversal"]`. Accumulate true positives / false positives / false negatives / true negatives overall and per `category`; return `{"precision": tp / (tp + fp) if (tp + fp) else 1.0, "recall": tp / (tp + fn) if (tp + fn) else 1.0, "by_category": {cat: {"correct": n_correct, "total": n_total} for each category}}`.

`main(argv=None)`: load `reversal_eval_corpus.json` (same directory), call `evaluate_corpus`, print precision/recall and the per-category table, following `docs/threshold-calibration.md`'s existing style of naming what is and isn't covered (explicitly print the `double_negation` category's result with a note that it is a known, expected-imperfect category, not a target for 100% accuracy).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/semantic_analysis && venv/bin/pytest tests/test_evaluate_reversal_detection.py -v`
Expected: PASS

- [ ] **Step 5: Populate `reversal_eval_corpus.json`**

Use a small codex agent to generate this dataset (per the project's established approach for missing datasets), given: `COMMITMENT_PHRASES`, `NEGATION_CUES`, and `NEGATION_PROXIMITY_TOKENS` from Task 1's `reversal_detection.py`, and this task's four categories with their exact meaning (`commitment`: on-topic, no negation, `expected_reversal: false`; `reversal`: meaning-reversed counterpart of a commitment sentence, `expected_reversal: true`; `distractor`: negation present but unrelated to any commitment or too far from one, `expected_reversal: false`; `double_negation`: two stacked negations that actually affirm the commitment, `expected_reversal: false`, included specifically to measure — not hide — the known limitation). Target at least 15 examples per category (60+ total), spanning a range of the barter-domain commitment phrases so no single phrase dominates the corpus.

- [ ] **Step 6: Run the evaluation and record results**

Run: `cd apps/semantic_analysis/ground_truth && ../venv/bin/python evaluate_reversal_detection.py`
Record the printed precision/recall/by-category table in a new short doc, `apps/semantic_analysis/ground_truth/reversal_detection_findings.md`, following `threshold_experiment_findings.md`'s reporting style — including the `double_negation` category's (expected non-perfect) result stated plainly, not omitted.

- [ ] **Step 7: Commit**

```bash
git add apps/semantic_analysis/ground_truth/evaluate_reversal_detection.py apps/semantic_analysis/ground_truth/reversal_eval_corpus.json apps/semantic_analysis/ground_truth/reversal_detection_findings.md apps/semantic_analysis/tests/test_evaluate_reversal_detection.py
git commit -m "feat: add synthesized evaluation corpus for meaning-reversal detection"
```
