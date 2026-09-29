# Advisory `severe` Warnings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a single `severe` warning from destroying a teacher's escrow, by making `severe` an advisory record instead of a payout veto.

**Architecture:** Two edits to pure/near-pure functions in `apps/backend/app/routes.py` remove the veto from the settlement path; one additive nullable column marks severe warnings advisory so Half B's adjudicator can find them later. In-session warning behaviour — when warnings fire, what they say, how they are broadcast and displayed — is deliberately unchanged.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.x (async), SQLite, pytest + pytest-asyncio (`asyncio_mode = auto`).

**Spec:** [docs/superpowers/specs/2026-09-28-advisory-severe-design.md](../specs/2026-09-28-advisory-severe-design.md)

## Global Constraints

- **Do not touch `apps/semantic_analysis/main.py`.** Spec F3: the dataset-A false-digression count must be *identical*, which holds by construction only if `classify()`, `UPPER = 0.36` and `LOWER = 0.14` are untouched.
- **Do not touch `apps/backend/app/escrow.py`.** `apply_settlement` already keys off `qa_score`; spec F4 fails if its behaviour changes for a given score.
- **Do not change the verdict percentage bands.** `< 40` and `>= 70` in `_decide_verdict_type` stay exactly as written.
- **Do not change when a warning fires.** The ladder in `run_warning_decision` (`1→silent`, `2→strong`, `3+→severe`) keeps its thresholds. Spec F5.
- **Columns added to existing tables must be additive and nullable**, registered in `_EXPECTED_COLUMNS` in `apps/backend/app/database.py`. That helper is not a migration framework and must never drop, rename or retype.
- Test commands run from the repo root using the backend venv: `apps/backend/venv/bin/python -m pytest <path> -q`. `apps/warning_engine` has no venv of its own; the backend venv runs its suite too.

## Review Focus

Input classes the spec implies but does not give a task of their own. Each has a test assigned to the task owning the code.

1. **Rolling deploy — old `warning_engine` posts `/warnings/log` with no `advisory` field.** Must persist as not-advisory, never 422. (Task 4)
2. **Legacy rows — `warnings.advisory` reads as `0`, not NULL, on rows written before this change.** SQLite backfills the DEFAULT on `ADD COLUMN ... DEFAULT 0`, so pre-existing rows hold `0`; NULL remains reachable through other write paths. Reads must treat **both `0` and NULL** as not-advisory and must not crash the verdict path. (Task 3) *(Corrected during execution — an earlier draft of this line claimed NULL; `models.py`'s comment on the column has always been right.)*
3. **Exact band boundaries — `on_topic_percentage` of exactly 40.0 and exactly 70.0.** The bands must not shift by a floating-point hair while the clause next to them is edited. (Task 1)
4. **Zero windows.** Both directions must be unchanged: a zero-window session with duration and confirmations passing returns `SUCCESSFUL` today, and one with neither returns `DISPUTE`. Removing the severe clause must move neither. (Task 1)
5. **A severe warning must still reach the WebSocket.** Spec F5 — the fix changes what a warning *costs*, not whether it is seen. (Task 4)

---

### Task 1: `severe` stops vetoing the verdict

**Files:**
- Modify: `apps/backend/app/routes.py:160-182` (`_decide_verdict_type`)
- Test: `apps/backend/tests/test_verdict_policy.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_decide_verdict_type(terminated: bool, duration_pass: bool, confirmation_pass: bool, topic: dict) -> str`. Signature unchanged. It no longer reads `topic["has_severe_warning"]`; it reads only `topic["has_evidence"]` and `topic["on_topic_percentage"]`.

- [ ] **Step 1: Write the failing tests**

```python
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.routes import _decide_verdict_type  # noqa: E402


def topic(pct, evidence=True):
    return {"has_evidence": evidence, "on_topic_percentage": pct}


def test_severe_warning_no_longer_vetoes_a_good_session():
    # The bug: one severe warning cost the teacher the whole escrow.
    assert _decide_verdict_type(False, True, True, topic(85.0)) == "SUCCESSFUL"


def test_sustained_drift_still_disputes():
    assert _decide_verdict_type(False, True, True, topic(35.0)) == "DISPUTE"


def test_boundary_forty_is_not_a_dispute():
    assert _decide_verdict_type(False, True, True, topic(40.0)) == "PARTIAL"


def test_boundary_seventy_is_successful():
    assert _decide_verdict_type(False, True, True, topic(70.0)) == "SUCCESSFUL"


def test_zero_window_session_behaves_exactly_as_before():
    # ISSUE-019 guarantees only that missing evidence never MANUFACTURES a
    # DISPUTE. A zero-window session that passed duration and both
    # confirmations does return SUCCESSFUL today; Half A must not change
    # either direction. Both are pinned so neither can drift.
    assert _decide_verdict_type(False, True, True, topic(0.0, evidence=False)) == "SUCCESSFUL"
    assert _decide_verdict_type(False, False, False, topic(0.0, evidence=False)) == "DISPUTE"


def test_termination_still_dominates():
    assert _decide_verdict_type(True, True, True, topic(95.0)) == "DISPUTE"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `apps/backend/venv/bin/python -m pytest apps/backend/tests/test_verdict_policy.py -q`
Expected: FAIL — `KeyError: 'has_severe_warning'` on every case, because the current clause reads a key these fixtures do not supply.

- [ ] **Step 3: Delete the severe clause**

In `_decide_verdict_type`, `topic_failed` becomes `topic["has_evidence"] and topic["on_topic_percentage"] < 40`. Change nothing else in the function; update its docstring so it no longer claims a severe warning can veto.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `apps/backend/venv/bin/python -m pytest apps/backend/tests/test_verdict_policy.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 5: Run the whole backend suite for regressions**

Run: `apps/backend/venv/bin/python -m pytest apps/backend/tests -q`
Expected: PASS (11 pre-existing + 6 new).

- [ ] **Step 6: Commit**

```bash
git add apps/backend/app/routes.py apps/backend/tests/test_verdict_policy.py
git commit -m "fix: a severe warning no longer vetoes session payout"
```

---

### Task 2: `_evaluate_topic_quality` reports a count, not a veto flag

**Files:**
- Modify: `apps/backend/app/routes.py:132-157` (`_evaluate_topic_quality`)
- Test: `apps/backend/tests/test_verdict_policy.py` (extend)

**Interfaces:**
- Consumes: Task 1's `_decide_verdict_type`, which no longer reads `has_severe_warning`.
- Produces: `_evaluate_topic_quality(db, barter_id) -> dict` with keys `has_evidence: bool`, `on_topic_percentage: float`, `severe_warning_count: int`. The key `has_severe_warning` is gone.

- [ ] **Step 1: Write the failing test**

Use the `db_session` fixture from `apps/backend/tests/conftest.py`. Seed one `BarterSession` with three `WindowResult` rows (two `correct`, one `incorrect`) and two `Warning` rows with `severity="severe"`.

```python
async def test_topic_quality_counts_severe_warnings(db_session):
    result = await _evaluate_topic_quality(db_session, barter_id)
    assert result["severe_warning_count"] == 2
    assert "has_severe_warning" not in result
    assert result["has_evidence"] is True
    assert result["on_topic_percentage"] == 66.67
```

- [ ] **Step 2: Run it to verify it fails**

Run: `apps/backend/venv/bin/python -m pytest apps/backend/tests/test_verdict_policy.py -q -k severe_warnings`
Expected: FAIL — `KeyError: 'severe_warning_count'`.

- [ ] **Step 3: Replace the existence query with a count**

Swap the `select(Warning).limit(1)` for `select(func.count()).select_from(Warning)` over the same `barter_session_id` / `severity == "severe"` filter, and return `severe_warning_count` in place of `has_severe_warning`. `func` is already imported in this module.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `apps/backend/venv/bin/python -m pytest apps/backend/tests -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/backend/app/routes.py apps/backend/tests/test_verdict_policy.py
git commit -m "refactor: report severe warning count instead of a veto flag"
```

---

### Task 3: the `warnings.advisory` column

**Files:**
- Modify: `apps/backend/app/models.py:103-111` (`Warning`)
- Modify: `apps/backend/app/database.py:28-32` (`_EXPECTED_COLUMNS`)
- Create: `apps/backend/migrations/009_advisory_warnings.sql`
- Test: `apps/backend/tests/test_advisory_column.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Warning.advisory: Mapped[bool | None]`, nullable, DDL `BOOLEAN DEFAULT 0`. Readers must treat **both `0` and NULL** as not-advisory.

- [ ] **Step 1: Write the failing test**

Follow the shape of `apps/backend/tests/test_window_label_columns.py` — a real SQLAlchemy engine over a legacy schema, not a stand-in.

```python
LEGACY_SCHEMA = """
CREATE TABLE warnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    window_ids TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

def test_advisory_column_is_added_to_a_legacy_database(legacy_engine):
    with legacy_engine.begin() as conn:
        added = _add_missing_columns(conn)
    assert "warnings.advisory" in added

def test_adding_the_column_is_idempotent(legacy_engine):
    with legacy_engine.begin() as conn:
        _add_missing_columns(conn)
        added_again = _add_missing_columns(conn)
    assert "warnings.advisory" not in added_again

def test_pre_existing_warning_rows_read_as_not_advisory(legacy_engine):
    # Review Focus 2: rows written before this change read as 0 here (SQLite
    # backfills the DEFAULT), and NULL is still reachable through other write
    # paths. The assertion is falsiness, which covers both.
    with legacy_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO warnings (barter_session_id, severity, message) "
            "VALUES (1, 'severe', 'legacy row')"
        ))
        _add_missing_columns(conn)
        value = conn.execute(text("SELECT advisory FROM warnings")).scalar()
    assert not value
```

- [ ] **Step 2: Run to verify failure**

Run: `apps/backend/venv/bin/python -m pytest apps/backend/tests/test_advisory_column.py -q`
Expected: FAIL — `advisory` is not in `_EXPECTED_COLUMNS`, so `added` is empty.

- [ ] **Step 3: Declare the column in all three places**

Add `advisory: Mapped[bool | None] = mapped_column(Boolean, default=False, nullable=True)` to `Warning` in `models.py` (import `Boolean` from sqlalchemy); add `("warnings", "advisory", "BOOLEAN DEFAULT 0")` to `_EXPECTED_COLUMNS`; write `009_advisory_warnings.sql` with the single `ALTER TABLE warnings ADD COLUMN advisory BOOLEAN DEFAULT 0;`, carrying a header comment in the style of `008` that points readers at the startup path as the route that works on an existing database.

- [ ] **Step 4: Run to verify passing**

Run: `apps/backend/venv/bin/python -m pytest apps/backend/tests -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/backend/app/models.py apps/backend/app/database.py \
        apps/backend/migrations/009_advisory_warnings.sql \
        apps/backend/tests/test_advisory_column.py
git commit -m "feat: add warnings.advisory column"
```

---

### Task 4: mark severe warnings advisory end to end, and fix the banner

**Files:**
- Modify: `apps/backend/app/schemas.py:37-42` (`WarningLogRequest`)
- Modify: `apps/backend/app/routes.py:851-860` (`log_warning`)
- Modify: `apps/warning_engine/main.py:178-260` (`run_warning_decision`), `:133` (banner)
- Test: `apps/backend/tests/test_advisory_column.py` (extend), `apps/warning_engine/tests/test_advisory_flag.py` (create)

**Interfaces:**
- Consumes: Task 3's `Warning.advisory`.
- Produces: `WarningLogRequest.advisory: bool = False`. The `/warnings/log` broadcast payload is unchanged — `advisory` is persisted, not broadcast.

- [ ] **Step 1: Write the failing tests**

Backend, using `backend_client`:

```python
async def test_severe_warning_is_persisted_as_advisory(backend_client):
    await backend_client.post("/warnings/log", json={
        "barter_id": 1, "severity": "severe", "reason": "3 consecutive", "advisory": True})
    # assert the stored row has advisory truthy

async def test_warning_log_accepts_a_payload_with_no_advisory_field(backend_client):
    # Review Focus 1: an old warning_engine during a rolling deploy.
    resp = await backend_client.post("/warnings/log", json={
        "barter_id": 1, "severity": "strong", "reason": "2 consecutive"})
    assert resp.status_code == 200
    # assert the stored row is not advisory

async def test_severe_warning_is_still_broadcast(backend_client, monkeypatch):
    # Review Focus 5 / spec F5: patch app.routes.manager.broadcast with an
    # AsyncMock, POST a severe warning, then assert it was awaited once and
    # that the broadcast payload has severity "severe" and no "advisory" key.
```

Warning engine, using the `warning_client` fixture and its `mock_http`:

```python
async def test_severe_warning_posts_advisory_true(warning_client):
    # POST three consecutive "incorrect" windows to /window/result, then find
    # the mock_http call to "/warnings/log" and assert its json carries
    # advisory=True and severity="severe".

async def test_strong_warning_is_not_advisory(warning_client):
    # Two consecutive incorrect -> severity "strong", advisory False.

async def test_ladder_thresholds_are_unchanged(warning_client):
    # Spec F5: the 1st incorrect window returns action "silent", the 2nd
    # "warning"/"strong", the 3rd "warning"/"severe".
```

- [ ] **Step 2: Run both suites to verify failure**

Run: `apps/backend/venv/bin/python -m pytest apps/backend/tests apps/warning_engine/tests -q`
Expected: FAIL — `advisory` is not accepted by the schema, not persisted, and not sent.

- [ ] **Step 3: Wire `advisory` through**

Add `advisory: bool = False` to `WarningLogRequest`; pass `advisory=req.advisory` into the `Warning(...)` constructor in `log_warning`, leaving the broadcast payload exactly as it is. In `run_warning_decision`, include `"advisory": severity == "severe"` in the `/warnings/log` payload.

- [ ] **Step 4: Fix the stale startup banner**

`apps/warning_engine/main.py:133` prints `1→silent  2→mild  3–4→strong  5+→severe`. The ladder is and always has been `1→silent  2→strong  3+→severe`; there is no `mild` tier. Correct the string.

- [ ] **Step 5: Run both suites to verify passing**

Run: `apps/backend/venv/bin/python -m pytest apps/backend/tests apps/warning_engine/tests -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/backend/app/schemas.py apps/backend/app/routes.py \
        apps/warning_engine/main.py apps/backend/tests/test_advisory_column.py \
        apps/warning_engine/tests/test_advisory_flag.py
git commit -m "feat: mark severe warnings advisory; fix stale ladder banner"
```

---

## Done when

- [ ] Both suites green: `apps/backend/venv/bin/python -m pytest apps/backend/tests apps/warning_engine/tests -q`
- [ ] `grep -rn "has_severe_warning" apps/ | grep -v tests/` returns nothing. The criterion is **no production reference remains**; the literal unfiltered grep still matches `tests/test_verdict_policy.py`'s brief-mandated `assert "has_severe_warning" not in result`, which is the check itself and must stay.
- [ ] `apps/semantic_analysis/main.py` and `apps/backend/app/escrow.py` are untouched in `git diff --stat` (spec F3, F4).
