"""window_results.meaning_reversal_detected: storage, GET exposure, and the
legacy-database backfill. NULL = "predates the check", distinct from False."""

import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.database import _EXPECTED_COLUMNS, _add_missing_columns  # noqa: E402
from app.models import WindowResult  # noqa: E402

COLUMN = "meaning_reversal_detected"

# window_results as it existed before this column (label columns included).
LEGACY_SCHEMA = """
CREATE TABLE window_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL,
    window_number INTEGER NOT NULL,
    classification VARCHAR(20) NOT NULL,
    cosine_similarity FLOAT NOT NULL,
    text_content TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    human_label VARCHAR(20),
    labeled_by_user_id INTEGER,
    labeled_at DATETIME
);
"""


@pytest.fixture
def legacy_engine(tmp_path):
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.executescript(LEGACY_SCHEMA)
    raw.commit()
    raw.close()
    engine = create_engine(f"sqlite+pysqlite:///{db}")
    yield engine
    engine.dispose()


def _columns(conn):
    return {row[1] for row in conn.execute(text("PRAGMA table_info(window_results)"))}


def _post(**over):
    body = {
        "barter_id": 1, "window_id": 1, "classification": "incorrect",
        "similarity_score": 0.5, "text_preview": "I will not pay.",
        "timestamp_start": 0.0, "timestamp_end": 25.0,
    }
    body.update(over)
    return body


# --- storage / API -------------------------------------------------------

@pytest.mark.asyncio
async def test_window_result_model_roundtrip_with_reversal_flag(backend_client, db_session):
    row = WindowResult(
        barter_session_id=1, window_number=1, classification="incorrect",
        cosine_similarity=0.5, text_content="I will not pay.",
        meaning_reversal_detected=True,
    )
    db_session.add(row)
    await db_session.commit()
    await db_session.refresh(row)
    assert row.meaning_reversal_detected is True


@pytest.mark.asyncio
async def test_post_window_result_stores_and_returns_reversal_flag(backend_client):
    resp = await backend_client.post("/window/result", json=_post(meaning_reversal_detected=True))
    assert resp.status_code == 200

    resp = await backend_client.get("/session/1/windows")
    assert resp.status_code == 200
    windows = resp.json()
    assert len(windows) == 1
    assert windows[0]["meaning_reversal_detected"] is True


@pytest.mark.asyncio
async def test_post_window_result_defaults_reversal_flag_false(backend_client):
    # Old sender (rolling deploy): payload has no such field at all.
    resp = await backend_client.post("/window/result", json=_post(
        window_id=2, classification="correct", text_preview="on topic"))
    assert resp.status_code == 200

    windows = (await backend_client.get("/session/1/windows")).json()
    matching = [w for w in windows if w["window_id"] == 2]
    assert matching[0]["meaning_reversal_detected"] is False


@pytest.mark.asyncio
async def test_legacy_null_row_is_returned_as_none_not_false(backend_client, db_session):
    # A row written before the check existed has NULL: unknown, not "clean".
    db_session.add(WindowResult(
        barter_session_id=1, window_number=7, classification="correct",
        cosine_similarity=0.5, text_content="old"))
    await db_session.commit()

    windows = (await backend_client.get("/session/1/windows")).json()
    assert windows[0]["meaning_reversal_detected"] is None


# --- legacy database migration -------------------------------------------

def test_legacy_schema_really_lacks_the_column(legacy_engine):
    with legacy_engine.begin() as conn:
        assert COLUMN not in _columns(conn)


def test_backfill_adds_the_column_to_a_legacy_database(legacy_engine):
    with legacy_engine.begin() as conn:
        added = _add_missing_columns(conn)
        assert COLUMN in _columns(conn)
    assert f"window_results.{COLUMN}" in added


def test_backfill_is_idempotent(legacy_engine):
    with legacy_engine.begin() as conn:
        _add_missing_columns(conn)
    with legacy_engine.begin() as conn:
        assert f"window_results.{COLUMN}" not in _add_missing_columns(conn)


def test_pre_existing_rows_read_as_null_unknown(legacy_engine):
    with legacy_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO window_results (barter_session_id, window_number, classification,"
            " cosine_similarity, text_content) VALUES (1, 1, 'incorrect', 0.1, 'legacy row')"
        ))
        _add_missing_columns(conn)
        value = conn.execute(text(f"SELECT {COLUMN} FROM window_results")).scalar()
    assert value is None


def test_backfilled_column_accepts_a_write(legacy_engine):
    with legacy_engine.begin() as conn:
        _add_missing_columns(conn)
        conn.execute(text(
            "INSERT INTO window_results (barter_session_id, window_number, classification,"
            f" cosine_similarity, text_content, {COLUMN}) VALUES (1, 1, 'incorrect', 0.1, 't', 1)"
        ))
        assert conn.execute(text(f"SELECT {COLUMN} FROM window_results")).scalar() == 1


def test_expected_column_is_declared_on_the_model():
    assert ("window_results", COLUMN) in {(t, c) for t, c, _ in _EXPECTED_COLUMNS}
    assert COLUMN in WindowResult.__table__.columns.keys()
