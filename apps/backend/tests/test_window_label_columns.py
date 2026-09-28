"""The human-label columns must exist on databases created before they were
declared in models.py.

create_all() creates missing TABLES only — never missing columns on a table that
already exists — so the feedback endpoint wrote human_label against databases
that had no such column, and every rating was lost. See
apps/semantic_analysis/ground_truth/design_decisions.md (D4).
"""

import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.database import _EXPECTED_COLUMNS, _add_missing_columns  # noqa: E402

LABEL_COLUMNS = ("human_label", "labeled_by_user_id", "labeled_at")

# window_results as it existed before the label columns were declared.
LEGACY_SCHEMA = """
CREATE TABLE window_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL,
    window_number INTEGER NOT NULL,
    classification VARCHAR(20) NOT NULL,
    cosine_similarity FLOAT NOT NULL,
    text_content TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@pytest.fixture
def legacy_engine(tmp_path):
    """A real SQLAlchemy engine over a pre-label-columns database.

    Exercising the genuine connection interface matters here: an earlier version
    of this test used a hand-rolled stand-in that accepted calls the real
    connection rejects, and passed against code that could not work.
    """
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


def test_legacy_schema_really_is_missing_the_columns(legacy_engine):
    with legacy_engine.begin() as conn:
        assert not _columns(conn) & set(LABEL_COLUMNS)


def test_backfill_adds_every_label_column(legacy_engine):
    with legacy_engine.begin() as conn:
        added = _add_missing_columns(conn)
        assert _columns(conn) >= set(LABEL_COLUMNS)
    assert sorted(added) == sorted(f"window_results.{c}" for c in LABEL_COLUMNS)


def test_backfill_is_idempotent(legacy_engine):
    with legacy_engine.begin() as conn:
        _add_missing_columns(conn)
    for _ in range(2):
        with legacy_engine.begin() as conn:
            assert _add_missing_columns(conn) == []


def test_backfill_accepts_a_write_to_the_new_columns(legacy_engine):
    with legacy_engine.begin() as conn:
        _add_missing_columns(conn)
        conn.execute(text(
            "INSERT INTO window_results (barter_session_id, window_number, classification,"
            " cosine_similarity, text_content, human_label, labeled_by_user_id, labeled_at)"
            " VALUES (1, 1, 'weakly_correct', 0.41, 'text', 'correct', 7, datetime('now'))"
        ))
        row = conn.execute(
            text("SELECT human_label, labeled_by_user_id FROM window_results")
        ).first()
    assert tuple(row) == ("correct", 7)


def test_missing_table_is_skipped_not_created(tmp_path):
    """create_all() owns table creation; the backfill must not invent tables."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'empty.db'}")
    try:
        with engine.begin() as conn:
            assert _add_missing_columns(conn) == []
            tables = {
                r[0] for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
            }
        assert "window_results" not in tables
    finally:
        engine.dispose()


def test_expected_columns_match_the_model():
    """Guard against the declaration drifting from the ORM again."""
    from app.models import Warning, WindowResult

    declared_by_table = {
        "window_results": set(WindowResult.__table__.columns.keys()),
        "warnings": set(Warning.__table__.columns.keys()),
    }
    for table, column, _ddl in _EXPECTED_COLUMNS:
        declared = declared_by_table[table]
        assert column in declared, f"{column} is backfilled but no longer declared on the model"
