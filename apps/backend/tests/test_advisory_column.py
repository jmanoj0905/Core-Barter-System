"""The warnings.advisory column must exist on databases created before it was
declared in models.py.

create_all() creates missing TABLES only — never missing columns on a table
that already exists — so a database built before this column was declared
keeps working right up until the first query touches `advisory`. See
apps/backend/app/database.py (_EXPECTED_COLUMNS / _add_missing_columns) for
the startup path that backfills it idempotently on an existing database.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, select, text

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.database import _add_missing_columns  # noqa: E402
from app.models import Warning  # noqa: E402

# warnings as it existed before the advisory column was declared.
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


@pytest.fixture
def legacy_engine(tmp_path):
    """A real SQLAlchemy engine over a pre-advisory-column database.

    Exercising the genuine connection interface matters here: a hand-rolled
    stand-in could accept calls the real connection rejects, and pass against
    code that could not work.
    """
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.executescript(LEGACY_SCHEMA)
    raw.commit()
    raw.close()

    engine = create_engine(f"sqlite+pysqlite:///{db}")
    yield engine
    engine.dispose()


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
    # SQLite backfills BOOLEAN DEFAULT 0 as 0, not NULL, on every pre-existing
    # row. Either way the value must read as falsy — assert falsiness, not
    # `is None`, so this pins the behaviour that matters (not-advisory) rather
    # than a specific stored representation.
    with legacy_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO warnings (barter_session_id, severity, message) "
            "VALUES (1, 'severe', 'legacy row')"
        ))
        _add_missing_columns(conn)
        value = conn.execute(text("SELECT advisory FROM warnings")).scalar()
    assert not value


@pytest.mark.asyncio
async def test_severe_warning_is_persisted_as_advisory(backend_client, db_session):
    resp = await backend_client.post("/warnings/log", json={
        "barter_id": 1, "severity": "severe", "reason": "3 consecutive", "advisory": True,
    })
    assert resp.status_code == 200

    warning = (await db_session.execute(select(Warning))).scalar_one()
    assert warning.advisory


@pytest.mark.asyncio
async def test_warning_log_accepts_a_payload_with_no_advisory_field(backend_client, db_session):
    # Review Focus 1: an old warning_engine during a rolling deploy — the
    # payload predates the `advisory` field entirely, not merely omits it.
    resp = await backend_client.post("/warnings/log", json={
        "barter_id": 1, "severity": "strong", "reason": "2 consecutive",
    })
    assert resp.status_code == 200

    warning = (await db_session.execute(select(Warning))).scalar_one()
    assert not warning.advisory


@pytest.mark.asyncio
async def test_severe_warning_is_still_broadcast(backend_client, monkeypatch):
    # Review Focus 5 / spec F5: advisory changes what a severe warning costs,
    # not whether it is seen — the broadcast is unchanged.
    mock_broadcast = AsyncMock()
    monkeypatch.setattr("app.routes.manager.broadcast", mock_broadcast)

    resp = await backend_client.post("/warnings/log", json={
        "barter_id": 1, "severity": "severe", "reason": "3 consecutive", "advisory": True,
    })
    assert resp.status_code == 200

    mock_broadcast.assert_awaited_once()
    _, payload = mock_broadcast.await_args.args
    assert payload["severity"] == "severe"
    assert "advisory" not in payload
