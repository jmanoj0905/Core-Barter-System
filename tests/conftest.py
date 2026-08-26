"""
E2E test fixtures for the Core Barter System.

Uses an in-memory SQLite database so tests run without PostgreSQL.
Provides async test clients for backend, warning engine, and semantic analysis.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# ---------------------------------------------------------------------------
# Set env vars BEFORE any backend import (Settings reads them at import time)
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DATABASE_URL_SYNC", "sqlite:///:memory:")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")

# ---------------------------------------------------------------------------
# Make service packages importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "backend"))

# ---------------------------------------------------------------------------
# Patch routes to use naive datetimes (SQLite doesn't store tz info).
# Production uses PostgreSQL which handles tz natively. This keeps the
# backend code unmodified while making tests work with SQLite.
# ---------------------------------------------------------------------------
import app.routes as _routes  # noqa: E402

_original_datetime = _routes.datetime


class _NaiveDatetime(_original_datetime.__class__):
    """datetime.now() always returns naive UTC (matches what SQLite stores)."""

    @classmethod
    def now(cls, tz=None):
        return _original_datetime.now()  # naive, no tz


_routes.datetime = _NaiveDatetime

# ---------------------------------------------------------------------------
# Backend: override database to use async SQLite
# ---------------------------------------------------------------------------
from app.models import Base  # noqa: E402

TEST_DB_URL = "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)


async def override_get_db():
    async with test_session_factory() as session:
        yield session


async def _default_fake_reserve(self, session_id, participants):
    """Default resource_agent stand-in: reserve 5cr per participant.

    The e2e suite runs without a live resource_agent (in-memory SQLite,
    ASGI transport, no Docker services). Individual tests that need to
    assert on the delegation itself (e.g. test_session_start_locks_real_escrow)
    override this via monkeypatch.setattr after the fixture has set it up.
    """
    return {
        "session_id": session_id,
        "escrows": [
            {"user_id": p["user_id"], "amount": 5, "state": "RESERVED"}
            for p in participants
        ],
        "created": True,
    }


async def _default_fake_settle(self, session_id, verdict_type, qa_score, per_user):
    return {
        "session_id": session_id,
        "mode": verdict_type,
        "breakdown": {str(uid): {"released": 5, "bonus": 0} for uid in per_user},
        "entry_id": 1,
    }


async def _default_fake_ensure_account(self, user_id):
    return {"user_id": user_id, "available": 100, "locked": 0}


@pytest_asyncio.fixture
async def backend_client(monkeypatch):
    """Async test client for the backend FastAPI app, backed by in-memory SQLite."""
    # Patch safety init to avoid loading NudeNet
    with patch("app.safety.init_detector", lambda: None):
        from app.main import app
        from app.database import get_db
        from app.clients.resource import ResourceClient

        # resource_agent isn't running in the e2e suite; stub it with sane
        # defaults so tests exercising the ordinary session flow don't need
        # a live service. Tests asserting delegation behavior itself
        # (e.g. test_session_start_locks_real_escrow) re-patch these via
        # their own monkeypatch calls, which take precedence.
        monkeypatch.setattr(ResourceClient, "reserve", _default_fake_reserve)
        monkeypatch.setattr(ResourceClient, "settle", _default_fake_settle)
        monkeypatch.setattr(ResourceClient, "ensure_account", _default_fake_ensure_account)

        app.dependency_overrides[get_db] = override_get_db

        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Seed Alice and Bob
        async with test_session_factory() as db:
            from app.models import User

            db.add(User(id=1, username="alice", trust_score=1.0))
            db.add(User(id=2, username="bob", trust_score=1.0))
            await db.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Warning Engine: test client (in-memory, no external calls)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def warning_client():
    """Async test client for the warning engine, with backend calls mocked."""
    we_path = str(ROOT / "apps" / "warning_engine")
    if we_path not in sys.path:
        sys.path.insert(0, we_path)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "warning_engine_main", ROOT / "apps" / "warning_engine" / "main.py"
    )
    we_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(we_main)

    app = we_main.app

    # Mock the http_client used by post_to_backend
    mock_http = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = lambda: None
    mock_http.post.return_value = mock_response

    we_main.http_client = mock_http
    we_main.sessions.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_http

    we_main.sessions.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def create_session_payload():
    """Default payload for creating a barter session."""
    return {
        "topic": "Python programming basics",
        "scope": "Variables, loops, functions, and data types in Python",
        "agreed_duration_minutes": 10,
        "teacher_user_id": 1,
        "learner_user_id": 2,
    }
