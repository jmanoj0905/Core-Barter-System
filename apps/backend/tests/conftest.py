import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from app.models import Base  # noqa: E402

TEST_DB_URL = "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true"
test_engine = create_async_engine(TEST_DB_URL, echo=False)
test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)


async def override_get_db():
    async with test_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def backend_client():
    with patch("app.safety.init_detector", lambda: None):
        from app.main import app
        from app.database import get_db

        app.dependency_overrides[get_db] = override_get_db

        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

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


@pytest_asyncio.fixture
async def db_session():
    async with test_session_factory() as session:
        yield session
