import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest_asyncio.fixture
async def semantic_client():
    spec = importlib.util.spec_from_file_location("semantic_analysis_main", APP_DIR / "main.py")
    sa_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sa_main)

    # Initialize model manually (lifespan not triggered in tests)
    from sentence_transformers import SentenceTransformer
    sa_main.model = SentenceTransformer("all-MiniLM-L6-v2")

    mock_http = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = lambda: None
    mock_http.post.return_value = mock_response

    sa_main.http_client = mock_http
    sa_main.contracts.clear()
    sa_main.engagement_state.clear()
    sa_main.contracts[1] = {
        "topic": "test", "scope": "test",
        "topic_embedding": sa_main.embed("test topic"),
        "teacher_user_id": 1, "learner_user_id": 2,
    }

    transport = ASGITransport(app=sa_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_http, sa_main
