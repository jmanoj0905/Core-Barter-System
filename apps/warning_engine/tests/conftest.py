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
async def warning_client():
    spec = importlib.util.spec_from_file_location("warning_engine_main", APP_DIR / "main.py")
    we_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(we_main)

    mock_http = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = lambda: None
    mock_http.post.return_value = mock_response

    we_main.http_client = mock_http
    we_main.sessions.clear()

    transport = ASGITransport(app=we_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_http, we_main

    we_main.sessions.clear()
