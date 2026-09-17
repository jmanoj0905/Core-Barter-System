import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("video-engagement")

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=15.0)
    yield
    await http_client.aclose()


app = FastAPI(title="Video Engagement", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"service": "video-engagement", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
