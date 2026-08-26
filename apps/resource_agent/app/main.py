import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("resource-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Resource Agent · Port 8004 · PostgreSQL")

    from app.database import async_session, init_db
    from app.ledger import SYSTEM_KINDS, get_or_create_account

    await init_db()

    async with async_session() as db:
        for kind in SYSTEM_KINDS:
            await get_or_create_account(db, kind, None)
        await db.commit()

    logger.info("Database ready — tables created, system accounts seeded")
    yield


app = FastAPI(title="Resource Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/resource/health")
async def health():
    return {"service": "resource-agent", "status": "ok"}
