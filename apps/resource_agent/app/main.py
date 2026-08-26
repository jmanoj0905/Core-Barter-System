import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("resource-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    logger.info("Resource Agent · Port 8004 · PostgreSQL")

    from app.database import async_session, init_db
    from app.ledger import SYSTEM_KINDS, get_or_create_account
    from app.reconciler import reconciler_loop

    await init_db()

    async with async_session() as db:
        for kind in SYSTEM_KINDS:
            await get_or_create_account(db, kind, None)
        await db.commit()

    logger.info("Database ready — tables created, system accounts seeded")

    task = asyncio.create_task(reconciler_loop())
    logger.info("Reconciler started (every 300s)")
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Resource Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/resource/health")
async def health():
    from app.reconciler import LAST_REPORT, is_healthy

    return {
        "service": "resource-agent",
        "status": "ok" if is_healthy(LAST_REPORT) else "degraded",
        "last_reconcile": LAST_REPORT,
    }
