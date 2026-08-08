import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from app.websocket import manager

# ── Terminal colours ────────────────────────────────────────────────────────
_R = "\033[0;31m";  _G = "\033[0;32m";  _Y = "\033[1;33m"
_C = "\033[0;36m";  _M = "\033[0;35m";  _B = "\033[1;34m"
_W = "\033[1;37m";  _NC = "\033[0m";    _BOLD = "\033[1m"

def _banner(msg): print(f"\n{_B}{_BOLD}{'─'*56}{_NC}\n  {_W}{_BOLD}{msg}{_NC}\n{_B}{_BOLD}{'─'*56}{_NC}", flush=True)
def _ok(msg):     print(f"  {_G}✓{_NC}  {msg}", flush=True)
def _info(msg):   print(f"  {_C}→{_NC}  {msg}", flush=True)
def _warn(msg):   print(f"  {_Y}⚠{_NC}  {_Y}{msg}{_NC}", flush=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("backend-core")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _banner("Backend Core  ·  Port 8000  ·  SQLite")
    _info("Initializing database …")

    from app.database import init_db, async_session
    from app.models import User
    from sqlalchemy import select

    await init_db()

    async with async_session() as db:
        for uid, name in [(1, "Alice"), (2, "Bob")]:
            result = await db.execute(select(User).where(User.id == uid))
            if not result.scalar_one_or_none():
                db.add(User(id=uid, username=name, trust_score=1.0))
        await db.commit()

    _ok("Database ready — tables created, users seeded")

    from app.safety import init_detector
    try:
        init_detector()
        _ok("NudeNet NSFW detector loaded")
    except Exception as e:
        _warn(f"NudeNet init skipped ({e}) — NSFW checks disabled")

    _ok("Service online — REST API + WebSocket ready")
    yield


app = FastAPI(title="Core Barter System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"service": "backend-core", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/warnings/{barter_id}")
async def warnings_ws(barter_id: int, ws: WebSocket):
    await manager.connect(barter_id, ws)
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        manager.disconnect(barter_id, ws)


import logging

logger = logging.getLogger("signal")

# WebRTC signaling: relay SDP offers/answers and ICE candidates between peers
signal_peers: dict[int, dict[int, WebSocket]] = {}


@app.websocket("/ws/signal/{barter_id}/{user_id}")
async def signal_ws(barter_id: int, user_id: int, ws: WebSocket):
    await ws.accept()
    signal_peers.setdefault(barter_id, {})[user_id] = ws
    logger.info(f"User {user_id} connected to signal WS for barter {barter_id}")

    # Tell the newcomer about any already-connected peer, and vice versa
    for uid, peer_ws in signal_peers[barter_id].items():
        if uid != user_id:
            try:
                logger.info(f"Notifying peer {uid} that user {user_id} joined")
                await peer_ws.send_json({"type": "peer_joined", "from": user_id})
                await ws.send_json({"type": "peer_joined", "from": uid})
            except Exception as e:
                logger.error(f"Error notifying peer: {e}")

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "unknown")
            logger.info(f"Relaying {msg_type} from user {user_id} to peer")
            
            target_id = 2 if user_id == 1 else 1
            target_ws = signal_peers.get(barter_id, {}).get(target_id)
            if target_ws:
                try:
                    await target_ws.send_json({**data, "from": user_id})
                except Exception as e:
                    logger.error(f"Error relaying to peer: {e}")
            else:
                logger.warning(f"No peer found for user {target_id}")
    except (WebSocketDisconnect, RuntimeError):
        signal_peers.get(barter_id, {}).pop(user_id, None)
