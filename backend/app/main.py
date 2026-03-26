import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from app.websocket import manager

logger = logging.getLogger("backend-core")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize NudeNet detector for NSFW frame checking
    from app.safety import init_detector
    try:
        init_detector()
    except Exception as e:
        logger.warning("NudeNet init failed (NSFW checks disabled): %s", e)
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


# WebRTC signaling: relay SDP offers/answers and ICE candidates between peers
signal_peers: dict[int, dict[int, WebSocket]] = {}


@app.websocket("/ws/signal/{barter_id}/{user_id}")
async def signal_ws(barter_id: int, user_id: int, ws: WebSocket):
    await ws.accept()
    signal_peers.setdefault(barter_id, {})[user_id] = ws

    # Tell the newcomer about any already-connected peer, and vice versa
    for uid, peer_ws in signal_peers[barter_id].items():
        if uid != user_id:
            try:
                await peer_ws.send_json({"type": "peer_joined", "from": user_id})
                await ws.send_json({"type": "peer_joined", "from": uid})
            except Exception:
                pass

    try:
        while True:
            data = await ws.receive_json()
            target_id = 2 if user_id == 1 else 1
            target_ws = signal_peers.get(barter_id, {}).get(target_id)
            if target_ws:
                try:
                    await target_ws.send_json({**data, "from": user_id})
                except Exception:
                    pass
    except (WebSocketDisconnect, RuntimeError):
        signal_peers.get(barter_id, {}).pop(user_id, None)
