import json
import logging
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger("websocket-manager")


class ConnectionManager:
    def __init__(self):
        # barter_id -> list of connected WebSockets
        self._connections: dict[int, list[WebSocket]] = defaultdict(list)

    async def connect(self, barter_id: int, ws: WebSocket):
        await ws.accept()
        self._connections[barter_id].append(ws)
        logger.info("Client connected to barter %d (%d total)", barter_id, len(self._connections[barter_id]))

    def disconnect(self, barter_id: int, ws: WebSocket):
        self._connections[barter_id].remove(ws)
        if not self._connections[barter_id]:
            del self._connections[barter_id]
        logger.info("Client disconnected from barter %d", barter_id)

    async def broadcast(self, barter_id: int, payload: dict):
        clients = self._connections.get(barter_id, [])
        if not clients:
            return
        message = json.dumps(payload)
        dead = []
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(barter_id, ws)


manager = ConnectionManager()
