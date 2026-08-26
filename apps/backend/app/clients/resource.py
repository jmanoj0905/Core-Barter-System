"""HTTP client for the Resource Agent.

Backend owns reputation; resource_agent owns credits. Every call here is
idempotent on the resource_agent side, so retries are safe.
"""

import httpx

from app.config import settings

TIMEOUT = 10.0


class ResourceUnavailable(Exception):
    """resource_agent could not be reached."""


class InsufficientCredits(Exception):
    def __init__(
        self,
        user_id: int,
        required: int,
        available: int,
        shortfall: int,
        regen_eta: dict | None = None,
    ):
        self.user_id = user_id
        self.required = required
        self.available = available
        self.shortfall = shortfall
        self.regen_eta = regen_eta
        super().__init__(f"user {user_id} short by {shortfall} credits")


class ResourceProtocolError(Exception):
    """resource_agent returned an error response backend doesn't understand
    the shape of (e.g. a 409 missing the keys this client expects)."""


class ResourceClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.RESOURCE_URL).rstrip("/")

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                res = await client.post(f"{self.base_url}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise ResourceUnavailable(str(exc)) from exc

        if res.status_code == 409:
            try:
                detail = res.json().get("detail", {})
            except ValueError:
                detail = {}
            if detail.get("code") == "INSUFFICIENT_CREDITS":
                required_keys = ("user_id", "required", "available", "shortfall")
                if all(k in detail for k in required_keys):
                    raise InsufficientCredits(
                        detail["user_id"],
                        detail["required"],
                        detail["available"],
                        detail["shortfall"],
                        detail.get("regen_eta"),
                    )
            raise ResourceProtocolError(
                f"resource_agent returned 409 with an unrecognized body: {detail!r}"
            )
        res.raise_for_status()
        return res.json()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                res = await client.get(f"{self.base_url}{path}", params=params or {})
        except httpx.HTTPError as exc:
            raise ResourceUnavailable(str(exc)) from exc
        res.raise_for_status()
        return res.json()

    async def ensure_account(self, user_id: int) -> dict:
        return await self._post("/resource/accounts", {"user_id": user_id})

    async def get_account(self, user_id: int, trust_score: float = 0.5) -> dict:
        return await self._get(
            f"/resource/accounts/{user_id}", {"trust_score": trust_score}
        )

    async def reserve(self, session_id: int, participants: list[dict]) -> dict:
        return await self._post(
            "/resource/escrow/reserve",
            {"session_id": session_id, "participants": participants},
        )

    async def settle(
        self, session_id: int, verdict_type: str, qa_score: float, per_user: dict
    ) -> dict:
        return await self._post(
            "/resource/escrow/settle",
            {
                "session_id": session_id,
                "verdict_type": verdict_type,
                "qa_score": qa_score,
                "per_user": {str(k): v for k, v in per_user.items()},
            },
        )

    async def void(self, session_id: int, reason: str) -> dict:
        return await self._post(
            "/resource/escrow/void", {"session_id": session_id, "reason": reason}
        )

    async def get_escrow(self, session_id: int) -> dict:
        return await self._get(f"/resource/escrow/{session_id}")

    async def get_ledger(self, user_id: int) -> dict:
        return await self._get(f"/resource/ledger/{user_id}")


resource_client = ResourceClient()
