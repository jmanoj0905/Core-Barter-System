from pydantic import BaseModel, Field


class AccountCreateRequest(BaseModel):
    user_id: int


class ReserveParticipant(BaseModel):
    user_id: int
    trust_score: float = Field(ge=0.0, le=1.0)


class ReserveRequest(BaseModel):
    session_id: int
    participants: list[ReserveParticipant]


class ParticipantMetrics(BaseModel):
    quality: float = 0.0
    engagement: float = 0.0
    no_show: bool = False


class SettleRequest(BaseModel):
    session_id: int
    verdict_type: str
    qa_score: float
    per_user: dict[str, ParticipantMetrics] = {}


class VoidRequest(BaseModel):
    session_id: int
    reason: str


class HoldRequest(BaseModel):
    reason: str


class DisputeResolveRequest(BaseModel):
    resolution: str  # "settle" | "void"
    note: str = ""
    verdict_type: str = "PARTIAL"
    qa_score: float = 0.5
    resolved_by: str = "admin"
