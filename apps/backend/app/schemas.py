from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SessionCreateRequest(BaseModel):
    skill_a: str = ""
    skill_b: str = ""
    topic: str = Field(min_length=1, max_length=200)
    scope: str = Field(min_length=1)
    agreed_duration_minutes: int = Field(gt=0, le=240)
    teacher_user_id: int = 1
    learner_user_id: int = 2

    @field_validator("topic", "scope")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def distinct_participants(self):
        if self.teacher_user_id == self.learner_user_id:
            raise ValueError("teacher_user_id and learner_user_id must be distinct")
        return self


class ConfirmRequest(BaseModel):
    user_id: int


class TerminateRequest(BaseModel):
    reason: str = "Manual termination"


class WarningLogRequest(BaseModel):
    barter_id: int
    severity: str
    reason: str
    window_ids: str = ""
    timestamp: str = ""


class FrameCheckRequest(BaseModel):
    barter_id: int
    user_id: int
    image_base64: str


class TranscriptSegmentRequest(BaseModel):
    barter_id: int
    user_id: int
    text: str
    duration_seconds: float
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0


class WindowResultRequest(BaseModel):
    barter_id: int
    window_id: int
    classification: str
    similarity_score: float
    text_preview: str = ""
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0


class DriftSummaryRequest(BaseModel):
    barter_id: int
    total_windows: int
    incorrect_windows: int
    percent_incorrect: float
    max_consecutive_incorrect: int
    total_drift_incidents: int = 0
    warning_count: int
    warnings: list[dict] = []
    terminated_early: bool = False


class WalletResponse(BaseModel):
    id: int
    user_id: int
    available_balance: int
    locked_balance: int
    total_earned: int
    total_spent: int
    trust_score: float = 1.0

    class Config:
        from_attributes = True


class EscrowResponse(BaseModel):
    id: int
    barter_session_id: int
    user_id: int
    amount: int
    status: str
    locked_at: str
    released_at: str | None = None
    release_type: str | None = None

    class Config:
        from_attributes = True

    @field_validator("locked_at", "released_at", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if v is None:
            return None
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return str(v)


class EscrowLockRequest(BaseModel):
    barter_session_id: int
    user_id: int


class EscrowReleaseRequest(BaseModel):
    escrow_id: int
    release_type: Literal["full_release", "partial_release", "refund", "penalty"]
    penalty_amount: int = Field(default=0, ge=0)


class SettlementRequest(BaseModel):
    barter_session_id: int
    qa_score: float


class SettlementResponse(BaseModel):
    provider_escrow_released: int
    learner_escrow_released: int
    provider_bonus: int
    learner_refund: int
    provider_trust_delta: float
    learner_trust_delta: float


class CreditTransactionResponse(BaseModel):
    id: int
    user_id: int
    barter_session_id: int | None
    transaction_type: str
    amount: int
    balance_after: int
    description: str

    class Config:
        from_attributes = True


class VideoEngagementRequest(BaseModel):
    user_id: int
    window_start: float
    window_end: float
    video_attention_score: float
    backend_used: str
    raw_signals: dict


class EngagementLogRequest(BaseModel):
    user_id: int | None = None
    speech_engagement_score: float | None = None
    video_attention_score: float | None = None
    fused_engagement_score: float
