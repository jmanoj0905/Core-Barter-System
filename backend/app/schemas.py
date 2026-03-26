from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    skill_a: str = ""
    skill_b: str = ""
    topic: str
    scope: str
    agreed_duration_minutes: int
    teacher_user_id: int = 1
    learner_user_id: int = 2


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
