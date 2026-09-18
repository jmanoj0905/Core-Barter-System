from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, DateTime, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    trust_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BarterSession(Base):
    __tablename__ = "barter_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user1_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    user2_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SessionContract(Base):
    __tablename__ = "session_contracts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    topic: Mapped[str] = mapped_column(String(200), nullable=False)
    scope: Mapped[str] = mapped_column(Text, nullable=True)
    agreed_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    teacher_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    learner_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp_start: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp_end: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WindowResult(Base):
    __tablename__ = "window_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    window_number: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str] = mapped_column(String(20), nullable=False)
    cosine_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Human ground-truth for this window's classification, collected via the
    # referee feedback endpoint — separate from `classification`, which is
    # always the model's own prediction. Used to calibrate thresholds and,
    # eventually, fine-tune the embedding model.
    human_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    labeled_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class VideoEngagementResult(Base):
    __tablename__ = "video_engagement_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    window_start: Mapped[float] = mapped_column(Float, nullable=False)
    window_end: Mapped[float] = mapped_column(Float, nullable=False)
    video_attention_score: Mapped[float] = mapped_column(Float, nullable=False)
    backend_used: Mapped[str] = mapped_column(String(20), nullable=False)
    raw_signals: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EngagementScoreLog(Base):
    __tablename__ = "engagement_score_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    speech_engagement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_attention_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    fused_engagement_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Warning(Base):
    __tablename__ = "warnings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    window_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Verdict(Base):
    __tablename__ = "verdicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(
        ForeignKey("barter_sessions.id"), unique=True, nullable=False
    )
    verdict_type: Mapped[str] = mapped_column(String(20), nullable=False)
    on_topic_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    warning_count: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_check: Mapped[str] = mapped_column(String(10), nullable=False)
    confirmation_check: Mapped[str] = mapped_column(String(10), nullable=False)
    actual_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    trust_delta_user1: Mapped[float] = mapped_column(Float, nullable=False)
    trust_delta_user2: Mapped[float] = mapped_column(Float, nullable=False)
    drift_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Set once by the single server-controlled finalization path (confirm_session).
    # Settlement and trust application check this to avoid re-running on repeated
    # or concurrent requests (ISSUE-001 / ISSUE-002 / ISSUE-007).
    finalized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Confirmation(Base):
    __tablename__ = "confirmations"
    # A user confirms a given session at most once. The application already
    # checks this before insert, but that check-then-insert is racy under
    # concurrent requests — the constraint is the actual guarantee
    # (ISSUE-026).
    __table_args__ = (UniqueConstraint("barter_session_id", "user_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    confirmed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Wallet(Base):
    __tablename__ = "wallets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False)
    available_balance: Mapped[int] = mapped_column(Integer, default=999999, nullable=False)
    locked_balance: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_earned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_spent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Escrow(Base):
    __tablename__ = "escrows"
    # At most one *locked* escrow per (session, user) — a released/refunded/
    # penalized row doesn't count, so a session can still show settlement
    # history without blocking a later re-lock. Backs up lock_escrow's
    # idempotency check, which alone is racy under concurrent requests
    # (ISSUE-026, ISSUE-007).
    __table_args__ = (
        Index(
            "ix_escrows_locked_unique",
            "barter_session_id",
            "user_id",
            unique=True,
            sqlite_where=text("status = 'locked'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    barter_session_id: Mapped[int] = mapped_column(ForeignKey("barter_sessions.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="locked", nullable=False)
    locked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    release_type: Mapped[str | None] = mapped_column(String(20), nullable=True)


class CreditTransaction(Base):
    __tablename__ = "credit_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    barter_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("barter_sessions.id"), nullable=True
    )
    transaction_type: Mapped[str] = mapped_column(String(30), nullable=False)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    balance_after: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
