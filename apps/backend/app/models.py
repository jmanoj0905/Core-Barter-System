from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, String, Text, DateTime
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
    trust_delta_user1: Mapped[float] = mapped_column(Float, nullable=False)
    trust_delta_user2: Mapped[float] = mapped_column(Float, nullable=False)
    drift_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Confirmation(Base):
    __tablename__ = "confirmations"

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
