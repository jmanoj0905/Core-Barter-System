import json
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException

# ── Terminal colours ────────────────────────────────────────────────────────
_R = "\033[0;31m"
_G = "\033[0;32m"
_Y = "\033[1;33m"
_C = "\033[0;36m"
_M = "\033[0;35m"
_B = "\033[1;34m"
_W = "\033[1;37m"
_NC = "\033[0m"
_BOLD = "\033[1m"


def _ok(msg):
    print(f"  {_G}✓{_NC}  {msg}", flush=True)


def _info(msg):
    print(f"  {_C}→{_NC}  {msg}", flush=True)


def _warn(msg):
    print(f"  {_Y}⚠{_NC}  {_Y}{msg}{_NC}", flush=True)


_VERDICT_FMT = {
    "SUCCESSFUL": f"{_G}{_BOLD}✓ SUCCESSFUL{_NC}",
    "PARTIAL": f"{_Y}{_BOLD}~ PARTIAL   {_NC}",
    "DISPUTE": f"{_R}{_BOLD}✗ DISPUTE   {_NC}",
}


def _verdict(barter_id, verdict_type, duration_pass, confirmation_pass):
    label = _VERDICT_FMT.get(verdict_type, verdict_type)
    dur = f"{_G}pass{_NC}" if duration_pass else f"{_R}fail{_NC}"
    conf = f"{_G}pass{_NC}" if confirmation_pass else f"{_R}fail{_NC}"
    print(f"\n  {_BOLD}[Barter {barter_id}  VERDICT]{_NC}", flush=True)
    print(f"  {label}   duration={dur}  confirmation={conf}", flush=True)


def _trust(barter_id, u1_before, u1_after, u2_before, u2_after):
    d1 = u1_after - u1_before
    d2 = u2_after - u2_before
    arrow1 = f"{_G}↑{_NC}" if d1 >= 0 else f"{_R}↓{_NC}"
    arrow2 = f"{_G}↑{_NC}" if d2 >= 0 else f"{_R}↓{_NC}"
    print(f"\n  {_BOLD}[Barter {barter_id}  TRUST UPDATE]{_NC}", flush=True)
    print(
        f"  User 1:  {u1_before:.3f}  {arrow1}  {_BOLD}{u1_after:.3f}{_NC}  (Δ{d1:+.3f})",
        flush=True,
    )
    print(
        f"  User 2:  {u2_before:.3f}  {arrow2}  {_BOLD}{u2_after:.3f}{_NC}  (Δ{d2:+.3f})",
        flush=True,
    )


from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import (
    BarterSession,
    Confirmation,
    EngagementScoreLog,
    SessionContract,
    TranscriptSegment,
    User,
    Verdict,
    VideoEngagementResult,
    Warning,
    WindowResult,
)
from app.schemas import (
    ConfirmRequest,
    DriftSummaryRequest,
    EngagementLogRequest,
    EscrowReleaseRequest,
    FrameCheckRequest,
    SessionCreateRequest,
    SettlementRequest,
    TerminateRequest,
    TranscriptSegmentRequest,
    VideoEngagementRequest,
    WarningLogRequest,
    WindowResultRequest,
)
from app.websocket import manager

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared finalization helpers
#
# One server-controlled policy for turning session evidence into a verdict,
# a settlement, and a trust update — used by confirm_session (the normal
# completion path) and reused by the standalone /verdict and /trust endpoints
# so both paths agree and neither can double-apply money or trust
# (ISSUE-001, ISSUE-002, ISSUE-003, ISSUE-018).
# ---------------------------------------------------------------------------


def _elapsed_seconds(session: BarterSession) -> float | None:
    if not session.started_at:
        return None
    end_time = session.ended_at or datetime.now(timezone.utc)
    started_at = session.started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    return (end_time - started_at).total_seconds()


def _duration_pass(session: BarterSession, contract: SessionContract | None) -> bool:
    if not contract:
        return False
    elapsed = _elapsed_seconds(session)
    if elapsed is None:
        return False
    return elapsed >= contract.agreed_duration_seconds * 0.8


async def _evaluate_topic_quality(db: AsyncSession, barter_id: int) -> dict:
    """Read actual monitoring evidence for this session directly from the
    window/warning tables, instead of trusting the verdict's cached
    on_topic_percentage (which can be stale or, with zero windows, wrongly
    read as 100% — ISSUE-019)."""
    windows_result = await db.execute(
        select(WindowResult).where(WindowResult.barter_session_id == barter_id)
    )
    windows = windows_result.scalars().all()
    total = len(windows)
    has_evidence = total > 0
    on_topic = sum(1 for w in windows if w.classification in ("correct", "weakly_correct"))
    on_topic_percentage = round(100.0 * on_topic / total, 2) if has_evidence else 0.0

    severe_result = await db.execute(
        select(Warning)
        .where(Warning.barter_session_id == barter_id, Warning.severity == "severe")
        .limit(1)
    )
    has_severe_warning = severe_result.scalar_one_or_none() is not None

    return {
        "has_evidence": has_evidence,
        "on_topic_percentage": on_topic_percentage,
        "has_severe_warning": has_severe_warning,
    }


def _decide_verdict_type(
    terminated: bool, duration_pass: bool, confirmation_pass: bool, topic: dict
) -> str:
    """Documented quality policy (ISSUE-003): topic monitoring evidence can
    veto an otherwise-complete session, and never counts missing evidence as
    proof of good behavior (ISSUE-019)."""
    if terminated:
        return "DISPUTE"

    topic_failed = topic["has_evidence"] and (
        topic["on_topic_percentage"] < 40 or topic["has_severe_warning"]
    )
    if topic_failed:
        return "DISPUTE"

    topic_ok_for_success = not topic["has_evidence"] or topic["on_topic_percentage"] >= 70
    topic_ok_for_partial = not topic["has_evidence"] or topic["on_topic_percentage"] >= 40

    if duration_pass and confirmation_pass and topic_ok_for_success:
        return "SUCCESSFUL"
    if (duration_pass or confirmation_pass) and topic_ok_for_partial:
        return "PARTIAL"
    return "DISPUTE"


async def _upsert_verdict_checks(
    db: AsyncSession,
    barter_id: int,
    verdict: Verdict | None,
    verdict_type: str,
    duration_pass: bool,
    confirmation_pass: bool,
    actual_duration_seconds: float | None = None,
) -> Verdict:
    if verdict:
        verdict.verdict_type = verdict_type
        verdict.duration_check = str(duration_pass).lower()
        verdict.confirmation_check = str(confirmation_pass).lower()
        verdict.actual_duration_seconds = actual_duration_seconds
    else:
        verdict = Verdict(
            barter_session_id=barter_id,
            verdict_type=verdict_type,
            on_topic_percentage=0.0,
            warning_count=0,
            duration_check=str(duration_pass).lower(),
            confirmation_check=str(confirmation_pass).lower(),
            trust_delta_user1=0.0,
            trust_delta_user2=0.0,
            actual_duration_seconds=actual_duration_seconds,
        )
        db.add(verdict)
    await db.flush()
    return verdict


async def _apply_finalization_trust(
    db: AsyncSession, session: BarterSession, contract: SessionContract, settlement: dict
) -> dict:
    """Apply the settlement's role-specific trust deltas exactly once, to the
    contract's actual teacher/learner (not a fixed formula recomputed per
    endpoint — ISSUE-018)."""
    u1 = (await db.execute(select(User).where(User.id == session.user1_id))).scalar_one()
    u2 = (await db.execute(select(User).where(User.id == session.user2_id))).scalar_one()

    delta_by_user = {
        contract.teacher_user_id: settlement["teacher_trust_delta"],
        contract.learner_user_id: settlement["learner_trust_delta"],
    }

    u1_before, u2_before = u1.trust_score, u2.trust_score
    u1.trust_score = max(0.0, min(1.0, u1.trust_score + delta_by_user.get(session.user1_id, 0.0)))
    u2.trust_score = max(0.0, min(1.0, u2.trust_score + delta_by_user.get(session.user2_id, 0.0)))

    _trust(session.id, u1_before, u1.trust_score, u2_before, u2.trust_score)

    return {
        "trust_delta_user1": round(u1.trust_score - u1_before, 4),
        "trust_delta_user2": round(u2.trust_score - u2_before, 4),
    }


# ---------------------------------------------------------------------------
# Session Lifecycle
# ---------------------------------------------------------------------------


@router.get("/session/{barter_id}/contract")
async def get_session_contract(barter_id: int, db: AsyncSession = Depends(get_db)):
    """Get contract details for a session."""
    result = await db.execute(
        select(SessionContract).where(SessionContract.barter_session_id == barter_id)
    )
    contract = result.scalar_one_or_none()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    return {
        "barter_id": contract.barter_session_id,
        "topic": contract.topic,
        "scope": contract.scope,
        "teacher_user_id": contract.teacher_user_id,
        "learner_user_id": contract.learner_user_id,
        "agreed_duration_seconds": contract.agreed_duration_seconds,
    }


@router.post("/session/create")
async def create_session(req: SessionCreateRequest, db: AsyncSession = Depends(get_db)):
    # Session participants must be the actual contract teacher/learner, not a
    # fixed Alice=1/Bob=2 pair — otherwise escrow/trust operations that key
    # off session.user1_id/user2_id silently diverge from the contract
    # (ISSUE-013).
    teacher = (
        await db.execute(select(User).where(User.id == req.teacher_user_id))
    ).scalar_one_or_none()
    if not teacher:
        raise HTTPException(status_code=404, detail=f"User {req.teacher_user_id} not found")
    learner = (
        await db.execute(select(User).where(User.id == req.learner_user_id))
    ).scalar_one_or_none()
    if not learner:
        raise HTTPException(status_code=404, detail=f"User {req.learner_user_id} not found")

    session = BarterSession(user1_id=req.teacher_user_id, user2_id=req.learner_user_id, status="proposed")
    db.add(session)
    await db.flush()

    contract = SessionContract(
        barter_session_id=session.id,
        topic=req.topic,
        scope=req.scope,
        agreed_duration_seconds=req.agreed_duration_minutes * 60,
        teacher_user_id=req.teacher_user_id,
        learner_user_id=req.learner_user_id,
    )
    db.add(contract)
    await db.flush()
    await db.commit()

    # Notify Person 3 (semantic analysis) — register contract
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            await client.post(
                f"{settings.SEMANTIC_URL}/session/{session.id}/contract",
                json={
                    "barter_id": session.id,
                    "topic": req.topic,
                    "scope": req.scope,
                    "teacher_user_id": req.teacher_user_id,
                    "learner_user_id": req.learner_user_id,
                },
            )
        except Exception:
            pass  # service may not be running yet

    # Notify Person 4 (warning engine) — init session
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            await client.post(
                f"{settings.WARNING_URL}/session/{session.id}/init",
                json={
                    "teacher_user_id": req.teacher_user_id,
                    "learner_user_id": req.learner_user_id,
                },
            )
        except Exception:
            pass

    _ok(
        f'Session created  barter={session.id}  topic="{req.topic}"  teacher={req.teacher_user_id}  learner={req.learner_user_id}'
    )

    return {
        "barter_id": session.id,
        "contract_id": contract.id,
        "status": session.status,
        "teacher_user_id": req.teacher_user_id,
        "learner_user_id": req.learner_user_id,
    }


@router.post("/session/{barter_id}/start")
async def start_session(barter_id: int, db: AsyncSession = Depends(get_db)):
    from app.escrow import lock_escrow
    from app.schemas import EscrowResponse

    result = await db.execute(select(BarterSession).where(BarterSession.id == barter_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status in ("completed", "terminated"):
        raise HTTPException(status_code=400, detail=f"Session already {session.status}")
    if session.status == "active":
        return {
            "barter_id": session.id,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "status": session.status,
        }

    contract_result = await db.execute(
        select(SessionContract).where(SessionContract.barter_session_id == barter_id)
    )
    contract = contract_result.scalar_one_or_none()
    if not contract:
        raise HTTPException(status_code=404, detail="Session contract not found")

    teacher_escrow = await lock_escrow(db, barter_id, contract.teacher_user_id)
    if not teacher_escrow:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient balance for escrow: user {contract.teacher_user_id}",
        )

    learner_escrow = await lock_escrow(db, barter_id, contract.learner_user_id)
    if not learner_escrow:
        # roll back teacher escrow already locked in this flush
        await db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient balance for escrow: user {contract.learner_user_id}",
        )

    session.status = "active"
    session.started_at = datetime.now(timezone.utc)
    await db.commit()

    t_amt = teacher_escrow.amount
    l_amt = learner_escrow.amount
    _ok(f"Session {session.id} started — escrow locked  teacher={t_amt}cr  learner={l_amt}cr")

    return {
        "barter_id": session.id,
        "started_at": session.started_at.isoformat(),
        "status": session.status,
        "teacher_escrow": EscrowResponse.model_validate(teacher_escrow).model_dump()
        if teacher_escrow
        else None,
        "learner_escrow": EscrowResponse.model_validate(learner_escrow).model_dump()
        if learner_escrow
        else None,
    }


@router.post("/session/{barter_id}/confirm")
async def confirm_session(barter_id: int, req: ConfirmRequest, db: AsyncSession = Depends(get_db)):
    from app.escrow import apply_settlement

    result = await db.execute(select(BarterSession).where(BarterSession.id == barter_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "active":
        raise HTTPException(
            status_code=400, detail=f"Session is {session.status}, cannot confirm"
        )

    contract_result = await db.execute(
        select(SessionContract).where(SessionContract.barter_session_id == barter_id)
    )
    contract = contract_result.scalar_one_or_none()
    if not contract:
        raise HTTPException(status_code=404, detail="Session contract not found")

    # Only the two actual contract participants can confirm — a supplied
    # user_id that isn't teacher/learner on this session cannot complete it
    # (ISSUE-006).
    participant_ids = {contract.teacher_user_id, contract.learner_user_id}
    if req.user_id not in participant_ids:
        raise HTTPException(
            status_code=403, detail="User is not a participant in this session"
        )

    existing = await db.execute(
        select(Confirmation).where(
            Confirmation.barter_session_id == barter_id,
            Confirmation.user_id == req.user_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="User already confirmed")

    confirmation = Confirmation(barter_session_id=barter_id, user_id=req.user_id)
    db.add(confirmation)
    try:
        await db.flush()
    except IntegrityError:
        # Lost a concurrent race to confirm — the unique constraint caught
        # what the earlier existence check couldn't (ISSUE-026).
        await db.rollback()
        raise HTTPException(status_code=400, detail="User already confirmed")

    all_confs = await db.execute(
        select(Confirmation).where(Confirmation.barter_session_id == barter_id)
    )
    confirmed_users = {c.user_id for c in all_confs.scalars().all()}
    both_confirmed = participant_ids <= confirmed_users

    if not both_confirmed:
        await db.commit()
        await manager.broadcast(barter_id, {
            "type": "peer_confirmed",
            "user_id": req.user_id,
            "message": "Other user marked complete",
        })
        return {
            "barter_id": barter_id,
            "confirmed_by": sorted(confirmed_users),
            "both_confirmed": False,
            "settlement": None,
        }

    # Both participants confirmed. Run finalization once, in order: freeze
    # the session, let analysis finish, generate the authoritative verdict
    # from actual evidence, settle escrow, apply trust once, commit, and only
    # then announce completion (ISSUE-001, ISSUE-003, ISSUE-009).
    session.status = "finalizing"
    session.ended_at = datetime.now(timezone.utc)
    await db.commit()

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            await client.post(f"{settings.AUDIO_URL}/session/{barter_id}/end")
        except Exception:
            pass

    duration_pass = _duration_pass(session, contract)
    topic = await _evaluate_topic_quality(db, barter_id)
    verdict_type = _decide_verdict_type(False, duration_pass, True, topic)

    verdict_result = await db.execute(
        select(Verdict).where(Verdict.barter_session_id == barter_id)
    )
    verdict = verdict_result.scalar_one_or_none()
    verdict = await _upsert_verdict_checks(
        db, barter_id, verdict, verdict_type, duration_pass, True,
        actual_duration_seconds=_elapsed_seconds(session),
    )

    qa_score = {"SUCCESSFUL": 1.0, "PARTIAL": 0.5}.get(verdict_type, 0.0)
    settlement_result = await apply_settlement(db, barter_id, qa_score)

    if "error" not in settlement_result and not verdict.finalized:
        trust_deltas = await _apply_finalization_trust(db, session, contract, settlement_result)
        verdict.trust_delta_user1 = trust_deltas["trust_delta_user1"]
        verdict.trust_delta_user2 = trust_deltas["trust_delta_user2"]
        verdict.finalized = True

    session.status = "completed"
    await db.commit()

    _verdict(barter_id, verdict_type, duration_pass, True)

    await manager.broadcast(barter_id, {
        "type": "both_confirmed",
        "barter_id": barter_id,
        "confirmed_by": sorted(confirmed_users),
    })

    return {
        "barter_id": barter_id,
        "confirmed_by": sorted(confirmed_users),
        "both_confirmed": True,
        "settlement": settlement_result,
    }


@router.get("/session/{barter_id}/status")
async def session_status(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(BarterSession).where(BarterSession.id == barter_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    contract_result = await db.execute(
        select(SessionContract).where(SessionContract.barter_session_id == barter_id)
    )
    contract = contract_result.scalar_one_or_none()
    agreed_minutes = (contract.agreed_duration_seconds // 60) if contract else 0

    elapsed_minutes = 0.0
    if session.started_at:
        end_time = session.ended_at or datetime.now(timezone.utc)
        # Ensure both datetimes are timezone-aware before subtracting
        started_at = session.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        elapsed_minutes = round((end_time - started_at).total_seconds() / 60, 1)

    confs = await db.execute(
        select(Confirmation).where(Confirmation.barter_session_id == barter_id)
    )
    both_confirmed = len(confs.scalars().all()) >= 2

    return {
        "barter_id": barter_id,
        "status": session.status,
        "elapsed_minutes": elapsed_minutes,
        "agreed_duration_minutes": agreed_minutes,
        "can_complete": session.status == "active",
        "both_confirmed": both_confirmed,
    }


@router.post("/session/{barter_id}/terminate")
async def terminate_session(
    barter_id: int,
    req: TerminateRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    from app.escrow import get_escrows_by_session, release_escrow

    result = await db.execute(select(BarterSession).where(BarterSession.id == barter_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status in ("completed", "terminated"):
        raise HTTPException(status_code=400, detail=f"Session already {session.status}")

    session.status = "terminated"
    session.ended_at = datetime.now(timezone.utc)

    # Manual/automatic termination isn't a QA outcome, so it doesn't run the
    # success/penalty settlement policy — it just returns both deposits.
    # Only still-locked escrows are touched, so repeated termination calls
    # don't double-refund (ISSUE-005).
    escrows = await get_escrows_by_session(db, barter_id)
    for escrow in escrows:
        if escrow.status == "locked":
            await release_escrow(db, escrow.id, "refund", 0)

    await db.commit()

    reason = req.reason if req else "Manual termination"
    await manager.broadcast(barter_id, {
        "type": "terminated",
        "barter_id": barter_id,
        "reason": reason,
    })

    return {"barter_id": barter_id, "status": "terminated"}


# ---------------------------------------------------------------------------
# Window Results (receives from warning engine, stores in DB + broadcasts)
# ---------------------------------------------------------------------------


@router.post("/window/result")
async def log_window_result(req: WindowResultRequest, db: AsyncSession = Depends(get_db)):
    window = WindowResult(
        barter_session_id=req.barter_id,
        window_number=req.window_id,
        classification=req.classification,
        cosine_similarity=req.similarity_score,
        text_content=req.text_preview,
    )
    db.add(window)
    await db.commit()

    await manager.broadcast(
        req.barter_id,
        {
            "type": "window",
            "window_id": req.window_id,
            "classification": req.classification,
            "similarity": round(req.similarity_score, 3),
            "text_preview": req.text_preview[:120],
        },
    )

    return {"window_db_id": window.id}


@router.get("/session/{barter_id}/windows")
async def get_windows(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WindowResult)
        .where(WindowResult.barter_session_id == barter_id)
        .order_by(WindowResult.window_number)
    )
    windows = result.scalars().all()
    return [
        {
            "window_id": w.window_number,
            "classification": w.classification,
            "similarity": round(w.cosine_similarity, 3),
            "text_preview": w.text_content,
            "created_at": w.created_at.isoformat(),
        }
        for w in windows
    ]


# ---------------------------------------------------------------------------
# Video Engagement Results
# ---------------------------------------------------------------------------


@router.post("/session/{barter_id}/video-engagement")
async def store_video_engagement(
    barter_id: int, req: VideoEngagementRequest, db: AsyncSession = Depends(get_db)
):
    row = VideoEngagementResult(
        barter_session_id=barter_id,
        user_id=req.user_id,
        window_start=req.window_start,
        window_end=req.window_end,
        video_attention_score=req.video_attention_score,
        backend_used=req.backend_used,
        raw_signals=json.dumps(req.raw_signals),
    )
    db.add(row)
    await db.commit()
    return {"status": "stored", "id": row.id}


@router.get("/session/{barter_id}/video-engagement")
async def list_video_engagement(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(VideoEngagementResult)
        .where(VideoEngagementResult.barter_session_id == barter_id)
        .order_by(VideoEngagementResult.window_start)
    )
    rows = result.scalars().all()
    return [
        {
            "user_id": r.user_id,
            "window_start": r.window_start,
            "window_end": r.window_end,
            "video_attention_score": r.video_attention_score,
            "backend_used": r.backend_used,
            "raw_signals": json.loads(r.raw_signals),
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@router.post("/session/{barter_id}/engagement-log")
async def store_engagement_log(
    barter_id: int, req: EngagementLogRequest, db: AsyncSession = Depends(get_db)
):
    row = EngagementScoreLog(
        barter_session_id=barter_id,
        user_id=req.user_id,
        speech_engagement_score=req.speech_engagement_score,
        video_attention_score=req.video_attention_score,
        fused_engagement_score=req.fused_engagement_score,
        created_at=datetime.now(timezone.utc),
    )
    db.add(row)
    await db.commit()
    return {"status": "stored", "id": row.id}


@router.get("/session/{barter_id}/engagement")
async def get_latest_engagement(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(EngagementScoreLog)
        .where(EngagementScoreLog.barter_session_id == barter_id)
        .order_by(EngagementScoreLog.created_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="No engagement data for this session")
    return {
        "user_id": row.user_id,
        "speech_engagement_score": row.speech_engagement_score,
        "video_attention_score": row.video_attention_score,
        "fused_engagement_score": row.fused_engagement_score,
        "created_at": row.created_at.isoformat(),
    }


@router.get("/session/{barter_id}/engagement/history")
async def get_engagement_history(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(EngagementScoreLog)
        .where(EngagementScoreLog.barter_session_id == barter_id)
        .order_by(EngagementScoreLog.created_at)
    )
    rows = result.scalars().all()
    return [
        {
            "user_id": r.user_id,
            "speech_engagement_score": r.speech_engagement_score,
            "video_attention_score": r.video_attention_score,
            "fused_engagement_score": r.fused_engagement_score,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Transcript Storage (receives from audio pipeline)
# ---------------------------------------------------------------------------


@router.post("/session/{barter_id}/transcript")
async def save_transcript_segment(
    barter_id: int, req: TranscriptSegmentRequest, db: AsyncSession = Depends(get_db)
):
    segment = TranscriptSegment(
        barter_session_id=barter_id,
        user_id=req.user_id,
        text=req.text,
        duration_seconds=req.duration_seconds,
        timestamp_start=req.timestamp_start,
        timestamp_end=req.timestamp_end,
    )
    db.add(segment)
    await db.commit()

    await manager.broadcast(
        barter_id,
        {
            "type": "transcript",
            "user_id": req.user_id,
            "speaker": "Alice" if req.user_id == 1 else "Bob",
            "text": req.text,
        },
    )

    return {"id": segment.id}


@router.get("/session/{barter_id}/transcript")
async def get_transcript(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TranscriptSegment)
        .where(TranscriptSegment.barter_session_id == barter_id)
        .order_by(TranscriptSegment.created_at)
    )
    segments = result.scalars().all()
    return [
        {
            "user_id": s.user_id,
            "speaker": "Alice" if s.user_id == 1 else "Bob",
            "text": s.text,
            "duration_seconds": s.duration_seconds,
            "timestamp_start": s.timestamp_start,
            "created_at": s.created_at.isoformat(),
        }
        for s in segments
    ]


# ---------------------------------------------------------------------------
# Warning Relay (receives from Person 4, stores in DB)
# ---------------------------------------------------------------------------


@router.post("/warnings/log")
async def log_warning(req: WarningLogRequest, db: AsyncSession = Depends(get_db)):
    warning = Warning(
        barter_session_id=req.barter_id,
        severity=req.severity,
        message=req.reason,
        window_ids=req.window_ids,
    )
    db.add(warning)
    await db.commit()

    await manager.broadcast(
        req.barter_id,
        {
            "warning_id": warning.id,
            "barter_id": req.barter_id,
            "severity": req.severity,
            "reason": req.reason,
            "window_ids": req.window_ids,
            "timestamp": req.timestamp,
        },
    )

    _sev = {
        "mild": f"\033[1;33m⚠  MILD\033[0m",
        "strong": f"\033[0;35m⚠⚠ STRONG\033[0m",
        "severe": f"\033[0;31m🚨 SEVERE\033[0m",
    }.get(req.severity, req.severity)
    print(
        f"  \033[1m[Barter {req.barter_id}  WARNING → WebSocket]\033[0m  {_sev}  {req.reason}",
        flush=True,
    )

    return {"warning_id": warning.id}


# ---------------------------------------------------------------------------
# Drift Summary (receives from Person 4 at session end)
# ---------------------------------------------------------------------------


@router.post("/session/{barter_id}/drift-summary")
async def receive_drift_summary(
    barter_id: int, req: DriftSummaryRequest, db: AsyncSession = Depends(get_db)
):
    drift_json = json.dumps(req.model_dump())

    # Zero analyzed windows means no evidence, not a clean 100% — reporting
    # it as full marks would treat missing data as demonstrated accuracy
    # (ISSUE-019). The raw total_windows=0 is still preserved in drift_json.
    on_topic_percentage = (
        round(100.0 - req.percent_incorrect, 2) if req.total_windows > 0 else 0.0
    )

    # Create or update verdict row with drift data
    existing = await db.execute(select(Verdict).where(Verdict.barter_session_id == barter_id))
    verdict = existing.scalar_one_or_none()

    if verdict:
        verdict.drift_summary = drift_json
        verdict.on_topic_percentage = on_topic_percentage
        verdict.warning_count = req.warning_count
    else:
        verdict = Verdict(
            barter_session_id=barter_id,
            verdict_type="PENDING",
            on_topic_percentage=on_topic_percentage,
            warning_count=req.warning_count,
            duration_check="pending",
            confirmation_check="pending",
            trust_delta_user1=0.0,
            trust_delta_user2=0.0,
            drift_summary=drift_json,
        )
        db.add(verdict)

    await db.flush()
    await db.commit()

    return {"status": "received"}


# ---------------------------------------------------------------------------
# Engagement Summary (receives from semantic analysis at session end)
# ---------------------------------------------------------------------------


@router.post("/session/{barter_id}/engagement-summary")
async def receive_engagement_summary(barter_id: int, req: dict, db: AsyncSession = Depends(get_db)):
    """Store learner engagement summary inside the verdict's drift_summary JSON."""
    existing = await db.execute(select(Verdict).where(Verdict.barter_session_id == barter_id))
    verdict = existing.scalar_one_or_none()

    if verdict:
        drift = json.loads(verdict.drift_summary) if verdict.drift_summary else {}
        drift["engagement"] = req
        verdict.drift_summary = json.dumps(drift)
    else:
        verdict = Verdict(
            barter_session_id=barter_id,
            verdict_type="PENDING",
            on_topic_percentage=0.0,
            warning_count=0,
            duration_check="pending",
            confirmation_check="pending",
            trust_delta_user1=0.0,
            trust_delta_user2=0.0,
            drift_summary=json.dumps({"engagement": req}),
        )
        db.add(verdict)

    await db.commit()
    return {"status": "received"}


# ---------------------------------------------------------------------------
# QA Verdict Generation
# ---------------------------------------------------------------------------


@router.post("/verdict/{barter_id}/generate")
async def generate_verdict(barter_id: int, db: AsyncSession = Depends(get_db)):
    session_result = await db.execute(select(BarterSession).where(BarterSession.id == barter_id))
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    existing = await db.execute(select(Verdict).where(Verdict.barter_session_id == barter_id))
    verdict = existing.scalar_one_or_none()

    # Already finalized (settled + trust applied) by confirm_session — the
    # verdict is terminal, so return it rather than recomputing (ISSUE-010).
    if verdict and verdict.finalized:
        return {
            "verdict": verdict.verdict_type,
            "duration_check": verdict.duration_check == "true",
            "confirmation_check": verdict.confirmation_check == "true",
        }

    contract_result = await db.execute(
        select(SessionContract).where(SessionContract.barter_session_id == barter_id)
    )
    contract = contract_result.scalar_one_or_none()

    duration_pass = _duration_pass(session, contract)

    confs = await db.execute(
        select(Confirmation).where(Confirmation.barter_session_id == barter_id)
    )
    confirmation_pass = len(confs.scalars().all()) >= 2

    terminated = session.status == "terminated"
    topic = await _evaluate_topic_quality(db, barter_id)
    verdict_type = _decide_verdict_type(terminated, duration_pass, confirmation_pass, topic)

    verdict = await _upsert_verdict_checks(
        db, barter_id, verdict, verdict_type, duration_pass, confirmation_pass,
        actual_duration_seconds=_elapsed_seconds(session),
    )
    await db.commit()

    _verdict(barter_id, verdict_type, duration_pass, confirmation_pass)

    return {
        "verdict": verdict_type,
        "duration_check": duration_pass,
        "confirmation_check": confirmation_pass,
    }


@router.get("/verdict/{barter_id}")
async def get_verdict(barter_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Verdict).where(Verdict.barter_session_id == barter_id))
    verdict = result.scalar_one_or_none()
    if not verdict:
        raise HTTPException(status_code=404, detail="Verdict not found")

    drift = json.loads(verdict.drift_summary) if verdict.drift_summary else None

    return {
        "barter_id": barter_id,
        "verdict": verdict.verdict_type,
        "duration_check": verdict.duration_check == "true",
        "confirmation_check": verdict.confirmation_check == "true",
        "on_topic_percentage": verdict.on_topic_percentage,
        "warning_count": verdict.warning_count,
        "trust_delta_user1": verdict.trust_delta_user1,
        "trust_delta_user2": verdict.trust_delta_user2,
        "actual_duration_seconds": verdict.actual_duration_seconds,
        "drift_summary": drift,
    }


# ---------------------------------------------------------------------------
# Trust Score Update
# ---------------------------------------------------------------------------


@router.post("/trust/{barter_id}/update")
async def update_trust(barter_id: int, db: AsyncSession = Depends(get_db)):
    session_result = await db.execute(select(BarterSession).where(BarterSession.id == barter_id))
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    verdict_result = await db.execute(select(Verdict).where(Verdict.barter_session_id == barter_id))
    verdict = verdict_result.scalar_one_or_none()
    if not verdict:
        raise HTTPException(status_code=404, detail="Verdict not found. Generate verdict first.")

    u1 = (await db.execute(select(User).where(User.id == session.user1_id))).scalar_one()
    u2 = (await db.execute(select(User).where(User.id == session.user2_id))).scalar_one()

    if verdict.finalized:
        # Trust was already applied exactly once, inside confirm_session's
        # finalization. Repeated or concurrent results-page loads must read
        # the stored result, not mutate trust again (ISSUE-002).
        u1_after, u2_after = u1.trust_score, u2.trust_score
        u1_before = round(u1_after - verdict.trust_delta_user1, 4)
        u2_before = round(u2_after - verdict.trust_delta_user2, 4)
        return {
            "user_1_trust": {"before": u1_before, "after": round(u1_after, 4)},
            "user_2_trust": {"before": u2_before, "after": round(u2_after, 4)},
        }

    # Legacy path: a verdict generated outside confirm-driven finalization
    # (e.g. a manually terminated session). Applies the QA-score trust
    # formula once, then locks it so this endpoint stays idempotent too.
    qa_scores = {"SUCCESSFUL": 1.0, "PARTIAL": 0.5, "DISPUTE": 0.0}
    qa_score = qa_scores.get(verdict.verdict_type, 0.0)

    # satisfaction_rating hardcoded to 4/5 for POC
    quality_adjusted = (qa_score + 0.8) / 2

    u1_before = u1.trust_score
    u2_before = u2.trust_score

    # new_trust = (previous_trust * 0.3) + (quality_adjusted * 0.7), clamped [0, 1]
    u1.trust_score = max(0.0, min(1.0, (u1.trust_score * 0.3) + (quality_adjusted * 0.7)))
    u2.trust_score = max(0.0, min(1.0, (u2.trust_score * 0.3) + (quality_adjusted * 0.7)))

    verdict.trust_delta_user1 = round(u1.trust_score - u1_before, 4)
    verdict.trust_delta_user2 = round(u2.trust_score - u2_before, 4)
    verdict.finalized = True

    await db.commit()

    _trust(barter_id, u1_before, u1.trust_score, u2_before, u2.trust_score)

    return {
        "user_1_trust": {"before": round(u1_before, 4), "after": round(u1.trust_score, 4)},
        "user_2_trust": {"before": round(u2_before, 4), "after": round(u2.trust_score, 4)},
    }


# ---------------------------------------------------------------------------
# Session Safety Monitor — NSFW Video Frame Check
# ---------------------------------------------------------------------------


@router.post("/safety/check-frame")
async def check_video_frame(req: FrameCheckRequest):
    """Check a video frame for NSFW content using nudenet (local inference).

    Called by the frontend at ~0.1 FPS (every 10 seconds).
    If NSFW detected, forwards a safety alert to the warning engine.
    """
    from app.safety import check_frame

    result = check_frame(req.image_base64)
    if result:
        # Forward safety alert to warning engine
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                await client.post(
                    f"{settings.WARNING_URL}/safety/alert",
                    json={
                        "barter_id": req.barter_id,
                        "user_id": req.user_id,
                        "warning_type": "nsfw",
                        "details": result,
                    },
                )
            except Exception:
                pass  # warning engine may not be running

    return {"checked": True, "flagged": result is not None}


# ---------------------------------------------------------------------------
# Escrow & Credit System
# ---------------------------------------------------------------------------


@router.get("/wallet/{user_id}")
async def get_wallet(user_id: int, db: AsyncSession = Depends(get_db)):
    from app.escrow import get_or_create_wallet
    from app.models import User

    wallet = await get_or_create_wallet(db, user_id)
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    trust_score = user.trust_score if user else 1.0
    await db.commit()

    return {
        "id": wallet.id,
        "user_id": wallet.user_id,
        "available_balance": wallet.available_balance,
        "locked_balance": wallet.locked_balance,
        "total_earned": wallet.total_earned,
        "total_spent": wallet.total_spent,
        "trust_score": trust_score,
    }


@router.post("/escrow/lock")
async def lock_escrow_endpoint(barter_id: int, db: AsyncSession = Depends(get_db)):
    from app.escrow import lock_escrow, get_or_create_wallet
    from app.schemas import EscrowResponse

    session_result = await db.execute(select(BarterSession).where(BarterSession.id == barter_id))
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    teacher_escrow = await lock_escrow(db, barter_id, session.user1_id)
    learner_escrow = await lock_escrow(db, barter_id, session.user2_id)

    await db.commit()

    return {
        "teacher_escrow": EscrowResponse.model_validate(teacher_escrow).model_dump()
        if teacher_escrow
        else None,
        "learner_escrow": EscrowResponse.model_validate(learner_escrow).model_dump()
        if learner_escrow
        else None,
    }


@router.get("/escrow/{barter_id}")
async def get_session_escrows(barter_id: int, db: AsyncSession = Depends(get_db)):
    from app.escrow import get_escrows_by_session
    from app.schemas import EscrowResponse

    escrows = await get_escrows_by_session(db, barter_id)
    return [EscrowResponse.model_validate(e).model_dump() for e in escrows]


@router.post("/escrow/release")
async def release_escrow_endpoint(req: EscrowReleaseRequest, db: AsyncSession = Depends(get_db)):
    from app.escrow import release_escrow
    from app.schemas import EscrowResponse

    try:
        escrow = await release_escrow(db, req.escrow_id, req.release_type, req.penalty_amount)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await db.commit()

    if not escrow:
        raise HTTPException(status_code=404, detail="Escrow not found or already released")

    return EscrowResponse.model_validate(escrow).model_dump()


@router.post("/settlement/{barter_id}")
async def settle_session(
    barter_id: int, req: SettlementRequest, db: AsyncSession = Depends(get_db)
):
    from app.escrow import apply_settlement
    from app.schemas import SettlementResponse
    from app.models import User

    result = await apply_settlement(db, barter_id, req.qa_score)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    contract_result = await db.execute(
        select(SessionContract).where(SessionContract.barter_session_id == barter_id)
    )
    contract = contract_result.scalar_one_or_none()

    if contract:
        teacher_result = await db.execute(select(User).where(User.id == contract.teacher_user_id))
        teacher = teacher_result.scalar_one_or_none()
        learner_result = await db.execute(select(User).where(User.id == contract.learner_user_id))
        learner = learner_result.scalar_one_or_none()

        if teacher:
            teacher.trust_score = max(
                0.0, min(1.0, teacher.trust_score + result["teacher_trust_delta"])
            )
        if learner:
            learner.trust_score = max(
                0.0, min(1.0, learner.trust_score + result["learner_trust_delta"])
            )

    await db.commit()

    return SettlementResponse(
        provider_escrow_released=result["teacher_escrow_released"],
        learner_escrow_released=result["learner_escrow_released"],
        provider_bonus=result["teacher_bonus"],
        learner_refund=result["learner_escrow_released"]
        if result["release_type"] == "penalty"
        else 0,
        provider_trust_delta=result["teacher_trust_delta"],
        learner_trust_delta=result["learner_trust_delta"],
    )


@router.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    users = result.scalars().all()
    return [
        {"id": u.id, "username": u.username, "trust_score": round(u.trust_score, 4)}
        for u in users
    ]


@router.get("/transactions/{user_id}")
async def get_user_transactions(user_id: int, db: AsyncSession = Depends(get_db)):
    from app.escrow import get_transactions_by_user
    from app.schemas import CreditTransactionResponse

    transactions = await get_transactions_by_user(db, user_id)
    return [CreditTransactionResponse.model_validate(t).model_dump() for t in transactions]
