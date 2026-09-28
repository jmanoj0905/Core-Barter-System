import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.models import BarterSession, Warning, WindowResult  # noqa: E402
from app.routes import _decide_verdict_type, _evaluate_topic_quality  # noqa: E402


def topic(pct, evidence=True):
    return {"has_evidence": evidence, "on_topic_percentage": pct}


def test_severe_warning_no_longer_vetoes_a_good_session():
    # The bug: one severe warning cost the teacher the whole escrow.
    assert _decide_verdict_type(False, True, True, topic(85.0)) == "SUCCESSFUL"


def test_sustained_drift_still_disputes():
    assert _decide_verdict_type(False, True, True, topic(35.0)) == "DISPUTE"


def test_boundary_forty_is_not_a_dispute():
    assert _decide_verdict_type(False, True, True, topic(40.0)) == "PARTIAL"


def test_boundary_seventy_is_successful():
    assert _decide_verdict_type(False, True, True, topic(70.0)) == "SUCCESSFUL"


def test_zero_window_session_behaves_exactly_as_before():
    # ISSUE-019 guarantees only that missing evidence never MANUFACTURES a
    # DISPUTE. A zero-window session that passed duration and both
    # confirmations does return SUCCESSFUL today; Half A must not change
    # either direction. Both are pinned so neither can drift.
    assert _decide_verdict_type(False, True, True, topic(0.0, evidence=False)) == "SUCCESSFUL"
    assert _decide_verdict_type(False, False, False, topic(0.0, evidence=False)) == "DISPUTE"


def test_termination_still_dominates():
    assert _decide_verdict_type(True, True, True, topic(95.0)) == "DISPUTE"


@pytest.mark.asyncio
async def test_topic_quality_counts_severe_warnings(backend_client, db_session):
    session = BarterSession(user1_id=1, user2_id=2, status="active")
    db_session.add(session)
    await db_session.flush()
    barter_id = session.id

    db_session.add_all(
        [
            WindowResult(
                barter_session_id=barter_id,
                window_number=1,
                classification="correct",
                cosine_similarity=0.9,
            ),
            WindowResult(
                barter_session_id=barter_id,
                window_number=2,
                classification="correct",
                cosine_similarity=0.9,
            ),
            WindowResult(
                barter_session_id=barter_id,
                window_number=3,
                classification="incorrect",
                cosine_similarity=0.1,
            ),
            Warning(barter_session_id=barter_id, severity="severe", message="off topic"),
            Warning(barter_session_id=barter_id, severity="severe", message="off topic again"),
        ]
    )
    await db_session.commit()

    result = await _evaluate_topic_quality(db_session, barter_id)
    assert result["severe_warning_count"] == 2
    assert "has_severe_warning" not in result
    assert result["has_evidence"] is True
    assert result["on_topic_percentage"] == 66.67
