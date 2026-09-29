import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from sqlalchemy import select, text  # noqa: E402

from app.models import BarterSession, Warning, WindowResult  # noqa: E402
from app.routes import _decide_verdict_type, _evaluate_topic_quality  # noqa: E402


def topic(pct, evidence=True, non_advisory_severe=0):
    """Build the dict `_evaluate_topic_quality` hands `_decide_verdict_type`.

    `non_advisory_severe_count` defaults to 0 so the existing percentage-band
    cases stay about the bands alone. Drift-severe warnings are written
    advisory, so they never land in this count; safety-severe warnings (a
    toxicity/NSFW hard block) are written non-advisory and do.
    """
    return {
        "has_evidence": evidence,
        "on_topic_percentage": pct,
        "non_advisory_severe_count": non_advisory_severe,
    }


def test_severe_warning_no_longer_vetoes_a_good_session():
    # The bug: one severe warning cost the teacher the whole escrow. A drift
    # severe warning is advisory, so it never reaches this count at all.
    assert _decide_verdict_type(False, True, True, topic(85.0)) == "SUCCESSFUL"


def test_non_advisory_severe_warning_still_vetoes():
    # A safety-severe (toxicity/NSFW hard block) is written non-advisory and
    # keeps its payout consequence, however good the on-topic numbers are.
    assert (
        _decide_verdict_type(False, True, True, topic(95.0, non_advisory_severe=1))
        == "DISPUTE"
    )


def test_non_advisory_severe_vetoes_without_window_evidence():
    # /safety/alert writes no WindowResult, so a hard-blocked session can have
    # no window evidence at all. The veto must not be gated on has_evidence,
    # or a hard block on a session with no windows would escape it.
    assert (
        _decide_verdict_type(
            False, True, True, topic(0.0, evidence=False, non_advisory_severe=1)
        )
        == "DISPUTE"
    )


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
    assert (
        _decide_verdict_type(True, True, True, topic(95.0, non_advisory_severe=1))
        == "DISPUTE"
    )


@pytest.mark.asyncio
async def test_topic_quality_counts_non_advisory_severe_warnings(backend_client, db_session):
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
            Warning(
                barter_session_id=barter_id,
                severity="severe",
                message="off topic",
                advisory=False,
            ),
            Warning(
                barter_session_id=barter_id,
                severity="severe",
                message="off topic again",
                advisory=False,
            ),
        ]
    )
    await db_session.commit()

    result = await _evaluate_topic_quality(db_session, barter_id)
    # Both seeded severe warnings are non-advisory, so the count is still 2 —
    # the same number the old `severe_warning_count` reported. What changed is
    # what the key means: it now counts only the warnings that still veto.
    assert result["non_advisory_severe_count"] == 2
    assert "has_severe_warning" not in result
    assert "severe_warning_count" not in result
    assert result["has_evidence"] is True
    assert result["on_topic_percentage"] == 66.67


@pytest.mark.asyncio
async def test_topic_quality_excludes_advisory_severe_warnings(backend_client, db_session):
    """A mix of advisory and non-advisory severe rows, plus a NULL row.

    `advisory` can hold 1, 0 or NULL: SQLite backfills 0 on rows that predate
    the column, and NULL is still reachable through other write paths. Both 0
    and NULL must read as not-advisory (see the comment on Warning.advisory),
    which a naive `advisory == False` filter would get wrong for NULL under SQL
    three-valued logic.
    """
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
            # drift-severe, written advisory by run_warning_decision
            Warning(
                barter_session_id=barter_id,
                severity="severe",
                message="3 consecutive off-topic",
                advisory=True,
            ),
            Warning(
                barter_session_id=barter_id,
                severity="severe",
                message="4 consecutive off-topic",
                advisory=True,
            ),
            # safety-severe, written non-advisory
            Warning(
                barter_session_id=barter_id,
                severity="severe",
                message="hard block: toxicity",
                advisory=False,
            ),
            # this one is forced to NULL below — must read as not-advisory
            Warning(
                barter_session_id=barter_id,
                severity="severe",
                message="null advisory row",
            ),
            # a non-severe warning never counts, whatever its advisory value
            Warning(
                barter_session_id=barter_id,
                severity="strong",
                message="2 consecutive off-topic",
                advisory=False,
            ),
        ]
    )
    await db_session.commit()

    # Passing advisory=None to the ORM does NOT store NULL — SQLAlchemy reads
    # an unset/None attribute as "apply the column default", so the row lands
    # as 0. A genuine NULL therefore has to be written as raw SQL, which is
    # also how it would arrive in production: a write path outside this model.
    await db_session.execute(
        text("UPDATE warnings SET advisory = NULL WHERE message = 'null advisory row'")
    )
    await db_session.commit()

    stored = (
        await db_session.execute(
            select(Warning.advisory).where(
                Warning.barter_session_id == barter_id,
                Warning.message == "null advisory row",
            )
        )
    ).scalar_one()
    assert stored is None, "the NULL-advisory row must really be NULL in the database"

    result = await _evaluate_topic_quality(db_session, barter_id)
    # the safety row + the NULL row; the two advisory rows are excluded
    assert result["non_advisory_severe_count"] == 2
    assert _decide_verdict_type(False, True, True, result) == "DISPUTE"


@pytest.mark.asyncio
async def test_a_null_advisory_severe_warning_alone_vetoes(backend_client, db_session):
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
                cosine_similarity=0.95,
            ),
            Warning(
                barter_session_id=barter_id,
                severity="severe",
                message="legacy row, advisory unknown",
            ),
        ]
    )
    await db_session.commit()

    # a genuine NULL, written the only way one can arrive: outside the ORM
    # default (see test_topic_quality_excludes_advisory_severe_warnings)
    await db_session.execute(
        text(
            "UPDATE warnings SET advisory = NULL "
            "WHERE message = 'legacy row, advisory unknown'"
        )
    )
    await db_session.commit()
    assert (
        await db_session.execute(
            select(Warning.advisory).where(
                Warning.barter_session_id == barter_id,
                Warning.message == "legacy row, advisory unknown",
            )
        )
    ).scalar_one() is None

    result = await _evaluate_topic_quality(db_session, barter_id)
    assert result["non_advisory_severe_count"] == 1
    assert result["on_topic_percentage"] == 100.0
    assert _decide_verdict_type(False, True, True, result) == "DISPUTE"


@pytest.mark.asyncio
async def test_advisory_only_session_still_settles(backend_client, db_session):
    """The original bug, end to end through the real query.

    Every severe warning is advisory drift and the on-topic percentage is
    healthy, so the session settles SUCCESSFUL and the escrow is released.
    """
    session = BarterSession(user1_id=1, user2_id=2, status="active")
    db_session.add(session)
    await db_session.flush()
    barter_id = session.id

    db_session.add_all(
        [
            WindowResult(
                barter_session_id=barter_id,
                window_number=n,
                classification="correct",
                cosine_similarity=0.9,
            )
            for n in range(1, 10)
        ]
        + [
            WindowResult(
                barter_session_id=barter_id,
                window_number=10,
                classification="incorrect",
                cosine_similarity=0.05,
            ),
            Warning(
                barter_session_id=barter_id,
                severity="severe",
                message="3 consecutive off-topic",
                advisory=True,
            ),
        ]
    )
    await db_session.commit()

    result = await _evaluate_topic_quality(db_session, barter_id)
    assert result["non_advisory_severe_count"] == 0
    assert result["on_topic_percentage"] == 90.0
    assert _decide_verdict_type(False, True, True, result) == "SUCCESSFUL"


@pytest.mark.asyncio
async def test_rolling_deploy_drift_severe_vetoes_and_that_is_accepted(
    backend_client, db_session
):
    """Documented, accepted consequence of the rolling deploy — not a bug.

    An old warning_engine posts /warnings/log with no `advisory` field, so the
    row persists as 0 (non-advisory) and therefore vetoes, reproducing the
    original bug for the duration of the deploy window. Every drift-severe row
    already in the database from before this branch behaves the same way. There
    is no way to tell them apart from a safety-severe after the fact, and
    inventing a heuristic (reason-string sniffing, timestamp windows) would be
    worse than the bounded exposure. This test pins the behaviour so that it is
    known rather than discovered.
    """
    session = BarterSession(user1_id=1, user2_id=2, status="active")
    db_session.add(session)
    await db_session.flush()
    barter_id = session.id
    db_session.add(
        WindowResult(
            barter_session_id=barter_id,
            window_number=1,
            classification="correct",
            cosine_similarity=0.9,
        )
    )
    await db_session.commit()

    # exactly what an old warning_engine sends: no "advisory" key at all
    resp = await backend_client.post(
        "/warnings/log",
        json={
            "barter_id": barter_id,
            "severity": "severe",
            "reason": "3 consecutive off-topic windows",
        },
    )
    assert resp.status_code == 200

    result = await _evaluate_topic_quality(db_session, barter_id)
    assert result["non_advisory_severe_count"] == 1
    assert _decide_verdict_type(False, True, True, result) == "DISPUTE"
