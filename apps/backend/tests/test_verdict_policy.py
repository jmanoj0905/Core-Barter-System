import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.routes import _decide_verdict_type  # noqa: E402


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
