from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.policy import (
    ParticipantOutcome,
    PolicyConfig,
    calculate_escrow,
    floor_topup_amount,
    plan_settlement,
    regen_amount,
    regen_rate,
)

CFG = PolicyConfig()
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


# ── escrow sizing ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "trust,expected",
    [
        (0.0, 40),    # no trust -> full base stake
        (0.30, 28),   # spec example: new user
        (0.72, 11),   # spec example: experienced user
        (0.95, 5),    # clamped up to MIN
        (1.0, 5),     # clamped up to MIN
    ],
)
def test_calculate_escrow_matches_spec_examples(trust, expected):
    assert calculate_escrow(trust, CFG) == expected


def test_calculate_escrow_never_exceeds_max():
    assert calculate_escrow(-5.0, CFG) == CFG.max_escrow


def test_calculate_escrow_returns_int():
    assert isinstance(calculate_escrow(0.37, CFG), int)


# ── regeneration ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "trust,expected", [(0.9, 20), (0.81, 20), (0.6, 10), (0.51, 10), (0.5, 5), (0.1, 5)]
)
def test_regen_rate_bands(trust, expected):
    assert regen_rate(trust, CFG) == expected


def test_regen_amount_is_zero_for_fresh_account():
    assert regen_amount(NOW, NOW, 0.5, 10, CFG) == 0


def test_regen_amount_accrues_per_whole_day():
    last = NOW - timedelta(days=3)
    assert regen_amount(last, NOW, 0.5, 0, CFG) == 15  # 3 days * 5/day


def test_regen_amount_ignores_partial_days():
    last = NOW - timedelta(hours=47)
    assert regen_amount(last, NOW, 0.5, 0, CFG) == 5  # 1 whole day only


def test_regen_amount_is_capped_at_regen_cap():
    last = NOW - timedelta(days=100)
    assert regen_amount(last, NOW, 0.9, 0, CFG) == CFG.regen_cap


def test_regen_amount_is_zero_when_already_at_cap():
    last = NOW - timedelta(days=10)
    assert regen_amount(last, NOW, 0.9, CFG.regen_cap, CFG) == 0


def test_regen_amount_is_zero_when_never_regenerated():
    assert regen_amount(None, NOW, 0.5, 0, CFG) == 0


# ── participation floor ─────────────────────────────────────────────────────

def test_floor_topup_lifts_a_stranded_user_to_the_floor():
    assert floor_topup_amount(1, False, None, NOW, CFG) == 4  # 1 -> 5


def test_floor_topup_is_zero_above_the_floor():
    assert floor_topup_amount(CFG.floor, False, None, NOW, CFG) == 0


def test_floor_topup_is_blocked_by_an_active_reservation():
    assert floor_topup_amount(0, True, None, NOW, CFG) == 0


def test_floor_topup_is_blocked_inside_the_cooldown():
    last = NOW - timedelta(hours=CFG.floor_cooldown_hours - 1)
    assert floor_topup_amount(0, False, last, NOW, CFG) == 0


def test_floor_topup_is_allowed_after_the_cooldown():
    last = NOW - timedelta(hours=CFG.floor_cooldown_hours + 1)
    assert floor_topup_amount(0, False, last, NOW, CFG) == CFG.floor


# ── settlement ──────────────────────────────────────────────────────────────

def _participants(**overrides):
    base = dict(quality=0.9, engagement=0.8, no_show=False)
    base.update(overrides)
    return [
        ParticipantOutcome(user_id=1, stake=28, **base),
        ParticipantOutcome(user_id=2, stake=11, **base),
    ]


def test_full_release_returns_stakes_and_pays_bonuses():
    plan = plan_settlement("SUCCESSFUL", 0.92, _participants(), CFG)

    assert plan.mode == "FULL_RELEASE"
    user_1 = plan.breakdown[1]
    assert user_1["stake_returned"] == 28
    assert user_1["teaching_bonus"] == 5   # round(5 * 0.9)
    assert user_1["engagement_bonus"] == 2  # engagement 0.8 >= 0.6


def test_full_release_withholds_engagement_bonus_below_threshold():
    plan = plan_settlement("SUCCESSFUL", 0.92, _participants(engagement=0.4), CFG)
    assert plan.breakdown[1]["engagement_bonus"] == 0


def test_full_release_charges_platform_fee_on_the_pool():
    plan = plan_settlement("SUCCESSFUL", 0.92, _participants(), CFG)
    # pool = 28 + 11 = 39; 5% = 1.95 -> round -> 2
    assert sum(b["fee_share"] for b in plan.breakdown.values()) == 2


def test_partial_release_scales_with_qa_score():
    plan = plan_settlement("PARTIAL", 0.60, _participants(), CFG)

    assert plan.mode == "PARTIAL_RELEASE"
    assert plan.breakdown[1]["stake_returned"] == 17  # round(28 * 0.60)
    assert plan.breakdown[2]["stake_returned"] == 7   # round(11 * 0.60)


def test_dispute_is_a_penalty():
    plan = plan_settlement("DISPUTE", 0.2, _participants(), CFG)
    assert plan.mode == "PENALTY"


def test_low_qa_score_overrides_a_successful_verdict():
    plan = plan_settlement("SUCCESSFUL", 0.10, _participants(), CFG)
    assert plan.mode == "PENALTY"


def test_no_show_forfeits_the_stake_to_the_counterparty():
    participants = [
        ParticipantOutcome(user_id=1, stake=28, quality=0.0, engagement=0.0, no_show=True),
        ParticipantOutcome(user_id=2, stake=11, quality=0.9, engagement=0.8, no_show=False),
    ]
    plan = plan_settlement("DISPUTE", 0.0, participants, CFG)

    assert plan.mode == "PENALTY"
    assert plan.breakdown[1]["penalty"] == 28
    # the wronged party is made whole: own stake back + forfeited stake
    assert plan.breakdown[2]["net"] > 0


def test_mutual_dispute_returns_stakes_but_pays_no_bonuses():
    """No party is identifiably at fault, so nobody is penalised and nobody is
    rewarded. Forfeiture requires a no_show."""
    plan = plan_settlement("DISPUTE", 0.2, _participants(), CFG)

    assert plan.mode == "PENALTY"
    for user_id in (1, 2):
        row = plan.breakdown[user_id]
        assert row["stake_returned"] == row["stake"]
        assert row["penalty"] == 0
        assert row["teaching_bonus"] == 0
        assert row["engagement_bonus"] == 0
        assert row["net"] == 0


def test_mutual_no_show_forfeits_the_whole_pool_to_the_platform():
    participants = [
        ParticipantOutcome(user_id=1, stake=28, quality=0.0, engagement=0.0, no_show=True),
        ParticipantOutcome(user_id=2, stake=11, quality=0.0, engagement=0.0, no_show=True),
    ]
    plan = plan_settlement("DISPUTE", 0.0, participants, CFG)

    platform = sum(m.amount for m in plan.movements if m.account == "platform_revenue")
    assert platform == 39
    assert plan.breakdown[1]["net"] == -28
    assert plan.breakdown[2]["net"] == -11


def test_settlement_movements_always_balance():
    for verdict, qa in [("SUCCESSFUL", 0.95), ("PARTIAL", 0.6), ("DISPUTE", 0.1)]:
        plan = plan_settlement(verdict, qa, _participants(), CFG)
        assert sum(m.amount for m in plan.movements) == 0, verdict


@hyp_settings(max_examples=300)
@given(
    verdict=st.sampled_from(["SUCCESSFUL", "PARTIAL", "DISPUTE"]),
    qa=st.floats(min_value=0.0, max_value=1.0),
    stake_a=st.integers(min_value=5, max_value=40),
    stake_b=st.integers(min_value=5, max_value=40),
    quality=st.floats(min_value=0.0, max_value=1.0),
    engagement=st.floats(min_value=0.0, max_value=1.0),
    no_show_a=st.booleans(),
)
def test_property_every_plan_balances_and_uses_integers(
    verdict, qa, stake_a, stake_b, quality, engagement, no_show_a
):
    participants = [
        ParticipantOutcome(1, stake_a, quality, engagement, no_show_a),
        ParticipantOutcome(2, stake_b, quality, engagement, False),
    ]
    plan = plan_settlement(verdict, qa, participants, CFG)

    assert sum(m.amount for m in plan.movements) == 0
    assert all(isinstance(m.amount, int) for m in plan.movements)
    for row in plan.breakdown.values():
        assert all(isinstance(value, int) for value in row.values())


@hyp_settings(max_examples=200)
@given(
    verdict=st.sampled_from(["SUCCESSFUL", "PARTIAL", "DISPUTE"]),
    qa=st.floats(min_value=0.0, max_value=1.0),
    stake_a=st.integers(min_value=5, max_value=40),
    stake_b=st.integers(min_value=5, max_value=40),
)
def test_property_locked_accounts_are_fully_drained(verdict, qa, stake_a, stake_b):
    """Settlement must empty both locked accounts — no credits stranded."""
    participants = [
        ParticipantOutcome(1, stake_a, 0.8, 0.7, False),
        ParticipantOutcome(2, stake_b, 0.8, 0.7, False),
    ]
    plan = plan_settlement(verdict, qa, participants, CFG)

    drained = {
        1: sum(m.amount for m in plan.movements if m.account == "user_locked:1"),
        2: sum(m.amount for m in plan.movements if m.account == "user_locked:2"),
    }
    assert drained[1] == -stake_a
    assert drained[2] == -stake_b
