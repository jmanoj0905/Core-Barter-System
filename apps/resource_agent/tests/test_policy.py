import math
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

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


def _round_half_up(value: float) -> int:
    """Independent reimplementation of the policy's rounding rule, so tests
    that check *what* is rounded don't also have to trust *how* it's rounded."""
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _expected_fee(pool: int, cfg: PolicyConfig) -> int:
    if pool <= 0:
        return 0
    return max(cfg.platform_fee_min, _round_half_up(pool * cfg.platform_fee_pct))


def _expected_split(total: int, user_ids) -> dict:
    """Test-local reimplementation of the documented `_split_evenly` contract:
    split total evenly, remainder to the earliest ids. Reimplemented rather
    than imported so a misdirected remainder in the app is actually caught,
    not just re-checked against itself."""
    user_ids = list(user_ids)
    if not user_ids:
        return {}
    base, remainder = divmod(total, len(user_ids))
    return {
        user_id: base + (1 if index < remainder else 0)
        for index, user_id in enumerate(user_ids)
    }


def _expected_teaching_bonus(participant, mode: str, qa_score: float, cfg: PolicyConfig) -> int:
    if mode == "FULL_RELEASE":
        scale = 1.0
    elif mode == "PARTIAL_RELEASE":
        scale = qa_score
    else:
        return 0
    return _round_half_up(cfg.teaching_bonus_base * participant.quality * scale)


def _expected_engagement_bonus(participant, mode: str, cfg: PolicyConfig) -> int:
    if mode not in ("FULL_RELEASE", "PARTIAL_RELEASE"):
        return 0
    return cfg.engagement_bonus if participant.engagement >= cfg.engagement_threshold else 0


def _expected_fee_shares(pool: int, participants, cfg: PolicyConfig, mode: str) -> dict:
    if mode not in ("FULL_RELEASE", "PARTIAL_RELEASE"):
        return {p.user_id: 0 for p in participants}
    fee = _expected_fee(pool, cfg)
    if fee <= 0:
        return {p.user_id: 0 for p in participants}
    return _expected_split(fee, [p.user_id for p in participants])


def _expected_compensation(participants, mode: str) -> dict:
    """Forfeited stakes (PENALTY, no-show only) split across the participants
    who did show up; zero for everyone if nobody or everybody no-showed."""
    if mode != "PENALTY":
        return {p.user_id: 0 for p in participants}

    forfeited_by = {p.user_id: (p.stake if p.no_show else 0) for p in participants}
    forfeited_total = sum(forfeited_by.values())
    if forfeited_total <= 0:
        return {p.user_id: 0 for p in participants}

    wronged = [p for p in participants if forfeited_by[p.user_id] == 0]
    if not wronged:
        return {p.user_id: 0 for p in participants}

    shares = _expected_split(forfeited_total, [p.user_id for p in wronged])
    return {p.user_id: shares.get(p.user_id, 0) for p in participants}


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


@pytest.mark.parametrize(
    "qa,expected_mode",
    [
        (CFG.partial_release_qa, "PARTIAL_RELEASE"),  # exactly at the penalty floor
        (
            math.nextafter(CFG.partial_release_qa, -math.inf),
            "PENALTY",
        ),  # the smallest float strictly below the penalty floor
        (CFG.full_release_qa, "FULL_RELEASE"),  # exactly at the full-release ceiling
        (
            math.nextafter(CFG.full_release_qa, -math.inf),
            "PARTIAL_RELEASE",
        ),  # the smallest float strictly below the full-release ceiling
    ],
)
def test_mode_boundaries_are_pinned_at_the_config_thresholds(qa, expected_mode):
    plan = plan_settlement("SUCCESSFUL", qa, _participants(), CFG)
    assert plan.mode == expected_mode


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
    no_show_b=st.booleans(),
)
def test_property_every_plan_balances_and_uses_integers(
    verdict, qa, stake_a, stake_b, quality, engagement, no_show_a, no_show_b
):
    participants = [
        ParticipantOutcome(1, stake_a, quality, engagement, no_show_a),
        ParticipantOutcome(2, stake_b, quality, engagement, no_show_b),
    ]
    plan = plan_settlement(verdict, qa, participants, CFG)

    assert sum(m.amount for m in plan.movements) == 0
    assert all(isinstance(m.amount, int) for m in plan.movements)
    for row in plan.breakdown.values():
        assert all(isinstance(value, int) for value in row.values())

    # The residual-to-platform_revenue line makes the zero-sum assertion above
    # true by construction for *any* per-participant numbers, so it cannot
    # catch a wrong bonus, fee, or compensation split. Reconcile each
    # participant's breakdown row against the movements it actually produced,
    # computed independently of the row itself, to close that gap.
    credited = {p.user_id: 0 for p in participants}
    for movement in plan.movements:
        if movement.account.startswith("user_available:"):
            user_id = int(movement.account.split(":", 1)[1])
            credited[user_id] += movement.amount

    for participant in participants:
        row = plan.breakdown[participant.user_id]
        expected_credit = (
            row["stake_returned"]
            + row["teaching_bonus"]
            + row["engagement_bonus"]
            + row["compensation"]
            - row["fee_share"]
        )
        assert credited[participant.user_id] == expected_credit
        assert row["net"] == expected_credit - row["stake"]

    # The reconciliation above only proves the movements agree with the
    # breakdown row — both descend from the same computed value, so a wrong
    # bonus or a misdirected remainder would sail through unnoticed. Pin
    # every non-stake field to a value computed test-side from the raw
    # inputs and the documented policy rules, independently of the row.
    pool = stake_a + stake_b
    expected_fee_shares = _expected_fee_shares(pool, participants, CFG, plan.mode)
    expected_compensation = _expected_compensation(participants, plan.mode)

    for participant in participants:
        row = plan.breakdown[participant.user_id]
        assert row["teaching_bonus"] == _expected_teaching_bonus(
            participant, plan.mode, qa, CFG
        )
        assert row["engagement_bonus"] == _expected_engagement_bonus(participant, plan.mode, CFG)
        assert row["fee_share"] == expected_fee_shares[participant.user_id]
        assert row["compensation"] == expected_compensation[participant.user_id]

    # The platform fee is policy, not an incidental byproduct of the residual:
    # pin the aggregate too, per mode.
    total_fee_share = sum(row["fee_share"] for row in plan.breakdown.values())
    if plan.mode in ("FULL_RELEASE", "PARTIAL_RELEASE"):
        assert total_fee_share == _expected_fee(pool, CFG)
    else:
        assert total_fee_share == 0


@hyp_settings(max_examples=200)
@given(
    verdict=st.sampled_from(["SUCCESSFUL", "PARTIAL", "DISPUTE"]),
    qa=st.floats(min_value=0.0, max_value=1.0),
    stake_a=st.integers(min_value=5, max_value=40),
    stake_b=st.integers(min_value=5, max_value=40),
    no_show_a=st.booleans(),
    no_show_b=st.booleans(),
)
def test_property_locked_accounts_are_fully_drained(
    verdict, qa, stake_a, stake_b, no_show_a, no_show_b
):
    """Settlement must empty both locked accounts — no credits stranded."""
    participants = [
        ParticipantOutcome(1, stake_a, 0.8, 0.7, no_show_a),
        ParticipantOutcome(2, stake_b, 0.8, 0.7, no_show_b),
    ]
    plan = plan_settlement(verdict, qa, participants, CFG)

    drained = {
        1: sum(m.amount for m in plan.movements if m.account == "user_locked:1"),
        2: sum(m.amount for m in plan.movements if m.account == "user_locked:2"),
    }
    assert drained[1] == -stake_a
    assert drained[2] == -stake_b
