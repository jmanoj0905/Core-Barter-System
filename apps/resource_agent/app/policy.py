"""Pure credit policy. No I/O, no database, no imports from app.models.

Every function here is deterministic given its arguments, which is what makes
the property tests in tests/test_policy.py meaningful.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Sequence

from app.ledger import Movement


def _round_half_up(value: float) -> int:
    """Round to the nearest integer, ties away from zero.

    Python's built-in round() is half-to-even, so round(4.5) == 4 — a
    surprise in a credit breakdown, where a user reading "5 x 0.9" expects 5.
    Every float-to-integer conversion in the money path goes through here so
    the rule is one function, stated once.
    """
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class PolicyConfig:
    base_escrow: int = 40
    min_escrow: int = 5
    max_escrow: int = 40
    initial_grant: int = 100
    regen_cap: int = 100
    floor: int = 5
    floor_cooldown_hours: int = 24
    teaching_bonus_base: int = 5
    engagement_bonus: int = 2
    engagement_threshold: float = 0.6
    platform_fee_pct: float = 0.05
    platform_fee_min: int = 1
    no_show_penalty: int = 20
    reserve_ttl_hours: int = 24
    full_release_qa: float = 0.85
    partial_release_qa: float = 0.5

    @classmethod
    def from_env(cls) -> "PolicyConfig":
        def _int(name: str, default: int) -> int:
            return int(os.environ.get(name, default))

        def _float(name: str, default: float) -> float:
            return float(os.environ.get(name, default))

        return cls(
            base_escrow=_int("BASE_ESCROW", 40),
            min_escrow=_int("MIN_ESCROW", 5),
            max_escrow=_int("MAX_ESCROW", 40),
            initial_grant=_int("INITIAL_GRANT", 100),
            regen_cap=_int("REGEN_CAP", 100),
            floor=_int("FLOOR", 5),
            floor_cooldown_hours=_int("FLOOR_COOLDOWN_HOURS", 24),
            teaching_bonus_base=_int("TEACHING_BONUS_BASE", 5),
            engagement_bonus=_int("ENGAGEMENT_BONUS", 2),
            engagement_threshold=_float("ENGAGEMENT_THRESHOLD", 0.6),
            platform_fee_pct=_float("PLATFORM_FEE_PCT", 0.05),
            platform_fee_min=_int("PLATFORM_FEE_MIN", 1),
            no_show_penalty=_int("NO_SHOW_PENALTY", 20),
            reserve_ttl_hours=_int("RESERVE_TTL_HOURS", 24),
            full_release_qa=_float("FULL_RELEASE_QA", 0.85),
            partial_release_qa=_float("PARTIAL_RELEASE_QA", 0.5),
        )


@dataclass(frozen=True)
class ParticipantOutcome:
    user_id: int
    stake: int
    quality: float
    engagement: float
    no_show: bool


@dataclass(frozen=True)
class SettlementPlan:
    mode: str
    movements: tuple[Movement, ...]
    breakdown: dict[int, dict[str, int]]


# ── escrow sizing ───────────────────────────────────────────────────────────

def calculate_escrow(trust_score: float, cfg: PolicyConfig) -> int:
    """Asymmetric stake: less trust means more skin in the game."""
    raw = _round_half_up(cfg.base_escrow * (1 - trust_score))
    return int(max(cfg.min_escrow, min(cfg.max_escrow, raw)))


# ── regeneration ────────────────────────────────────────────────────────────

def regen_rate(trust_score: float, cfg: PolicyConfig) -> int:
    if trust_score > 0.8:
        return 20
    if trust_score > 0.5:
        return 10
    return 5


def regen_amount(
    last_regen_at: datetime | None,
    now: datetime,
    trust_score: float,
    current_balance: int,
    cfg: PolicyConfig,
) -> int:
    """Credits owed since last_regen_at, capped so the balance never exceeds
    regen_cap. Only whole elapsed days accrue."""
    if last_regen_at is None:
        return 0

    elapsed_days = int((now - last_regen_at).total_seconds() // 86400)
    if elapsed_days <= 0:
        return 0

    owed = elapsed_days * regen_rate(trust_score, cfg)
    headroom = cfg.regen_cap - current_balance
    return int(max(0, min(owed, headroom)))


# ── participation floor ─────────────────────────────────────────────────────

def floor_topup_amount(
    available: int,
    has_active_reservation: bool,
    last_topup_at: datetime | None,
    now: datetime,
    cfg: PolicyConfig,
) -> int:
    """The death-spiral guard: lift a stranded user back to the floor.

    Blocked while an escrow is reserved (so it cannot be farmed mid-session)
    and while inside the cooldown window.
    """
    if available >= cfg.floor:
        return 0
    if has_active_reservation:
        return 0
    if last_topup_at is not None:
        elapsed_hours = (now - last_topup_at).total_seconds() / 3600
        if elapsed_hours < cfg.floor_cooldown_hours:
            return 0
    return int(cfg.floor - available)


# ── settlement ──────────────────────────────────────────────────────────────

def _mode_for(verdict_type: str, qa_score: float, cfg: PolicyConfig) -> str:
    if verdict_type == "DISPUTE" or qa_score < cfg.partial_release_qa:
        return "PENALTY"
    if verdict_type == "SUCCESSFUL" and qa_score >= cfg.full_release_qa:
        return "FULL_RELEASE"
    return "PARTIAL_RELEASE"


def _platform_fee(pool: int, cfg: PolicyConfig) -> int:
    if pool <= 0:
        return 0
    return int(max(cfg.platform_fee_min, _round_half_up(pool * cfg.platform_fee_pct)))


def plan_settlement(
    verdict_type: str,
    qa_score: float,
    participants: Sequence[ParticipantOutcome],
    cfg: PolicyConfig,
) -> SettlementPlan:
    """Turn a verdict into balanced credit movements.

    Invariant: the returned movements always sum to zero, and every locked
    account is drained by exactly the participant's stake.
    """
    mode = _mode_for(verdict_type, qa_score, cfg)
    movements: list[Movement] = []
    breakdown: dict[int, dict[str, int]] = {}

    pool = sum(p.stake for p in participants)
    fee = _platform_fee(pool, cfg) if mode in ("FULL_RELEASE", "PARTIAL_RELEASE") else 0
    fee_shares = _split_fee(fee, participants)

    forfeited_total = 0
    forfeited_by: dict[int, int] = {}

    # Pass 1: drain locked accounts, decide each participant's returned stake.
    for participant in participants:
        user_id = participant.user_id
        movements.append(Movement(f"user_locked:{user_id}", -participant.stake))

        if mode == "PENALTY" and participant.no_show:
            returned = 0
        elif mode == "PENALTY":
            returned = participant.stake
        elif mode == "FULL_RELEASE":
            returned = participant.stake
        else:
            returned = _round_half_up(participant.stake * qa_score)

        forfeited = participant.stake - returned
        if mode == "PENALTY":
            forfeited_by[user_id] = forfeited
            forfeited_total += forfeited

        breakdown[user_id] = {
            "stake": participant.stake,
            "stake_returned": returned,
            "teaching_bonus": 0,
            "engagement_bonus": 0,
            "penalty": forfeited if mode == "PENALTY" else 0,
            "fee_share": fee_shares.get(user_id, 0),
            "compensation": 0,
            "net": 0,
        }

    # Pass 2: bonuses (release modes only).
    if mode in ("FULL_RELEASE", "PARTIAL_RELEASE"):
        scale = 1.0 if mode == "FULL_RELEASE" else qa_score
        for participant in participants:
            row = breakdown[participant.user_id]
            row["teaching_bonus"] = _round_half_up(
                cfg.teaching_bonus_base * participant.quality * scale
            )
            row["engagement_bonus"] = (
                cfg.engagement_bonus
                if participant.engagement >= cfg.engagement_threshold
                else 0
            )

    # Pass 3: penalty compensation — forfeited stakes go to the wronged parties.
    if mode == "PENALTY" and forfeited_total > 0:
        wronged = [p for p in participants if forfeited_by.get(p.user_id, 0) == 0]
        if wronged:
            shares = _split_evenly(forfeited_total, [p.user_id for p in wronged])
            for user_id, share in shares.items():
                breakdown[user_id]["compensation"] = share
        else:
            # Everyone forfeited: the pool goes to the platform.
            movements.append(Movement("platform_revenue", forfeited_total))

    # Pass 4: emit credit movements and compute net change per participant.
    minted = 0
    for participant in participants:
        user_id = participant.user_id
        row = breakdown[user_id]
        credited = (
            row["stake_returned"]
            + row["teaching_bonus"]
            + row["engagement_bonus"]
            + row["compensation"]
            - row["fee_share"]
        )
        if credited:
            movements.append(Movement(f"user_available:{user_id}", credited))
        row["net"] = credited - row["stake"]
        minted += row["teaching_bonus"] + row["engagement_bonus"]

    if fee:
        movements.append(Movement("platform_revenue", fee))
    if minted:
        movements.append(Movement("platform_mint", -minted))

    # Remainder of the pool not returned to anyone (partial release) goes to
    # platform revenue, which is also what balances the entry.
    residual = -sum(m.amount for m in movements)
    if residual:
        movements.append(Movement("platform_revenue", residual))

    return SettlementPlan(mode=mode, movements=tuple(movements), breakdown=breakdown)


def _split_evenly(total: int, user_ids: Sequence[int]) -> dict[int, int]:
    """Split `total` across user_ids, giving remainders to the earliest ids."""
    if not user_ids:
        return {}
    base, remainder = divmod(total, len(user_ids))
    return {
        user_id: base + (1 if index < remainder else 0)
        for index, user_id in enumerate(user_ids)
    }


def _split_fee(fee: int, participants: Sequence[ParticipantOutcome]) -> dict[int, int]:
    if fee <= 0:
        return {}
    return _split_evenly(fee, [p.user_id for p in participants])
