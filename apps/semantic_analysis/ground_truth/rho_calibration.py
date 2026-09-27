"""Calibrate — or reject — RHO in the relative threshold `thr = RHO * R`.

GROUND TRUTH ONLY. Nothing here ships. `thr = RHO * R` appears nowhere outside
this directory: production `classify()` keeps its fixed UPPER/LOWER behaviour,
and this module never imports or mutates `apps/semantic_analysis/main.py`.

The mechanism (spec 7.1), pinned exactly and deliberately not "improved":

    R = UPPER                      # 0.36, the floor R starts at
    for each window i:
        thr_i = rho * R            # R as it stood BEFORE window i could update it
        if sim_i >= thr_i:         # only an at-or-above-threshold window may raise R
            R = min(R_CAP, max(R, sim_i))

So the first threshold of every session is always `rho * 0.36` — at rho=0.45
that is 0.162, the handoff's own stated degradation figure and the reason this
reading was chosen. A window *below* the current thr may never raise R, however
high its cosine: a digression must not raise the ceiling it is judged against.
Spec 7.1 states that this is a choice, not a deduction; the variant in which any
window may update R is a different mechanism and is not implemented here.

Note an arithmetic consequence worth recording: because `thr = rho * R` with
rho < 1, thr is always strictly below R, so any window with `sim >= R` also has
`sim >= thr`. The below-threshold guard therefore only ever *bites* for rho > 1
and is a no-op across the swept grid (0.20-0.80). It is implemented and tested
as written regardless, because it is the mechanism's stated semantics.

Parameters in scope: exactly three — RHO, R's cap (0.80), and that update rule.
Nothing here touches the escalation ladder, mass policy, or `warning_engine`.

Scoring is the error mix, never accuracy (spec 7.2): a window is *predicted a
digression* when its cosine sits below its threshold. Gold labels come from
`eval_dataset.tools.gold_labels.gold_label`, which returns None for boundary
windows; those are excluded and counted in neither class. Every table carries
the control: a flat `LOWER = 0.14` on exactly the same windows.

Driver:  cd apps/semantic_analysis && ./venv/bin/python -m ground_truth.rho_calibration
"""

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval_dataset.tools.gold_labels import gold_label  # noqa: E402
from eval_dataset.tools.replay import replay, synthetic_durations  # noqa: E402
from eval_dataset.tools.script_parser import parse_script  # noqa: E402

# Production constants, read from the service module so they cannot drift.
_WINDOWING_PATH = _REPO_ROOT / "apps" / "semantic_analysis" / "windowing.py"


def _load_windowing():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_rho_windowing", _WINDOWING_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_windowing = _load_windowing()

UPPER = _windowing.UPPER  # 0.36 — R's floor and starting value
LOWER = _windowing.LOWER  # 0.14 — the flat control threshold
R_CAP = 0.80  # R is capped here; one of the three parameters in scope

# RHO swept 0.20-0.80 at 0.01 inclusive. Built with integer arithmetic so the
# keys are exactly two decimals (float arange yields 0.6100000000000001).
GRID = [round(n / 100, 2) for n in range(20, 81)]

CORPUS_DIR = _REPO_ROOT / "eval_dataset" / "scripts" / "calibration"

FOLD_FAMILIES = ("session", "topic", "author")

AUTHORS = ("agent:opus-5", "agent:sonnet-5", "agent:haiku-4.5", "real:annomi")

AGENT_AUTHORS = tuple(a for a in AUTHORS if a.startswith("agent:"))
REAL_AUTHORS = tuple(a for a in AUTHORS if a.startswith("real:"))

# The two categories with zero declared digressions. Spec 7.5 criterion 2 asks
# about the false-digression rate on on-topic material specifically, so the
# pooled mix is also reported restricted to these.
ON_TOPIC_CATEGORIES = ("abstract_on_topic", "depressed_baseline")

WER_NOT_RUN = {
    "status": "not_run",
    "targets_planned": [0, 10, 20, 30],
    "reason": (
        "The WER robustness leg was not run. The user declined the AWS Transcribe "
        "spend, and the sweep is additionally blocked: the S3 bucket "
        "eval_dataset/tools/transcribe.py requires belongs to another account. No "
        "real-duration transcripts exist (eval_dataset/transcripts/synthetic is "
        "empty), so durations=\"timed\" cannot be exercised. These are not zeros "
        "and must not be read as measurements."
    ),
    "measurements": None,
    "alignment_contract_tested": (
        "align_wer_runs() aligns two runs by ReplayWindow.index and reports window "
        "count mismatches; both behaviours are unit-tested on hand-built window "
        "lists in test_rho_calibration.py (a window emitted at WER-0 may be skipped "
        "at WER-30 for falling under MIN_CONTENT_TOKENS, so a positional comparison "
        "silently misaligns the runs)."
    ),
}


# --------------------------------------------------------------------------
# the mechanism
# --------------------------------------------------------------------------


def running_thresholds(similarities, rho: float) -> list[float]:
    """One threshold per window: thr_i = rho * R, with R as it stood before i.

    R starts at UPPER (0.36); a window updates R only if `sim >= rho * R_current`;
    R is capped at R_CAP (0.80) and never decreases.
    """
    thresholds: list[float] = []
    r = UPPER
    for similarity in similarities:
        thr = rho * r
        thresholds.append(thr)
        if similarity >= thr:
            r = min(R_CAP, max(r, similarity))
    return thresholds


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorMix:
    """The D2-shaped error mix. Accuracy is deliberately absent (spec 7.2)."""

    caught: int
    missed: int
    false_digressions: int
    on_topic_total: int
    digression_total: int
    excluded: int = 0

    @property
    def caught_rate(self) -> float:
        return self.caught / self.digression_total if self.digression_total else 0.0

    @property
    def false_digression_rate(self) -> float:
        return self.false_digressions / self.on_topic_total if self.on_topic_total else 0.0

    def __add__(self, other):
        return ErrorMix(
            self.caught + other.caught,
            self.missed + other.missed,
            self.false_digressions + other.false_digressions,
            self.on_topic_total + other.on_topic_total,
            self.digression_total + other.digression_total,
            self.excluded + other.excluded,
        )

    def as_dict(self) -> dict:
        return {
            "caught": self.caught,
            "missed": self.missed,
            "false_digressions": self.false_digressions,
            "on_topic_total": self.on_topic_total,
            "digression_total": self.digression_total,
            "excluded": self.excluded,
            "caught_rate": self.caught_rate,
            "false_digression_rate": self.false_digression_rate,
        }


ZERO_MIX = ErrorMix(0, 0, 0, 0, 0, 0)


@dataclass(frozen=True)
class ScoredSession:
    """A session's cosines, scored once and reused across the whole grid.

    Windows are embedded once (~380 windows across the 25 sessions); the 61 RHO
    values then sweep these cached numbers. Re-embedding per RHO would be 61x
    the work for identical results.
    """

    session_id: str
    topic: str
    scope: str
    category: str
    author: str
    durations: str
    window_indices: tuple[int, ...]
    similarities: tuple[float, ...]
    gold: tuple  # "on_topic" | "off_topic" | None, positionally aligned

    def fold_key(self, family: str) -> str:
        if family == "session":
            return self.session_id
        if family == "topic":
            return self.topic
        if family == "author":
            return self.author
        raise ValueError(
            f"unknown fold family {family!r}; the held-out unit is always the "
            f"session (R is per-session running state), so one of {FOLD_FAMILIES}"
        )


def _mix_for_thresholds(sessions, thresholds_per_session) -> ErrorMix:
    total = ZERO_MIX
    for session, thresholds in zip(sessions, thresholds_per_session):
        caught = missed = false_digressions = 0
        on_topic_total = digression_total = excluded = 0
        for similarity, thr, gold in zip(session.similarities, thresholds, session.gold):
            if gold is None:
                excluded += 1  # boundary window: neither class (spec 5.2)
                continue
            predicted_digression = similarity < thr
            if gold == "off_topic":
                digression_total += 1
                if predicted_digression:
                    caught += 1
                else:
                    missed += 1
            else:
                on_topic_total += 1
                if predicted_digression:
                    false_digressions += 1
        total = total + ErrorMix(
            caught, missed, false_digressions, on_topic_total, digression_total, excluded
        )
    return total


def _check_durations(sessions, durations: str) -> None:
    for session in sessions:
        if session.durations != durations:
            raise ValueError(
                f"{session.session_id} was scored with durations="
                f"{session.durations!r}, not {durations!r}. Only "
                '"synthetic" runs end-to-end: the WER/timed leg is not run '
                "(no real-duration transcripts exist)."
            )


def evaluate(sessions, rho: float, durations: str = "synthetic") -> ErrorMix:
    """Error mix for the relative threshold at `rho` over `sessions`.

    A `depressed_baseline` session whose R never leaves its floor is scored
    normally — an unmoved R is a measurement, not missing data.
    """
    _check_durations(sessions, durations)
    return _mix_for_thresholds(
        sessions, [running_thresholds(s.similarities, rho) for s in sessions]
    )


def evaluate_flat(sessions, threshold: float = LOWER, durations: str = "synthetic") -> ErrorMix:
    """The control: a flat threshold on exactly the same windows."""
    _check_durations(sessions, durations)
    return _mix_for_thresholds(
        sessions, [[threshold] * len(s.similarities) for s in sessions]
    )


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------

OBJECTIVE = {
    "primary": "informedness = caught_rate - false_digression_rate",
    "tie_break": "lower false_digression_rate, then lower RHO",
    "secondary_reported": (
        "asymmetric cost = false_digressions + 0.5 * missed, normalised by class "
        "totals; reported alongside but not used to pick the argmax"
    ),
    "rationale": (
        "D2's lesson: accuracy ties across a plateau while the error mix does not, "
        "and a false accusation is asymmetrically expensive. The tie-break therefore "
        "breaks toward fewer false digressions."
    ),
}


def informedness(mix: ErrorMix) -> float:
    return mix.caught_rate - mix.false_digression_rate


def asymmetric_cost(mix: ErrorMix) -> float:
    return mix.false_digression_rate + 0.5 * (1.0 - mix.caught_rate)


def accuracy(mix: ErrorMix) -> float:
    """Reported only for the objective-sensitivity block; never picks the argmax."""
    labellable = mix.on_topic_total + mix.digression_total
    if not labellable:
        return 0.0
    correct = mix.caught + (mix.on_topic_total - mix.false_digressions)
    return correct / labellable


def digression_f1(mix: ErrorMix) -> float:
    """F1 of the digression class. Reported only for objective sensitivity."""
    predicted = mix.caught + mix.false_digressions
    if not predicted or not mix.digression_total:
        return 0.0
    precision = mix.caught / predicted
    recall = mix.caught / mix.digression_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def weighted_informedness(mix: ErrorMix, weight: float) -> float:
    """caught_rate - weight * false_digression_rate. weight=1 is the primary."""
    return mix.caught_rate - weight * mix.false_digression_rate


# Every alternative is expressed as a score to MAXIMISE, so one argmax helper
# serves them all. The primary stays weight=1 informedness; these only measure
# how far the argmax moves under a different reading of the same error mix.
ALTERNATIVE_OBJECTIVES = {
    "informedness_primary": informedness,
    "committed_secondary_asymmetric_cost": lambda m: -asymmetric_cost(m),
    "accuracy": accuracy,
    "digression_class_f1": digression_f1,
    "weighted_informedness_w0.25": lambda m: weighted_informedness(m, 0.25),
    "weighted_informedness_w0.50": lambda m: weighted_informedness(m, 0.50),
    "weighted_informedness_w1.00": lambda m: weighted_informedness(m, 1.00),
    "weighted_informedness_w2.00": lambda m: weighted_informedness(m, 2.00),
    "weighted_informedness_w3.00": lambda m: weighted_informedness(m, 3.00),
}


def objective_sensitivity(by_rho: dict) -> dict:
    """Each alternative objective's argmax over the full-corpus grid.

    I-2: the committed secondary objective does not agree with the primary, and
    that disagreement must be on the record rather than left to prose. The
    primary is NOT changed by this block; only disclosed against.
    """
    out = {}
    for name, score in ALTERNATIVE_OBJECTIVES.items():
        best = min(GRID, key=lambda rho: (-score(by_rho[rho]), rho))
        out[name] = {
            "argmax": best,
            "value_at_its_argmax": score(by_rho[best]),
            "value_at_primary_argmax": score(by_rho[argmax_rho(by_rho)]),
        }
    out["_parameterisation"] = {
        "weighted_informedness_wK": "caught_rate - K * false_digression_rate",
        "note": (
            "K weights the FALSE-DIGRESSION side, so larger K is more "
            "false-accusation-averse and pulls the argmax down. A sweep "
            "parameterised the other way round (cost = false_digression_rate + "
            "W * (1 - caught_rate), i.e. weighting the MISSED side) maps as "
            "W = 1/K: W=0.25 -> K=4, W=0.5 -> K=2 (this is exactly the committed "
            "asymmetric_cost, argmax 0.20), W=1 -> K=1 (the primary, 0.26), "
            "W=2 -> K=0.5 (0.36). The two readings describe the same grid."
        ),
        "committed_secondary_asymmetric_cost": (
            "false_digression_rate + 0.5 * (1 - caught_rate); equivalent to "
            "weighted_informedness with K=2"
        ),
    }
    primary = out["informedness_primary"]["argmax"]
    secondary = out["committed_secondary_asymmetric_cost"]["argmax"]
    out["_summary"] = {
        "primary_argmax": primary,
        "committed_secondary_argmax": secondary,
        "primary_and_secondary_agree": primary == secondary,
        "distinct_argmaxes_across_objectives": sorted(
            {v["argmax"] for k, v in out.items() if not k.startswith("_")}
        ),
        "note": (
            "The primary objective (informedness, weight 1 on the "
            "false-digression rate) and the committed secondary objective "
            "(asymmetric_cost) DISAGREE on this corpus. Accuracy and "
            "digression-class F1 also disagree with the primary. The argmax is "
            "therefore a function of the objective choice as much as of the "
            "data, and the objective was chosen by this task rather than "
            "pre-registered in the spec. Read the argmax as conditional on it."
        ),
    }
    return out


def argmax_rho(mixes: dict) -> float:
    """Pick RHO by the pre-registered objective with its pre-registered tie-break."""
    return min(
        GRID,
        key=lambda rho: (
            -informedness(mixes[rho]),
            mixes[rho].false_digression_rate,
            rho,
        ),
    )


# --------------------------------------------------------------------------
# WER alignment (leg not run; the contract is still pinned)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WerAlignment:
    matched: tuple
    baseline_only: tuple
    degraded_only: tuple
    baseline_count: int
    degraded_count: int

    @property
    def count_mismatch(self) -> int:
        """degraded - baseline. Non-zero means the runs emitted different windows."""
        return self.degraded_count - self.baseline_count

    def as_dict(self) -> dict:
        return {
            "matched_indices": [b.index for b, _ in self.matched],
            "baseline_only": list(self.baseline_only),
            "degraded_only": list(self.degraded_only),
            "baseline_count": self.baseline_count,
            "degraded_count": self.degraded_count,
            "count_mismatch": self.count_mismatch,
        }


def align_wer_runs(baseline_windows, degraded_windows) -> WerAlignment:
    """Align two runs of the same script by `ReplayWindow.index`, never position.

    A window emitted at WER-0 can be skipped at WER-30 (its cleaned text fell
    under MIN_CONTENT_TOKENS) while still consuming a window index, so zipping
    the two lists positionally pairs unrelated windows and silently misaligns
    the comparison. Count mismatches are reported, never hidden.
    """
    baseline_by_index = {w.index: w for w in baseline_windows}
    degraded_by_index = {w.index: w for w in degraded_windows}
    shared = sorted(set(baseline_by_index) & set(degraded_by_index))
    return WerAlignment(
        matched=tuple((baseline_by_index[i], degraded_by_index[i]) for i in shared),
        baseline_only=tuple(sorted(set(baseline_by_index) - set(degraded_by_index))),
        degraded_only=tuple(sorted(set(degraded_by_index) - set(baseline_by_index))),
        baseline_count=len(baseline_windows),
        degraded_count=len(degraded_windows),
    )


# --------------------------------------------------------------------------
# grid search over folds
# --------------------------------------------------------------------------


@dataclass
class GridResult:
    folds: str
    grid: dict = field(default_factory=dict)
    overall_argmax: float = 0.0
    overall_error_mix: ErrorMix = ZERO_MIX
    flat_control: ErrorMix = ZERO_MIX
    per_fold: list = field(default_factory=list)
    argmax_stability: dict = field(default_factory=dict)
    pooled_cross_validated: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "folds": self.folds,
            "overall_argmax": self.overall_argmax,
            "overall_argmax_is_in_sample": True,
            "overall_error_mix": self.overall_error_mix.as_dict(),
            "overall_error_mix_note": (
                "IN-SAMPLE: evaluated at an argmax fitted on all sessions, "
                "including the ones being scored. Spec 7.2's gate ('if thr = "
                "RHO*R does not beat a flat threshold there is nothing to ship') "
                "belongs to pooled_cross_validated, not to this block."
            ),
            "flat_lower_control": self.flat_control.as_dict(),
            "argmax_stability": self.argmax_stability,
            "pooled_cross_validated": self.pooled_cross_validated,
            "per_fold": self.per_fold,
            "grid": {f"{rho:.2f}": row for rho, row in self.grid.items()},
        }


def _per_session_mixes(sessions, durations="synthetic"):
    """{session_id: {rho: ErrorMix}} plus the flat control, each computed once."""
    relative = {}
    flat = {}
    for session in sessions:
        relative[session.session_id] = {
            rho: evaluate([session], rho, durations=durations) for rho in GRID
        }
        flat[session.session_id] = evaluate_flat([session], durations=durations)
    return relative, flat


def _aggregate(per_session, session_ids):
    total = ZERO_MIX
    for sid in session_ids:
        total = total + per_session[sid]
    return total


def grid_search(sessions, folds: str) -> GridResult:
    """Sweep the grid and report the argmax per fold plus the full grid.

    `folds` in {"session", "topic", "author"}. The held-out unit is always the
    session: R is per-session running state, so holding out a window is
    incoherent.
    """
    if folds not in FOLD_FAMILIES:
        raise ValueError(f"folds must be one of {FOLD_FAMILIES}, got {folds!r}")
    if not sessions:
        raise ValueError("no sessions to calibrate on")

    durations = sessions[0].durations
    relative, flat = _per_session_mixes(sessions, durations=durations)
    all_ids = [s.session_id for s in sessions]

    by_rho = {rho: _aggregate({k: v[rho] for k, v in relative.items()}, all_ids) for rho in GRID}
    flat_total = _aggregate(flat, all_ids)

    grid = {}
    for rho in GRID:
        mix = by_rho[rho]
        grid[rho] = {
            "relative": {
                "rho": rho,
                "first_threshold": rho * UPPER,
                "max_threshold": rho * R_CAP,
                "error_mix": mix.as_dict(),
                "informedness": informedness(mix),
                "asymmetric_cost": asymmetric_cost(mix),
            },
            # The control on every row (spec 7.2). It does not depend on rho;
            # it is repeated per row so no table can be read without it.
            "flat_lower": {
                "threshold": LOWER,
                "error_mix": flat_total.as_dict(),
                "informedness": informedness(flat_total),
                "asymmetric_cost": asymmetric_cost(flat_total),
            },
        }

    overall_argmax = argmax_rho(by_rho)

    groups = {}
    for session in sessions:
        groups.setdefault(session.fold_key(folds), []).append(session.session_id)

    per_fold = []
    for key in sorted(groups):
        held_out = groups[key]
        train_ids = [sid for sid in all_ids if sid not in held_out]
        train_by_rho = {
            rho: _aggregate({k: v[rho] for k, v in relative.items()}, train_ids) for rho in GRID
        }
        fold_argmax = argmax_rho(train_by_rho)
        held_sessions = [s for s in sessions if s.session_id in held_out]
        per_fold.append(
            {
                "fold": key,
                "held_out_sessions": held_out,
                "argmax": fold_argmax,
                "train_error_mix_at_fold_argmax": train_by_rho[fold_argmax].as_dict(),
                "held_out_error_mix_at_fold_argmax": evaluate(
                    held_sessions, fold_argmax, durations=durations
                ).as_dict(),
                "held_out_error_mix_at_overall_argmax": evaluate(
                    held_sessions, overall_argmax, durations=durations
                ).as_dict(),
                "held_out_flat_lower_control": evaluate_flat(
                    held_sessions, durations=durations
                ).as_dict(),
            }
        )

    # I-3: the out-of-sample numbers spec 7.2's gate actually belongs to. Every
    # session sits in exactly one fold, so summing each fold's held-out mix at
    # that fold's own (training-fitted) argmax pools a genuine cross-validated
    # mix over all the same windows the in-sample headline uses.
    pooled = ZERO_MIX
    pooled_flat = ZERO_MIX
    pooled_on_topic = ZERO_MIX
    pooled_on_topic_flat = ZERO_MIX
    for record in per_fold:
        held_ids = set(record["held_out_sessions"])
        held_sessions = [s for s in sessions if s.session_id in held_ids]
        pooled = pooled + evaluate(held_sessions, record["argmax"], durations=durations)
        pooled_flat = pooled_flat + evaluate_flat(held_sessions, durations=durations)
        on_topic_only = [s for s in held_sessions if s.category in ON_TOPIC_CATEGORIES]
        if on_topic_only:
            pooled_on_topic = pooled_on_topic + evaluate(
                on_topic_only, record["argmax"], durations=durations
            )
            pooled_on_topic_flat = pooled_on_topic_flat + evaluate_flat(
                on_topic_only, durations=durations
            )

    pooled_block = {
        "definition": (
            "Each fold's held-out sessions scored at that fold's own "
            "training-fitted argmax, summed across folds. Every session appears "
            "exactly once, so this is a true leave-one-out cross-validated mix "
            "over the same windows as the in-sample headline."
        ),
        "relative": pooled.as_dict(),
        "flat_lower_control": pooled_flat.as_dict(),
        "informedness_relative": informedness(pooled),
        "informedness_flat_lower": informedness(pooled_flat),
        "beats_flat_control_on_informedness": informedness(pooled)
        > informedness(pooled_flat),
        "caught_delta_vs_flat": pooled.caught - pooled_flat.caught,
        "false_digressions_delta_vs_flat": (
            pooled.false_digressions - pooled_flat.false_digressions
        ),
        "on_topic_categories_only": {
            "categories": list(ON_TOPIC_CATEGORIES),
            "note": (
                "Spec 7.5 criterion 2 asks about the false-digression rate on "
                "on-topic material; these two categories declare no digressions, "
                "so this is that quantity, pooled out-of-sample."
            ),
            "relative": pooled_on_topic.as_dict(),
            "flat_lower_control": pooled_on_topic_flat.as_dict(),
            "false_digressions_delta_vs_flat": (
                pooled_on_topic.false_digressions - pooled_on_topic_flat.false_digressions
            ),
        },
        "in_sample_comparison": {
            "in_sample_relative": by_rho[overall_argmax].as_dict(),
            "in_sample_informedness": informedness(by_rho[overall_argmax]),
            "optimism": informedness(by_rho[overall_argmax]) - informedness(pooled),
        },
    }

    fold_argmaxes = [f["argmax"] for f in per_fold]
    matching = sum(1 for a in fold_argmaxes if a == overall_argmax)
    stability = {
        "n_folds": len(per_fold),
        "folds_matching_overall_argmax": matching,
        "fraction_matching": matching / len(per_fold) if per_fold else 0.0,
        "identical": len(set(fold_argmaxes)) == 1,
        "distinct_argmaxes": sorted(set(fold_argmaxes)),
        "argmax_min": min(fold_argmaxes) if fold_argmaxes else None,
        "argmax_max": max(fold_argmaxes) if fold_argmaxes else None,
    }

    return GridResult(
        folds=folds,
        grid=grid,
        overall_argmax=overall_argmax,
        overall_error_mix=by_rho[overall_argmax],
        flat_control=flat_total,
        per_fold=per_fold,
        argmax_stability=stability,
        pooled_cross_validated=pooled_block,
    )


# --------------------------------------------------------------------------
# corpus loading
# --------------------------------------------------------------------------


def corpus_paths() -> list[Path]:
    return sorted(CORPUS_DIR.glob("*/sess_*.txt"))


def load_sessions(durations: str = "synthetic", paths=None) -> list[ScoredSession]:
    """Replay and score each corpus session exactly once."""
    if durations != "synthetic":
        raise ValueError(
            'only durations="synthetic" runs: the timed/WER leg is not run '
            "(no real-duration transcripts exist — see WER_NOT_RUN)"
        )
    from eval_dataset.tools.embed_windows import score_windows  # heavy import, deferred

    sessions = []
    for path in paths if paths is not None else corpus_paths():
        script = parse_script(str(path), require_digression_labels=True)
        windows = replay(script, synthetic_durations(script))
        scores = score_windows(windows, script.topic, script.scope)
        sessions.append(
            ScoredSession(
                session_id=path.stem,
                topic=script.topic,
                scope=script.scope,
                category=script.category,
                author=script.author,
                durations=durations,
                window_indices=tuple(w.index for w in windows),
                similarities=tuple(s.similarity for s in scores),
                gold=tuple(gold_label(w, script) for w in windows),
            )
        )
    return sessions


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _group_mixes(sessions, key, rho, durations="synthetic") -> dict:
    groups = {}
    for session in sessions:
        groups.setdefault(key(session), []).append(session)
    out = {}
    for name in sorted(groups):
        members = groups[name]
        out[name] = {
            "sessions": [s.session_id for s in members],
            "relative": evaluate(members, rho, durations=durations).as_dict(),
            "flat_lower": evaluate_flat(members, durations=durations).as_dict(),
        }
    return out


def excluded_by_category(sessions) -> dict:
    out = {}
    for session in sessions:
        bucket = out.setdefault(
            session.category,
            {"windows": 0, "excluded": 0, "on_topic": 0, "off_topic": 0, "sessions": 0},
        )
        bucket["sessions"] += 1
        bucket["windows"] += len(session.gold)
        bucket["excluded"] += sum(1 for g in session.gold if g is None)
        bucket["on_topic"] += sum(1 for g in session.gold if g == "on_topic")
        bucket["off_topic"] += sum(1 for g in session.gold if g == "off_topic")
    for bucket in out.values():
        bucket["excluded_fraction"] = (
            bucket["excluded"] / bucket["windows"] if bucket["windows"] else 0.0
        )
    return dict(sorted(out.items()))


def _content_tokens(text: str) -> set:
    stop = {
        "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with",
        "more", "less", "reducing", "increasing", "changing", "avoiding",
        "approach", "management", "compliance", "rules", "activity",
    }
    tokens = {t.strip(".,!?;:\"'()[]{}").lower() for t in text.split()}
    return {t for t in tokens if len(t) > 3 and t not in stop}


def abstract_on_topic_cosines(sessions) -> dict:
    """Per-script cosine distribution for the five abstract_on_topic sessions.

    LIM-2 from the Task 11 review: spec 5.4 justifies AnnoMI on the premise that
    its content is on-topic while sharing little vocabulary with its topic label.
    That does not hold uniformly — sess_CAL16 ("smoking cessation") is saturated
    with smoking/cigarette/smoke, while sess_CAL19 ("reducing gambling") fits the
    premise. A high-overlap script scores high cosines and is therefore very
    rarely false-flagged, which flatters the headline false-digression rate while
    still inflating its denominator. (Exactly: sess_CAL16 contributes 2 false
    digressions at the argmax, from its two sub-LOWER windows — few, not none.)
    This block lets a reader see which of the five actually exercise the
    low-overlap case.
    """
    out = {}
    for session in sorted(
        (s for s in sessions if s.category == "abstract_on_topic"),
        key=lambda s: s.session_id,
    ):
        sims = list(session.similarities)
        topic_tokens = _content_tokens(f"{session.topic} {session.scope}")
        out[session.session_id] = {
            "topic": session.topic,
            "scope": session.scope,
            "author": session.author,
            "n_windows": len(sims),
            "min": min(sims) if sims else None,
            "median": statistics.median(sims) if sims else None,
            "max": max(sims) if sims else None,
            "mean": statistics.fmean(sims) if sims else None,
            "p25": statistics.quantiles(sims, n=4)[0] if len(sims) > 1 else None,
            "p75": statistics.quantiles(sims, n=4)[2] if len(sims) > 1 else None,
            "topic_label_content_tokens": sorted(topic_tokens),
            "windows_at_or_above_UPPER": sum(1 for s in sims if s >= UPPER),
            "windows_below_LOWER": sum(1 for s in sims if s < LOWER),
        }
    return out


def agent_vs_real(sessions, rho, durations="synthetic") -> dict:
    """Split the corpus by author provenance and compare relative against flat.

    I-4: the entire net improvement over the flat control comes from the 10
    real:annomi sessions — arithmetically the same window set as
    abstract_on_topic + depressed_baseline. On the 15 agent-authored sessions the
    relative threshold nets a small caught gain and an *extra* false digression.
    That bears directly on whether any of this generalises, so it is recorded
    rather than left to be derived from per_author.
    """
    out = {}
    for name, authors in (("agent_authored", AGENT_AUTHORS), ("real_data", REAL_AUTHORS)):
        members = [s for s in sessions if s.author in authors]
        rel = evaluate(members, rho, durations=durations)
        flat = evaluate_flat(members, durations=durations)
        out[name] = {
            "authors": list(authors),
            "sessions": [s.session_id for s in members],
            "n_sessions": len(members),
            "categories": sorted({s.category for s in members}),
            "relative": rel.as_dict(),
            "flat_lower_control": flat.as_dict(),
            "caught_delta_vs_flat": rel.caught - flat.caught,
            "false_digressions_delta_vs_flat": (
                rel.false_digressions - flat.false_digressions
            ),
            "informedness_relative": informedness(rel),
            "informedness_flat_lower": informedness(flat),
        }
    out["_summary"] = {
        "note": (
            "The net gain of the relative threshold over the flat control is "
            "carried by the real-data (AnnoMI) half of the corpus. On the "
            "agent-authored half the false-digression count does not improve. "
            "The real:annomi sessions are exactly the abstract_on_topic and "
            "depressed_baseline categories, so this split and the per-category "
            "split describe the same window set from two directions."
        ),
        "agent_false_digression_delta": out["agent_authored"][
            "false_digressions_delta_vs_flat"
        ],
        "real_false_digression_delta": out["real_data"][
            "false_digressions_delta_vs_flat"
        ],
        "agent_caught_delta": out["agent_authored"]["caught_delta_vs_flat"],
        "real_caught_delta": out["real_data"]["caught_delta_vs_flat"],
    }
    return out


def predictions(sessions, rho, durations="synthetic") -> list:
    rows = []
    for session in sessions:
        thresholds = running_thresholds(session.similarities, rho)
        for index, similarity, thr, gold in zip(
            session.window_indices, session.similarities, thresholds, session.gold
        ):
            rows.append(
                {
                    "session_id": session.session_id,
                    "window_index": index,
                    "category": session.category,
                    "topic": session.topic,
                    "author": session.author,
                    "similarity": round(similarity, 6),
                    "gold": gold,
                    "relative_threshold": round(thr, 6),
                    "relative_prediction": (
                        "digression" if similarity < thr else "on_topic"
                    ),
                    "flat_lower_prediction": (
                        "digression" if similarity < LOWER else "on_topic"
                    ),
                }
            )
    return rows


def paired_changes(rows) -> dict:
    """Window-level relative-vs-flat comparison on labellable windows only."""
    improved = regressed = unchanged = 0
    for row in rows:
        if row["gold"] is None:
            continue
        want = "digression" if row["gold"] == "off_topic" else "on_topic"
        rel_ok = row["relative_prediction"] == want
        flat_ok = row["flat_lower_prediction"] == want
        if rel_ok and not flat_ok:
            improved += 1
        elif flat_ok and not rel_ok:
            regressed += 1
        else:
            unchanged += 1
    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "net_correct": improved - regressed,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(sessions) -> dict:
    families = {family: grid_search(sessions, folds=family) for family in FOLD_FAMILIES}
    overall_argmax = families["session"].overall_argmax
    rows = predictions(sessions, overall_argmax)

    report = {
        "protocol": "rho_relative_threshold_calibration",
        "production_evidence": False,
        "mechanism": {
            "formula": "thr_i = RHO * R, with R read before window i may update it",
            "R_start": UPPER,
            "R_cap": R_CAP,
            "update_rule": "if sim_i >= thr_i: R = min(R_CAP, max(R, sim_i))",
            "first_threshold_is_always": "RHO * 0.36 (0.162 at RHO=0.45)",
            "parameters_in_scope": ["RHO", "R_cap", "R_update_rule"],
            "spec": "7.1 — a choice, not a deduction; the any-window-updates-R "
                    "variant is a different mechanism and is not implemented",
            "ships": False,
        },
        "objective": OBJECTIVE,
        "grid_definition": {
            "from": GRID[0],
            "to": GRID[-1],
            "step": 0.01,
            "n": len(GRID),
            "built_with": "integer arithmetic + round(n/100, 2)",
        },
        "control": {
            "flat_lower": LOWER,
            "note": "reported on every table and every fold; if thr = RHO*R does "
                    "not beat a flat threshold there is nothing to ship (spec 7.2)",
        },
        "limitations": [
            "No production evidence: this mechanism ships nothing and was never "
            "served. thr = RHO*R appears nowhere outside ground_truth/.",
            "The WER robustness leg was NOT RUN — see wer_robustness for the "
            "reason. durations=\"synthetic\" is the only path exercised.",
            "The durations=\"timed\" path is unreachable in this file: the STT "
            "sweep was skipped, so no real-duration transcript exists for "
            "run_stt_sweep.timed_durations to measure. load_sessions() refuses "
            "it rather than wiring a path that cannot be exercised.",
            "Because thr = RHO*R with RHO < 1, thr is always strictly below R, so "
            "any window with sim >= R also clears thr. The below-threshold guard "
            "is therefore a no-op across the swept grid; it bites only for "
            "RHO > 1 and is unit-tested there.",
            "Synthetic durations (0.4 s/word) set the window boundaries, so the "
            "window population itself is an assumption, not a measurement.",
            "LIM-2: the abstract_on_topic premise (on-topic content, low "
            "vocabulary overlap with the topic label) does not hold uniformly "
            "across the five sessions — see abstract_on_topic_cosine_distribution.",
            "25 sessions / ~380 windows is a small corpus; per-fold argmax "
            "movement of a few hundredths is within its resolution.",
            "leave-one-topic-out is degenerate on this corpus: topics are "
            "distinct by construction, one per session, so the topic folds are "
            "the session folds under a different key and carry no independent "
            "information. Both families are reported as required, but their "
            "agreement is arithmetic, not evidence.",
            "The overall argmax is IN-SAMPLE (fitted on all 25 sessions). "
            "Pooled leave-one-out numbers are in each family's "
            "pooled_cross_validated block, and that is where spec 7.2's "
            "beats-a-flat-threshold gate belongs.",
            "The three fold families report the same overall argmax by "
            "construction, not by replication — see "
            "fold_family_structural_invariance.",
            "The primary objective (informedness) and the committed secondary "
            "(asymmetric_cost) disagree on this corpus; so do accuracy and "
            "digression-class F1. See objective_sensitivity. The argmax is "
            "conditional on an objective this task chose, not one the spec "
            "pre-registered.",
            "The net improvement over the flat control is carried entirely by "
            "the 10 real-data (AnnoMI) sessions; on the 15 agent-authored "
            "sessions the false-digression count does not improve. See "
            "agent_authored_vs_real_data.",
            "The real:annomi author fold holds out 10 of the 25 sessions and "
            "contains zero declared digressions, so its held-out mix has "
            "digression_total = 0 and can only measure false digressions. Its "
            "*training* argmax is fitted on the 15 agent-authored sessions "
            "alone, which is why that fold's argmax moves furthest.",
        ],
        "corpus": {
            "n_sessions": len(sessions),
            "n_windows": sum(len(s.similarities) for s in sessions),
            "durations": "synthetic",
            "categories": excluded_by_category(sessions),
            "authors": {
                author: [s.session_id for s in sessions if s.author == author]
                for author in AUTHORS
            },
        },
        "excluded_windows_by_category": excluded_by_category(sessions),
        "abstract_on_topic_cosine_distribution": abstract_on_topic_cosines(sessions),
        "agent_authored_vs_real_data": agent_vs_real(sessions, overall_argmax),
        "objective_sensitivity": objective_sensitivity(
            {rho: evaluate(sessions, rho) for rho in GRID}
        ),
        "wer_robustness": WER_NOT_RUN,
    }

    for family, result in families.items():
        block = result.as_dict()
        block["per_topic"] = _group_mixes(
            sessions, lambda s: s.topic, result.overall_argmax
        )
        block["per_category"] = _group_mixes(
            sessions, lambda s: s.category, result.overall_argmax
        )
        block["per_author"] = _group_mixes(
            sessions, lambda s: s.author, result.overall_argmax
        )
        block["paired_changes"] = paired_changes(
            predictions(sessions, result.overall_argmax)
        )
        block["predictions"] = predictions(sessions, result.overall_argmax)
        report[f"leave_one_{family}_out"] = block

    # I-1: the overall argmax CANNOT vary by fold family — grid_search fits it on
    # all sessions regardless of `folds`. Every block except `per_fold` and
    # `argmax_stability` is byte-identical across the three families. So this is
    # a property of the code, not corroboration from the data, and is named and
    # annotated to make that impossible to misread. There is no cross-family
    # agreement *finding* in this file.
    report["fold_family_structural_invariance"] = {
        "is_evidence": False,
        "kind": "structural property of the implementation, not a result",
        "overall_argmaxes": {f: families[f].overall_argmax for f in FOLD_FAMILIES},
        "identical_by_construction": True,
        "why": (
            "grid_search() fits overall_argmax on every session irrespective of "
            "the fold family, so the three values are the same computation run "
            "three times. The grid, overall_error_mix, flat_lower_control, "
            "per_topic, per_category, per_author, paired_changes and all "
            "predictions rows are identical across the three family blocks; only "
            "per_fold, argmax_stability and pooled_cross_validated differ. Do "
            "not read this agreement as replication. (This is broader than the "
            "separate topic-equals-session degeneracy noted in limitations.)"
        ),
        "what_actually_differs_by_family": [
            "per_fold",
            "argmax_stability",
            "pooled_cross_validated",
        ],
        "per_family_fold_stability": {
            f: families[f].argmax_stability for f in FOLD_FAMILIES
        },
        "per_family_pooled_cross_validated_informedness": {
            f: families[f].pooled_cross_validated["informedness_relative"]
            for f in FOLD_FAMILIES
        },
    }
    report["headline"] = {
        "argmax": overall_argmax,
        "argmax_is_in_sample": True,
        "in_sample_warning": (
            "relative_error_mix_at_argmax is IN-SAMPLE: the argmax was fitted on "
            "all 25 sessions, including the ones scored here. Use "
            "pooled_cross_validated_by_family for the out-of-sample comparison "
            "spec 7.2's gate asks for."
        ),
        "relative_error_mix_at_argmax": families["session"].overall_error_mix.as_dict(),
        "flat_lower_control": families["session"].flat_control.as_dict(),
        "paired_changes_relative_vs_flat": paired_changes(rows),
        "pooled_cross_validated_by_family": {
            family: {
                "argmaxes_used": sorted(
                    {f["argmax"] for f in families[family].per_fold}
                ),
                "relative": families[family].pooled_cross_validated["relative"],
                "flat_lower_control": families[family].pooled_cross_validated[
                    "flat_lower_control"
                ],
                "informedness_relative": families[family].pooled_cross_validated[
                    "informedness_relative"
                ],
                "informedness_flat_lower": families[family].pooled_cross_validated[
                    "informedness_flat_lower"
                ],
                "beats_flat_control_on_informedness": families[
                    family
                ].pooled_cross_validated["beats_flat_control_on_informedness"],
                "on_topic_categories_only": families[family].pooled_cross_validated[
                    "on_topic_categories_only"
                ],
            }
            for family in FOLD_FAMILIES
        },
        "objective_sensitivity_summary": report["objective_sensitivity"]["_summary"],
        "agent_vs_real_summary": report["agent_authored_vs_real_data"]["_summary"],
    }
    report["inputs"] = [
        {"path": str(p.relative_to(_REPO_ROOT)), "sha256": _sha256(p)}
        for p in corpus_paths()
    ]
    return report


DEFAULT_OUTPUT = Path(__file__).with_name("rho_calibration_results.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--durations", default="synthetic", choices=["synthetic"])
    parser.add_argument(
        "--all-wer",
        action="store_true",
        help="accepted and refused: the WER leg is not run (see WER_NOT_RUN)",
    )
    args = parser.parse_args(argv)

    if args.all_wer:
        print("--all-wer requested but NOT RUN: " + WER_NOT_RUN["reason"], file=sys.stderr)

    sessions = load_sessions(durations=args.durations)
    report = build_report(sessions)
    Path(args.output).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    head = report["headline"]
    print(f"sessions={len(sessions)} windows={report['corpus']['n_windows']}")
    for family in FOLD_FAMILIES:
        block = report[f"leave_one_{family}_out"]
        st = block["argmax_stability"]
        print(
            f"leave_one_{family}_out: argmax={block['overall_argmax']:.2f} "
            f"stability={st['folds_matching_overall_argmax']}/{st['n_folds']} "
            f"identical={st['identical']} distinct={st['distinct_argmaxes']}"
        )
    print(f"relative @ argmax {head['argmax']:.2f} (IN-SAMPLE): "
          f"{head['relative_error_mix_at_argmax']}")
    print(f"flat LOWER control:  {head['flat_lower_control']}")
    print(f"paired: {head['paired_changes_relative_vs_flat']}")
    for family, pooled in head["pooled_cross_validated_by_family"].items():
        rel, flat = pooled["relative"], pooled["flat_lower_control"]
        print(
            f"pooled CV ({family}): relative caught={rel['caught']}/"
            f"{rel['digression_total']} fd={rel['false_digressions']}/"
            f"{rel['on_topic_total']} J={pooled['informedness_relative']:.4f} | "
            f"flat caught={flat['caught']} fd={flat['false_digressions']} "
            f"J={pooled['informedness_flat_lower']:.4f} | "
            f"beats_flat={pooled['beats_flat_control_on_informedness']}"
        )
    obj = report["objective_sensitivity"]["_summary"]
    print(f"objectives: primary={obj['primary_argmax']} "
          f"secondary={obj['committed_secondary_argmax']} "
          f"agree={obj['primary_and_secondary_agree']} "
          f"distinct={obj['distinct_argmaxes_across_objectives']}")
    avr = report["agent_authored_vs_real_data"]["_summary"]
    print(f"agent vs real: agent fd_delta={avr['agent_false_digression_delta']} "
          f"caught_delta={avr['agent_caught_delta']} | "
          f"real fd_delta={avr['real_false_digression_delta']} "
          f"caught_delta={avr['real_caught_delta']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
