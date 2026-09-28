"""Offline experiments that turn the video sub-signal weights into a measured
result instead of a guess.

Two independent modes:

**Corpus mode (no backend, no live session)** — the primary experiment. Runs
the MediaPipe detector over labeled frame corpora, grid-searches the
(w_eyes, w_head, w_gaze) simplex, and reports class separation, a decision
threshold, and a group-disjoint held-out check. Each corpus directory is a
named split, reported separately as well as pooled. Corpora come from the
public datasets (`public_datasets.py`) or from an optional local webcam
capture (`collect_attention_frames.py`):

    venv/bin/python weight_search.py \
        --corpus datasets/head_pose_image_database datasets/closed_open_eyes

**Backend-correlation mode** — the original method from the design spec:
correlate `video_attention_score` against the speech-based
`engagement_score` already stored for a pilot session. Weaker evidence (the
speech score is a proxy, not a label of visual attention) and it needs real
session data:

    python weight_search.py --barter-id 1 --learner-user-id 2

Nothing here imports cv2 or mediapipe at module level; the detector is
lazy-imported inside `score_corpus_frames` so every metric function stays
testable under a plain system python.
"""

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

from scoring import video_attention_score


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or statistics.pvariance(xs) == 0 or statistics.pvariance(ys) == 0:
        return 0.0
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def _weight_grid(step: float) -> list[tuple[float, float, float]]:
    """All (w1, w2, w3) triples on the step grid that sum to 1.0."""
    steps = [round(i * step, 4) for i in range(int(round(1.0 / step)) + 1)]
    grid = []
    for w1, w2 in itertools.product(steps, steps):
        w3 = round(1.0 - w1 - w2, 4) + 0.0
        if w3 == 0.0:
            w3 = 0.0  # normalise the -0.0 that float subtraction can produce
        if 0.0 <= w3 <= 1.0:
            grid.append((w1, w2, w3))
    return grid


def sweep_subsignal_weights(
    raw_signals: list[dict], ground_truth: list[float], step: float = 0.1
) -> list[dict]:
    """Try every (w_eyes, w_head, w_gaze) on the grid, correlate resulting
    video_attention_score series against ground_truth, return all results."""
    results = []
    for weights in _weight_grid(step):
        scores = [video_attention_score(s, weights) for s in raw_signals]
        corr = pearson_correlation(scores, ground_truth)
        results.append({"weights": weights, "correlation": corr})
    return sorted(results, key=lambda r: -r["correlation"])


def sweep_fusion_weights(
    video_scores: list[float], speech_scores: list[float], step: float = 0.1
) -> list[dict]:
    """Try every (w_video, w_speech) pair summing to 1.0, correlate the fused
    series against the speech series alone (sanity: fused should track speech
    reasonably while still being influenced by video)."""
    results = []
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        w_video = round(i * step, 4)
        w_speech = round(1.0 - w_video, 4)
        fused = [w_speech * s + w_video * v for s, v in zip(speech_scores, video_scores)]
        corr = pearson_correlation(fused, speech_scores)
        results.append({"w_video": w_video, "w_speech": w_speech, "correlation": corr})
    return sorted(results, key=lambda r: -r["correlation"])


def _fetch_session_data(
    backend_url: str, barter_id: int, learner_user_id: int, backend: str = "local"
) -> tuple[list[dict], list[float]]:
    import httpx  # lazy: corpus mode must not need the HTTP client

    video_rows = httpx.get(f"{backend_url}/session/{barter_id}/video-engagement").json()
    history_rows = httpx.get(f"{backend_url}/session/{barter_id}/engagement/history").json()

    video_rows = [
        r for r in video_rows
        if r["user_id"] == learner_user_id and r["backend_used"] == backend
    ]

    raw_signals = [r["raw_signals"] for r in video_rows]
    speech_scores = [
        r["speech_engagement_score"] for r in history_rows
        if r["speech_engagement_score"] is not None
    ]
    return raw_signals, speech_scores


def _write_design_choices(path: str, subsignal_results: list[dict], fusion_results: list[dict] | None):
    """Write the backend-correlation mode's results to its own markdown file.

    Deliberately NOT design-choices.md: that file is hand-maintained and
    documents the method, and this mode is the weaker of the two experiments.
    """
    lines = ["# Video Engagement — Backend-Correlation Weight Results\n"]
    lines.append("Generated by `weight_search.py --barter-id ...`. Weights below were picked by\n"
                  "grid search + Pearson correlation against the existing speech-based\n"
                  "`engagement_score` (a proxy, not a label of visual attention). The\n"
                  "labeled-frame corpus mode is the stronger experiment — see\n"
                  "`docs/video_engagement/design-choices.md`.\n")

    lines.append("\n## Sub-signal weights (eyes_open, head_deviation, gaze_centered)\n")
    lines.append("| w_eyes | w_head | w_gaze | correlation |\n|---|---|---|---|\n")
    for r in subsignal_results[:10]:
        w1, w2, w3 = r["weights"]
        lines.append(f"| {w1} | {w2} | {w3} | {r['correlation']:.4f} |\n")
    best = subsignal_results[0]
    lines.append(f"\n**Chosen:** `{best['weights']}` (correlation {best['correlation']:.4f})\n")

    if fusion_results:
        lines.append("\n## Fusion weights (w_speech, w_video)\n")
        lines.append("| w_speech | w_video | correlation |\n|---|---|---|\n")
        for r in fusion_results[:10]:
            lines.append(f"| {r['w_speech']} | {r['w_video']} | {r['correlation']:.4f} |\n")
        best_fusion = fusion_results[0]
        lines.append(f"\n**Chosen:** w_speech={best_fusion['w_speech']}, w_video={best_fusion['w_video']}\n")

    with open(path, "w") as f:
        f.writelines(lines)



# ---------------------------------------------------------------------------
# Corpus mode: labeled frames -> sub-signals -> weights + decision threshold
# ---------------------------------------------------------------------------

SUB_SIGNAL_KEYS = ("eyes_open", "head_deviation", "gaze_centered")

# Label -> target attention value used as the fit target.
#
#   attentive     1.0  the positive class: all three sub-signals should be good.
#   eyes_closed   0.0  breaks `eyes_open` alone.
#   looking_away  0.0  breaks `head_deviation` alone.
#   gaze_off      0.0  breaks `gaze_centered` alone.
#   absent        None EXCLUDED from the fit. `video_attention_score` needs
#                      landmarks, and a frame with no face yields none, so
#                      there is no score to regress on. Production already
#                      treats a no-detection frame as "no sample" rather than
#                      as zero attention (`process_buffer` filters None), so
#                      scoring `absent` as 0.0 here would fit a case the
#                      service never evaluates. `absent` rows instead measure
#                      detector coverage: they should produce no detection.
LABEL_TARGETS: dict[str, float | None] = {
    "attentive": 1.0,
    "eyes_closed": 0.0,
    "looking_away": 0.0,
    "gaze_off": 0.0,
    "absent": None,
}


def label_to_target(label: str) -> float | None:
    """Target attention value for a label, or None if the label is excluded."""
    if label not in LABEL_TARGETS:
        raise ValueError(
            f"Unknown label {label!r}; expected one of {sorted(LABEL_TARGETS)}"
        )
    return LABEL_TARGETS[label]


def load_corpus_manifest(corpus_dir: str | Path, split: str | None = None) -> list[dict]:
    """Read `labels.jsonl` from a capture corpus into validated records.

    Each record is {"frame": absolute Path, "label": str, "target": float|None,
    "timestamp": str|None, "split": str}. `split` defaults to the corpus
    directory's name and keeps per-dataset results separable once several
    corpora are pooled. Raises on a missing manifest, a malformed line, an
    unknown label, or a manifest row pointing at a frame that is not on disk —
    a silently shrinking corpus would quietly bias the fit.
    """
    corpus_path = Path(corpus_dir)
    split_name = split or corpus_path.resolve().name
    manifest = corpus_path / "labels.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No labels.jsonl in {corpus_path} — capture a corpus first with "
            "collect_attention_frames.py"
        )

    records: list[dict] = []
    with manifest.open(encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest}:{line_no}: invalid JSON ({exc})") from exc
            for field in ("frame", "label"):
                if not row.get(field):
                    raise ValueError(f"{manifest}:{line_no}: missing {field}")
            target = label_to_target(row["label"])
            frame_path = Path(row["frame"])
            if not frame_path.is_absolute():
                frame_path = corpus_path / frame_path
            if not frame_path.is_file():
                raise FileNotFoundError(f"{manifest}:{line_no}: missing frame {frame_path}")
            records.append(
                {
                    "frame": frame_path,
                    "label": row["label"],
                    "target": target,
                    "timestamp": row.get("timestamp"),
                    "split": split_name,
                    "group": str(row.get("group") or split_name),
                }
            )
    if not records:
        raise ValueError(f"{manifest}: no labeled frames")
    return records


def score_corpus_frames(records: list[dict], detector=None) -> tuple[list[dict], dict]:
    """Run the local MediaPipe detector on each record; return (samples, coverage).

    `samples` holds only rows the weight fit can use: a detected face and a
    non-None target. `coverage` reports what was dropped and why, which is the
    honesty check on the fit — a corpus where most `attentive` frames failed to
    detect cannot support any weight claim.

    `detector` defaults to `main.process_frame_local` (lazy-imported so this
    module stays importable without mediapipe/cv2); tests inject a fake.
    """
    if detector is None:
        from main import process_frame_local as detector  # lazy: needs cv2 + mediapipe

    samples: list[dict] = []
    coverage = {
        "frames_total": len(records),
        "used": 0,
        "no_detection_on_scoring_label": 0,
        "absent_correctly_undetected": 0,
        "absent_falsely_detected": 0,
        "per_label_total": {},
        "per_label_detected": {},
    }
    for record in records:
        label = record["label"]
        coverage["per_label_total"][label] = coverage["per_label_total"].get(label, 0) + 1
        sub_signals = detector(record["frame"].read_bytes())
        detected = sub_signals is not None
        if detected:
            coverage["per_label_detected"][label] = coverage["per_label_detected"].get(label, 0) + 1

        if record["target"] is None:
            key = "absent_falsely_detected" if detected else "absent_correctly_undetected"
            coverage[key] += 1
            continue
        if not detected:
            coverage["no_detection_on_scoring_label"] += 1
            continue
        samples.append(
            {
                "frame": str(record["frame"]),
                "split": record.get("split", "corpus"),
                "group": record.get("group", record.get("split", "corpus")),
                "label": label,
                "target": record["target"],
                "raw_signals": {k: float(sub_signals[k]) for k in SUB_SIGNAL_KEYS},
            }
        )
    coverage["used"] = len(samples)
    return samples, coverage


def roc_auc(scores: list[float], targets: list[float]) -> float:
    """ROC-AUC for a binary target (1.0 = attentive) via the rank-sum identity.

    Equal to P(score of a random attentive frame > score of a random
    inattentive one), with ties counted as half. 0.5 is chance, 1.0 perfect
    separation. Pure python — no sklearn.
    """
    if len(scores) != len(targets):
        raise ValueError("scores and targets must be the same length")
    positives = [s for s, t in zip(scores, targets) if t >= 0.5]
    negatives = [s for s, t in zip(scores, targets) if t < 0.5]
    if not positives or not negatives:
        raise ValueError("ROC-AUC needs at least one positive and one negative sample")

    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and scores[ordered[j + 1]] == scores[ordered[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based; ties share the mean
        for k in range(i, j + 1):
            ranks[ordered[k]] = average_rank
        i = j + 1

    positive_rank_sum = sum(r for r, t in zip(ranks, targets) if t >= 0.5)
    n_pos, n_neg = len(positives), len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_threshold(scores: list[float], targets: list[float]) -> dict:
    """Pick the decision threshold on the fused score that best separates the classes.

    Predicts "attentive" when `score >= threshold`. Candidates are the
    midpoints between adjacent distinct scores plus one below the minimum and
    one above the maximum, so every achievable confusion matrix is considered.
    Ranked by Youden's J (= tpr - fpr = balanced accuracy * 2 - 1), which
    weights both classes equally — plain accuracy would let an unbalanced
    corpus win by always predicting the larger class. Ties are broken by
    taking the median tied candidate, so the threshold sits in the middle of
    the winning plateau rather than hugging an observed score.

    Computed as a single sweep over the sorted scores (O(n log n)); the naive
    re-scan per candidate is O(n^2) and does not finish on a few thousand
    frames times a few hundred weight triples.
    """
    if len(scores) != len(targets):
        raise ValueError("scores and targets must be the same length")
    n_pos = sum(1 for t in targets if t >= 0.5)
    n_neg = len(targets) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("Threshold selection needs at least one positive and one negative sample")

    # Positives/negatives per distinct score, ascending.
    counts: dict[float, list[int]] = {}
    for score, target in zip(scores, targets):
        bucket = counts.setdefault(score, [0, 0])
        bucket[0 if target >= 0.5 else 1] += 1
    distinct = sorted(counts)

    # suffix[i] = (#positives, #negatives) with score >= distinct[i].
    m = len(distinct)
    suffix_pos = [0] * (m + 1)
    suffix_neg = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        pos, neg = counts[distinct[i]]
        suffix_pos[i] = suffix_pos[i + 1] + pos
        suffix_neg[i] = suffix_neg[i + 1] + neg

    # Candidate i selects exactly the scores at distinct[i:].
    candidates = [distinct[0] - 0.01]
    candidates += [(a + b) / 2.0 for a, b in zip(distinct, distinct[1:])]
    candidates.append(distinct[-1] + 0.01)

    scored = []
    for i, threshold in enumerate(candidates):
        tp, fp = suffix_pos[i], suffix_neg[i]
        tpr = tp / n_pos
        fpr = fp / n_neg
        scored.append(
            {
                "threshold": threshold,
                "youden_j": tpr - fpr,
                "tpr": tpr,
                "fpr": fpr,
                "balanced_accuracy": (tpr + (1.0 - fpr)) / 2.0,
                "accuracy": (tp + (n_neg - fp)) / len(scores),
            }
        )

    best_j = max(row["youden_j"] for row in scored)
    tied = [row for row in scored if row["youden_j"] == best_j]
    return tied[len(tied) // 2]


def mean_score_per_label(scores: list[float], labels: list[str]) -> dict[str, float]:
    """Mean fused score per label — the per-class separation readout."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    grouped: dict[str, list[float]] = {}
    for score, label in zip(scores, labels):
        grouped.setdefault(label, []).append(score)
    return {label: statistics.fmean(values) for label, values in sorted(grouped.items())}


def _metrics(scores: list[float], targets: list[float], labels: list[str]) -> dict:
    attentive = [s for s, t in zip(scores, targets) if t >= 0.5]
    inattentive = [s for s, t in zip(scores, targets) if t < 0.5]
    threshold = best_threshold(scores, targets)
    return {
        "n": len(scores),
        "auc": roc_auc(scores, targets),
        "balanced_accuracy": threshold["balanced_accuracy"],
        "separation": statistics.fmean(attentive) - statistics.fmean(inattentive),
        "pearson_r": pearson_correlation(scores, targets),
        "threshold": threshold,
        "mean_score_per_label": mean_score_per_label(scores, labels),
    }


def metrics_at_threshold(scores: list[float], targets: list[float], threshold: float) -> dict:
    """Confusion-matrix metrics for an externally chosen threshold.

    Used to report how one pooled decision threshold behaves on each split —
    a threshold is only useful if it is a single number the service can apply,
    so each split has to be scored at the SAME cut, not at its own best one.
    """
    n_pos = sum(1 for t in targets if t >= 0.5)
    n_neg = len(targets) - n_pos
    tp = sum(1 for s, t in zip(scores, targets) if s >= threshold and t >= 0.5)
    fp = sum(1 for s, t in zip(scores, targets) if s >= threshold and t < 0.5)
    tpr = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")
    return {
        "n": len(scores),
        "threshold": threshold,
        "tpr": tpr,
        "fpr": fpr,
        "balanced_accuracy": (tpr + (1.0 - fpr)) / 2.0,
        "accuracy": (tp + (n_neg - fp)) / len(scores) if scores else float("nan"),
    }


def evaluate_corpus_weights(samples: list[dict], weights: tuple[float, float, float]) -> dict:
    """Full metric set for one weight triple: pooled, plus one block per split.

    `mean_split_auc` is the headline objective rather than the pooled AUC,
    because the splits differ wildly in size and each one is evidence about a
    different sub-signal — pooling would let the bigger dataset decide a weight
    it cannot speak to. `min_split_balanced_accuracy` guards against a triple
    that wins on average by abandoning one split entirely.
    """
    scores = [video_attention_score(s["raw_signals"], weights) for s in samples]
    targets = [s["target"] for s in samples]
    labels = [s["label"] for s in samples]

    result = {"weights": weights}
    result.update(_metrics(scores, targets, labels))

    splits = sorted({s.get("split", "corpus") for s in samples})
    per_split = {}
    for split in splits:
        picked = [i for i, s in enumerate(samples) if s.get("split", "corpus") == split]
        split_targets = [targets[i] for i in picked]
        if len({t >= 0.5 for t in split_targets}) < 2:
            # A split with only one class can still be reported, but it has no
            # AUC and cannot contribute to the objective.
            per_split[split] = {
                "n": len(picked),
                "single_class": True,
                "mean_score_per_label": mean_score_per_label(
                    [scores[i] for i in picked], [labels[i] for i in picked]
                ),
            }
            continue
        per_split[split] = _metrics(
            [scores[i] for i in picked], split_targets, [labels[i] for i in picked]
        )

    usable = [m for m in per_split.values() if not m.get("single_class")]
    result["per_split"] = per_split
    result["mean_split_auc"] = statistics.fmean([m["auc"] for m in usable]) if usable else result["auc"]
    result["min_split_auc"] = min((m["auc"] for m in usable), default=result["auc"])
    result["min_split_balanced_accuracy"] = min(
        (m["balanced_accuracy"] for m in usable), default=result["balanced_accuracy"]
    )
    # Per-split metrics at the POOLED best threshold — what production would do.
    pooled_threshold = result["threshold"]["threshold"]
    result["per_split_at_pooled_threshold"] = {
        split: metrics_at_threshold(
            [scores[i] for i, s in enumerate(samples) if s.get("split", "corpus") == split],
            [targets[i] for i, s in enumerate(samples) if s.get("split", "corpus") == split],
            pooled_threshold,
        )
        for split in splits
    }
    return result


def sweep_corpus_weights(samples: list[dict], step: float = 0.05) -> list[dict]:
    """Grid-search the weight simplex against labeled frames, best first.

    Objective, in order:

    1. **mean per-split ROC-AUC** — threshold-free ranking quality, averaged
       over splits so a large dataset cannot outvote a small one.
    2. **minimum per-split balanced accuracy** at each split's own best
       threshold — rejects triples that win on average while failing a split.
    3. **class mean separation** — margin, as a proxy for surviving a camera
       or lighting change.

    Pearson r against the 0/1 target is reported but deliberately NOT the
    objective: with binary targets it rewards a spread-out score distribution
    rather than a separable one.
    """
    if not samples:
        raise ValueError("No scored samples — nothing to fit")
    results = [evaluate_corpus_weights(samples, weights) for weights in _weight_grid(step)]
    return sorted(
        results,
        key=lambda r: (-r["mean_split_auc"], -r["min_split_balanced_accuracy"], -r["separation"]),
    )


def choose_weights(
    results: list[dict],
    tolerance: float = 0.005,
    evidence_free_index: int = 2,
    evidence_free_floor: float | None = None,
) -> dict:
    """Pick one triple out of the ranked sweep, breaking plateaus explicitly.

    The objective is usually flat across a wide plateau of triples — several
    weightings separate the classes equally well. Picking the sort's first row
    would silently award that plateau to floating-point noise, so instead:

    1. keep every triple within `tolerance` of the best mean per-split AUC;
    2. drop triples that weight the *evidence-free* sub-signal below
       `evidence_free_floor`, when a floor is given;
    3. prefer the highest minimum per-split balanced accuracy;
    4. then the LOWEST weight on the evidence-free sub-signal;
    5. then the widest separation.

    `evidence_free_index` is the sub-signal no available labeled data can rank
    (default `w_gaze`: no fetched public dataset contains head-forward,
    eyes-to-the-side frames). Steps 2 and 4 pull in opposite directions on
    purpose. Step 4 is the conservative half: where the data cannot justify a
    weight, take the smallest the plateau allows. Step 2 is the product half:
    a weight of exactly zero DELETES a sub-signal from production, which is a
    stronger claim than "unmeasured" — it asserts the signal is worthless,
    including for the inattention mode (`gaze_off`) that the corpus happens
    not to contain. A floor of one grid step keeps the signal alive at the
    smallest weight the evidence permits. Pass `None` to allow zero.

    If the floor would empty the plateau, it is ignored rather than applied to
    an arbitrary lower-AUC row; the returned dict's "evidence_free_floor_applied"
    records which happened.
    """
    if not results:
        raise ValueError("No sweep results to choose from")
    best = max(r["mean_split_auc"] for r in results)
    plateau = [r for r in results if r["mean_split_auc"] >= best - tolerance]

    floored = plateau
    floor_applied = False
    if evidence_free_floor is not None:
        floored = [r for r in plateau if r["weights"][evidence_free_index] >= evidence_free_floor]
        floor_applied = bool(floored)
        if not floored:
            floored = plateau

    chosen = min(
        floored,
        key=lambda r: (
            -r["min_split_balanced_accuracy"],
            r["weights"][evidence_free_index],
            -r["separation"],
        ),
    )
    chosen = dict(chosen)
    chosen["plateau_size"] = len(plateau)
    chosen["evidence_free_floor"] = evidence_free_floor
    chosen["evidence_free_floor_applied"] = floor_applied
    return chosen


def format_corpus_report(
    results: list[dict],
    coverage: dict,
    chosen: dict | None = None,
    top_n: int = 10,
    holdout: dict | None = None,
) -> str:
    """Human-readable report for corpus mode."""
    chosen = chosen if chosen is not None else results[0]
    lines = ["Video engagement weight search — labeled-frame corpus", ""]
    lines.append(f"Frames in manifest : {coverage['frames_total']}")
    lines.append(f"Used in fit        : {coverage['used']}")
    lines.append(f"Dropped (no face on a scoring label): {coverage['no_detection_on_scoring_label']}")
    lines.append(
        "absent frames      : "
        f"{coverage['absent_correctly_undetected']} correctly undetected, "
        f"{coverage['absent_falsely_detected']} falsely detected"
    )
    lines.append("")
    lines.append("Detector coverage per label:")
    for label, total in sorted(coverage["per_label_total"].items()):
        detected = coverage["per_label_detected"].get(label, 0)
        lines.append(f"  {label:<13} {detected}/{total} frames with a detected face")
    if coverage.get("per_split_used"):
        lines.append("")
        lines.append("Samples per split (after detection):")
        for split, n in sorted(coverage["per_split_used"].items()):
            lines.append(f"  {split:<24} {n}")

    lines.append("")
    lines.append(f"Top {top_n} weight triples (mean per-split AUC, then min split bal.acc, then separation):")
    lines.append(
        f"{'w_eyes':>7} {'w_head':>7} {'w_gaze':>7} {'mAUC':>7} {'minBA':>7} "
        f"{'pooled':>7} {'sep':>7} {'thresh':>7}"
    )
    for row in results[:top_n]:
        w1, w2, w3 = row["weights"]
        lines.append(
            f"{w1:7.2f} {w2:7.2f} {w3:7.2f} {row['mean_split_auc']:7.4f} "
            f"{row['min_split_balanced_accuracy']:7.4f} {row['auc']:7.4f} "
            f"{row['separation']:7.4f} {row['threshold']['threshold']:7.4f}"
        )

    lines.append("")
    lines.append(f"CHOSEN weights   : {tuple(round(w, 4) for w in chosen['weights'])}")
    lines.append(
        f"CHOSEN threshold : {chosen['threshold']['threshold']:.4f} "
        "(predict attentive when fused score >= threshold)"
    )
    lines.append(
        f"  mean per-split AUC {chosen['mean_split_auc']:.4f} | pooled AUC {chosen['auc']:.4f} | "
        f"pooled balanced accuracy {chosen['balanced_accuracy']:.4f} | "
        f"tpr {chosen['threshold']['tpr']:.4f} | fpr {chosen['threshold']['fpr']:.4f} | "
        f"pearson r {chosen['pearson_r']:.4f}"
    )
    lines.append("  Mean fused score per label (pooled):")
    for label, mean in chosen["mean_score_per_label"].items():
        lines.append(f"    {label:<13} {mean:.4f}")

    for split, metrics in sorted(chosen.get("per_split", {}).items()):
        lines.append("")
        lines.append(f"  Split {split} (n={metrics['n']}):")
        if metrics.get("single_class"):
            lines.append("    only one class present — reported, not fitted")
        else:
            lines.append(
                f"    AUC {metrics['auc']:.4f} | best-threshold bal.acc {metrics['balanced_accuracy']:.4f} "
                f"| own best threshold {metrics['threshold']['threshold']:.4f} "
                f"| separation {metrics['separation']:.4f}"
            )
        for label, mean in metrics["mean_score_per_label"].items():
            lines.append(f"    mean score {label:<13} {mean:.4f}")
        at_pooled = chosen.get("per_split_at_pooled_threshold", {}).get(split)
        if at_pooled:
            lines.append(
                f"    at the pooled threshold: tpr {at_pooled['tpr']:.4f} | fpr {at_pooled['fpr']:.4f} "
                f"| bal.acc {at_pooled['balanced_accuracy']:.4f}"
            )

    if holdout:
        lines.append("")
        lines.append(f"HELD-OUT groups (never seen by the fit), n={holdout['n']}:")
        if "auc" in holdout:
            lines.append(f"  AUC {holdout['auc']:.4f}")
        at_fit = holdout["at_fitted_threshold"]
        lines.append(
            f"  at the fitted threshold: tpr {at_fit['tpr']:.4f} | fpr {at_fit['fpr']:.4f} "
            f"| bal.acc {at_fit['balanced_accuracy']:.4f} | accuracy {at_fit['accuracy']:.4f}"
        )
        for label, mean in holdout["mean_score_per_label"].items():
            lines.append(f"  mean score {label:<13} {mean:.4f}")
        for split, block in sorted(holdout["per_split"].items()):
            at_split = block["at_fitted_threshold"]
            auc_text = f"AUC {block['auc']:.4f} | " if "auc" in block else ""
            lines.append(
                f"  {split} (n={block['n']}): {auc_text}tpr {at_split['tpr']:.4f} | "
                f"fpr {at_split['fpr']:.4f} | bal.acc {at_split['balanced_accuracy']:.4f}"
            )
    return "\n".join(lines)


def partition_by_group(samples: list[dict], holdout_fraction: float = 0.3) -> tuple[list[dict], list[dict]]:
    """Split samples into (fit, holdout) with no group in both halves.

    A group is a subject (head-pose splits) or a shard (the eye-state split).
    Splitting by group rather than by frame is the whole point: consecutive
    frames of one person are near-duplicates, so a random frame split would
    report how well the weights memorised those faces, not whether they
    transfer.

    Groups are held out per split AND per majority class. The class
    stratification matters: in the eye-state dataset every shard is
    single-class, so holding out "the last 30% of shards" would produce a
    holdout of positives only, where false-positive rate is undefined. Within
    each (split, majority class) bucket the LAST `holdout_fraction` of the
    sorted group names is held out — deterministic and re-checkable, with no
    RNG seed to remember. A bucket with only one group keeps it in the fit
    half; `coverage["holdout_groups"]` records exactly what went where.
    """
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    if holdout_fraction == 0.0:
        return list(samples), []

    # (split, group) -> class counts, to bucket each group by its majority class.
    group_counts: dict[tuple[str, str], list[int]] = {}
    for sample in samples:
        key = (sample.get("split", "corpus"), sample.get("group", sample.get("split", "corpus")))
        counts = group_counts.setdefault(key, [0, 0])
        counts[0 if sample["target"] >= 0.5 else 1] += 1

    buckets: dict[tuple[str, str], list[str]] = {}
    for (split, group), (pos, neg) in group_counts.items():
        majority = "attentive" if pos >= neg else "inattentive"
        buckets.setdefault((split, majority), []).append(group)

    holdout_groups: set[tuple[str, str]] = set()
    for (split, _majority), groups in buckets.items():
        ordered = sorted(groups)
        if len(ordered) < 2:
            continue
        n_holdout = max(1, int(round(len(ordered) * holdout_fraction)))
        n_holdout = min(n_holdout, len(ordered) - 1)  # never empty the fit half
        for group in ordered[-n_holdout:]:
            holdout_groups.add((split, group))

    fit, holdout = [], []
    for sample in samples:
        key = (sample.get("split", "corpus"), sample.get("group", sample.get("split", "corpus")))
        (holdout if key in holdout_groups else fit).append(sample)
    return fit, holdout


def evaluate_holdout(holdout: list[dict], weights: tuple[float, float, float], threshold: float) -> dict:
    """Score a held-out set with weights and a threshold fitted elsewhere.

    This is the only number in the report that was not optimised for.
    """
    scores = [video_attention_score(s["raw_signals"], weights) for s in holdout]
    targets = [s["target"] for s in holdout]
    labels = [s["label"] for s in holdout]
    result = {
        "n": len(holdout),
        "mean_score_per_label": mean_score_per_label(scores, labels),
        "at_fitted_threshold": metrics_at_threshold(scores, targets, threshold),
    }
    if len({t >= 0.5 for t in targets}) == 2:
        result["auc"] = roc_auc(scores, targets)
    per_split = {}
    for split in sorted({s.get("split", "corpus") for s in holdout}):
        picked = [i for i, s in enumerate(holdout) if s.get("split", "corpus") == split]
        split_scores = [scores[i] for i in picked]
        split_targets = [targets[i] for i in picked]
        block = {
            "n": len(picked),
            "at_fitted_threshold": metrics_at_threshold(split_scores, split_targets, threshold),
        }
        if len({t >= 0.5 for t in split_targets}) == 2:
            block["auc"] = roc_auc(split_scores, split_targets)
        per_split[split] = block
    result["per_split"] = per_split
    return result


def compact_result(result: dict) -> dict:
    """One sweep row, trimmed for the JSON file.

    The full per-split blocks are kept only for the chosen triple; storing them
    for all ~231 grid points makes the results file an order of magnitude
    larger without adding anything a reader would use.
    """
    return {
        "weights": list(result["weights"]),
        "mean_split_auc": result["mean_split_auc"],
        "min_split_auc": result["min_split_auc"],
        "min_split_balanced_accuracy": result["min_split_balanced_accuracy"],
        "pooled_auc": result["auc"],
        "pooled_balanced_accuracy": result["balanced_accuracy"],
        "separation": result["separation"],
        "pearson_r": result["pearson_r"],
        "threshold": result["threshold"]["threshold"],
        "per_split_auc": {
            split: metrics.get("auc") for split, metrics in result["per_split"].items()
        },
    }


def run_corpus_mode(
    corpus_dirs: list[str | Path] | str | Path,
    step: float,
    results_path: str | Path,
    top_n: int = 10,
    detector=None,
    tolerance: float = 0.005,
    gaze_floor: float | None = None,
    holdout_fraction: float = 0.3,
) -> dict:
    """Load one or more labeled corpora, score, sweep, print and persist.

    Each corpus directory becomes its own named split (the directory name), so
    per-dataset results stay separable. Returns the JSON payload written.
    """
    if isinstance(corpus_dirs, (str, Path)):
        corpus_dirs = [corpus_dirs]

    records: list[dict] = []
    for corpus_dir in corpus_dirs:
        records.extend(load_corpus_manifest(corpus_dir))

    samples, coverage = score_corpus_frames(records, detector=detector)
    per_split_used: dict[str, int] = {}
    for sample in samples:
        split = sample.get("split", "corpus")
        per_split_used[split] = per_split_used.get(split, 0) + 1
    coverage["per_split_used"] = per_split_used

    fit_samples, holdout_samples = partition_by_group(samples, holdout_fraction)
    coverage["fit_samples"] = len(fit_samples)
    coverage["holdout_samples"] = len(holdout_samples)
    coverage["holdout_groups"] = sorted(
        {(s.get("split", "corpus"), s.get("group")) for s in holdout_samples}
    )

    results = sweep_corpus_weights(fit_samples, step=step)
    chosen = choose_weights(results, tolerance=tolerance, evidence_free_floor=gaze_floor)
    holdout = (
        evaluate_holdout(holdout_samples, chosen["weights"], chosen["threshold"]["threshold"])
        if holdout_samples
        else None
    )
    print(format_corpus_report(results, coverage, chosen=chosen, top_n=top_n, holdout=holdout))

    payload = {
        "mode": "corpus",
        "corpus_dirs": [str(Path(d).resolve()) for d in corpus_dirs],
        "grid_step": step,
        "plateau_tolerance": tolerance,
        "holdout_fraction": holdout_fraction,
        "evidence_free_signal": "gaze_centered",
        "evidence_free_floor": gaze_floor,
        "evidence_free_floor_applied": chosen["evidence_free_floor_applied"],
        "plateau_size": chosen["plateau_size"],
        "label_targets": LABEL_TARGETS,
        "objective": (
            "max mean per-split ROC-AUC; within plateau_tolerance of the best, keep w_gaze >= "
            "evidence_free_floor, then max min per-split balanced accuracy, then lowest w_gaze, "
            "then widest class mean separation"
        ),
        "coverage": coverage,
        "chosen": {
            "weights": list(chosen["weights"]),
            "threshold": chosen["threshold"]["threshold"],
            "mean_split_auc": chosen["mean_split_auc"],
            "pooled_auc": chosen["auc"],
            "balanced_accuracy": chosen["balanced_accuracy"],
            "separation": chosen["separation"],
            "pearson_r": chosen["pearson_r"],
            "tpr": chosen["threshold"]["tpr"],
            "fpr": chosen["threshold"]["fpr"],
            "mean_score_per_label": chosen["mean_score_per_label"],
            "per_split": chosen["per_split"],
            "per_split_at_pooled_threshold": chosen["per_split_at_pooled_threshold"],
            "plateau_size": chosen["plateau_size"],
        },
        "holdout": holdout,
        "all_results": [compact_result(r) for r in results],
    }
    results_file = Path(results_path)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nWrote machine-readable results to {results_file}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", help="one or more labeled-frame corpus dirs (corpus mode); each becomes a named split. Omit for backend-correlation mode")
    parser.add_argument("--holdout-fraction", type=float, default=0.3, help="fraction of each split's GROUPS (subjects/shards) held out of the fit")
    parser.add_argument("--gaze-floor", type=float, default=None, help="smallest w_gaze the chooser may pick (default: one grid step; 0 allows dropping the gaze signal)")
    parser.add_argument("--tolerance", type=float, default=0.005, help="plateau width on mean per-split AUC when breaking ties")
    parser.add_argument("--step", type=float, default=None, help="weight grid step (default 0.05 corpus / 0.1 backend)")
    parser.add_argument("--top", type=int, default=10, help="rows to print in the corpus report")
    parser.add_argument("--results-json", default="../../docs/video_engagement/weight-search-results.json")
    parser.add_argument("--barter-id", type=int)
    parser.add_argument("--learner-user-id", type=int)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--backend", default="local", choices=["local", "aws"])
    parser.add_argument("--out", default="../../docs/video_engagement/backend-correlation-results.md")
    args = parser.parse_args()

    if args.corpus:
        run_corpus_mode(
            args.corpus,
            step=args.step if args.step is not None else 0.05,
            results_path=args.results_json,
            top_n=args.top,
            tolerance=args.tolerance,
            gaze_floor=args.gaze_floor if args.gaze_floor is not None else (args.step if args.step is not None else 0.05),
            holdout_fraction=args.holdout_fraction,
        )
        raise SystemExit(0)

    if args.barter_id is None or args.learner_user_id is None:
        raise SystemExit(
            "Backend-correlation mode needs --barter-id and --learner-user-id "
            "(or use --corpus <dir> for the offline labeled-frame experiment)."
        )

    step = args.step if args.step is not None else 0.1
    raw_signals, speech_scores = _fetch_session_data(
        args.backend_url, args.barter_id, args.learner_user_id, args.backend
    )
    if len(raw_signals) < 2:
        raise SystemExit("Not enough data yet — need a pilot session with real windows first.")

    ground_truth = speech_scores[: len(raw_signals)]
    subsignal_results = sweep_subsignal_weights(raw_signals, ground_truth, step=step)

    video_scores = [video_attention_score(s, subsignal_results[0]["weights"]) for s in raw_signals]
    fusion_results = None
    if len(speech_scores) >= 2:
        fusion_results = sweep_fusion_weights(
            video_scores[: len(speech_scores)], speech_scores[: len(video_scores)], step=step
        )

    _write_design_choices(args.out, subsignal_results, fusion_results)
    print(f"Wrote {args.out}")
