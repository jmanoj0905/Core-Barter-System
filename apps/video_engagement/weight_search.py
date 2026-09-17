"""Offline experiment: pick sub-signal and fusion weights by correlating
video_attention_score against the existing speech-based engagement_score
(the only per-learner ground truth this repo has — see spec section
"Weight search"). Not run automatically; invoke manually against a pilot
session's data:

    python weight_search.py --barter-id 1 --backend-url http://localhost:8000
"""

import argparse
import itertools
import math
import statistics

import httpx

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
        w3 = round(1.0 - w1 - w2, 4)
        if 0.0 <= w3 <= 1.0:
            grid.append((w1, w2, round(w3, 4)))
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


def _fetch_session_data(backend_url: str, barter_id: int) -> tuple[list[dict], list[float]]:
    video_rows = httpx.get(f"{backend_url}/session/{barter_id}/video-engagement").json()
    history_rows = httpx.get(f"{backend_url}/session/{barter_id}/engagement/history").json()

    raw_signals = [r["raw_signals"] for r in video_rows]
    speech_scores = [
        r["speech_engagement_score"] for r in history_rows
        if r["speech_engagement_score"] is not None
    ]
    return raw_signals, speech_scores


def _write_design_choices(path: str, subsignal_results: list[dict], fusion_results: list[dict] | None):
    lines = ["# Video Engagement — Weight Design Choices\n"]
    lines.append("Every weight below was picked by grid search + correlation against the\n"
                  "existing speech-based `engagement_score`, not hand-picked. See\n"
                  "`weight_search.py` and the design spec for the full method.\n")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--barter-id", type=int, required=True)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--out", default="../../docs/video_engagement/design-choices.md")
    args = parser.parse_args()

    raw_signals, speech_scores = _fetch_session_data(args.backend_url, args.barter_id)
    if len(raw_signals) < 2:
        raise SystemExit("Not enough data yet — need a pilot session with real windows first.")

    ground_truth = speech_scores[: len(raw_signals)]
    subsignal_results = sweep_subsignal_weights(raw_signals, ground_truth, step=args.step)

    video_scores = [video_attention_score(s, subsignal_results[0]["weights"]) for s in raw_signals]
    fusion_results = None
    if len(speech_scores) >= 2:
        fusion_results = sweep_fusion_weights(
            video_scores[: len(speech_scores)], speech_scores[: len(video_scores)], step=args.step
        )

    _write_design_choices(args.out, subsignal_results, fusion_results)
    print(f"Wrote {args.out}")
