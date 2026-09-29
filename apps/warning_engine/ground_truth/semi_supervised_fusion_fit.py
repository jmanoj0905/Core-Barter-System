import argparse
import asyncio
import csv

from fit_fusion_weights import (
    _sigmoid,
    accuracy_at_threshold,
    fit_logistic_regression,
    roc_auc,
    split_by_scenario,
)


def self_train(
    labeled_rows: list[dict], unlabeled_rows: list[dict],
    confidence_margin: float = 0.3, max_rounds: int = 3,
    learning_rate: float = 0.1, epochs: int = 1000,
) -> tuple[float, float, float]:
    training_set = list(labeled_rows)
    remaining = list(unlabeled_rows)
    a, b, bias = fit_logistic_regression(training_set, learning_rate, epochs)

    for _ in range(max_rounds):
        if not remaining:
            break
        confident, still_unlabeled = [], []
        for row in remaining:
            pred = _sigmoid(a * row["speech_score"] + b * row["video_score"] + bias)
            if abs(pred - 0.5) >= confidence_margin:
                confident.append({**row, "engaged_label": 1 if pred >= 0.5 else 0})
            else:
                still_unlabeled.append(row)
        if not confident:
            break
        training_set.extend(confident)
        remaining = still_unlabeled
        a, b, bias = fit_logistic_regression(training_set, learning_rate, epochs)

    return a, b, bias


async def fetch_unlabeled_pairs(backend_url: str, barter_ids: list[int], http_client) -> list[dict]:
    pairs = []
    for barter_id in barter_ids:
        resp = await http_client.get(f"{backend_url}/session/{barter_id}/engagement/history")
        resp.raise_for_status()
        for row in resp.json():
            speech, video = row.get("speech_engagement_score"), row.get("video_attention_score")
            if speech is not None and video is not None:
                pairs.append({"speech_score": speech, "video_score": video})
    return pairs


def _load_labeled_corpus(path: str) -> list[dict]:
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "scenario_id": int(row["scenario_id"]),
                "speech_score": float(row["speech_score"]),
                "video_score": float(row["video_score"]),
                "engaged_label": int(row["engaged_label"]),
            })
    return rows


def _load_offline_unlabeled(path: str) -> list[dict]:
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "speech_score": float(row["speech_score"]),
                "video_score": float(row["video_score"]),
            })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Self-training (semi-supervised) extension for fusion-weight fitting"
    )
    parser.add_argument("--corpus", required=True, help="Path to labeled fusion_corpus.csv")
    parser.add_argument("--offline-corpus", help="Path to a CSV of unlabeled speech/video score pairs")
    parser.add_argument("--backend-url", help="Backend base URL to fetch unlabeled pairs from")
    parser.add_argument("--barter-ids", help="Comma-separated barter session IDs to fetch history for")
    parser.add_argument("--confidence-margin", type=float, default=0.3)
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args(argv)

    has_offline = args.offline_corpus is not None
    has_backend = args.backend_url is not None or args.barter_ids is not None
    if has_offline == has_backend:
        parser.error("provide exactly one of --offline-corpus or --backend-url/--barter-ids")
    if has_backend and (args.backend_url is None or args.barter_ids is None):
        parser.error("--backend-url and --barter-ids must be provided together")

    labeled_rows = _load_labeled_corpus(args.corpus)
    train_rows, test_rows = split_by_scenario(labeled_rows, holdout_fraction=0.2, seed=42)

    if has_offline:
        unlabeled_rows = _load_offline_unlabeled(args.offline_corpus)
    else:
        import httpx

        barter_ids = [int(x) for x in args.barter_ids.split(",")]

        async def _fetch():
            async with httpx.AsyncClient() as client:
                return await fetch_unlabeled_pairs(args.backend_url, barter_ids, client)

        unlabeled_rows = asyncio.run(_fetch())

    a, b, bias = self_train(
        train_rows, unlabeled_rows,
        confidence_margin=args.confidence_margin, max_rounds=args.max_rounds,
    )

    fitted_scores = [_sigmoid(a * r["speech_score"] + b * r["video_score"] + bias) for r in test_rows]
    fitted_labels = [r["engaged_label"] for r in test_rows]
    fitted_auc = roc_auc(fitted_scores, fitted_labels)
    fitted_acc = accuracy_at_threshold(fitted_scores, fitted_labels, threshold=0.5)

    baseline_scores = [0.7 * r["speech_score"] + 0.3 * r["video_score"] for r in test_rows]
    baseline_auc = roc_auc(baseline_scores, fitted_labels)
    baseline_acc = accuracy_at_threshold(baseline_scores, fitted_labels, threshold=0.5)

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"bias = {bias}")
    print()
    print("Held-out metrics:")
    print(f"{'Model':<20} {'AUC':<15} {'Accuracy':<15}")
    print("-" * 50)
    print(f"{'Self-trained':<20} {fitted_auc:<15.6f} {fitted_acc:<15.6f}")
    print(f"{'Baseline (0.7/0.3)':<20} {baseline_auc:<15.6f} {baseline_acc:<15.6f}")

    return 0


if __name__ == "__main__":
    exit(main())
