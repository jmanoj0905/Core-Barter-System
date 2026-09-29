import argparse
import csv
import math
import random


def _sigmoid(x: float) -> float:
    if x < -700:  # avoid OverflowError from math.exp on extreme negative x
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def fit_logistic_regression(
    rows: list[dict], learning_rate: float = 0.1, epochs: int = 1000
) -> tuple[float, float, float]:
    a = b = bias = 0.0
    n = len(rows)
    for _ in range(epochs):
        grad_a = grad_b = grad_bias = 0.0
        for row in rows:
            speech, video, label = row["speech_score"], row["video_score"], row["engaged_label"]
            pred = _sigmoid(a * speech + b * video + bias)
            error = pred - label
            grad_a += error * speech
            grad_b += error * video
            grad_bias += error
        a -= learning_rate * grad_a / n
        b -= learning_rate * grad_b / n
        bias -= learning_rate * grad_bias / n
    return a, b, bias


def split_by_scenario(
    rows: list[dict], holdout_fraction: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    scenario_ids = sorted({row["scenario_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(scenario_ids)
    n_holdout = max(1, int(len(scenario_ids) * holdout_fraction))
    holdout_ids = set(scenario_ids[:n_holdout])
    train = [r for r in rows if r["scenario_id"] not in holdout_ids]
    test = [r for r in rows if r["scenario_id"] in holdout_ids]
    return train, test


def roc_auc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5

    # Handle tied scores with mid-rank averaging
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        score = pairs[i][0]
        # Find all pairs with the same score
        j = i
        while j < len(pairs) and pairs[j][0] == score:
            j += 1
        # Ranks span from i+1 to j (1-indexed)
        # Average rank is (i + 1 + j) / 2
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j

    auc = (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
    return auc


def accuracy_at_threshold(scores: list[float], labels: list[int], threshold: float = 0.5) -> float:
    correct = sum(1 for s, l in zip(scores, labels) if (s >= threshold) == (l == 1))
    return correct / len(labels)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fit logistic regression to fusion corpus")
    parser.add_argument("--corpus", required=True, help="Path to fusion_corpus.csv")
    args = parser.parse_args(argv)

    # Load corpus
    rows = []
    with open(args.corpus, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "scenario_id": int(row["scenario_id"]),
                "speech_score": float(row["speech_score"]),
                "video_score": float(row["video_score"]),
                "engaged_label": int(row["engaged_label"]),
            })

    # Split by scenario
    train_rows, test_rows = split_by_scenario(rows, holdout_fraction=0.2, seed=42)

    # Fit logistic regression on training data
    a, b, bias = fit_logistic_regression(train_rows, learning_rate=0.1, epochs=1000)

    # Compute metrics on held-out test split
    fitted_scores = [_sigmoid(a * r["speech_score"] + b * r["video_score"] + bias) for r in test_rows]
    fitted_labels = [r["engaged_label"] for r in test_rows]
    fitted_auc = roc_auc(fitted_scores, fitted_labels)
    fitted_acc = accuracy_at_threshold(fitted_scores, fitted_labels, threshold=0.5)

    # Compute metrics for the fixed 0.7/0.3 baseline on the same test split
    baseline_scores = [0.7 * r["speech_score"] + 0.3 * r["video_score"] for r in test_rows]
    baseline_auc = roc_auc(baseline_scores, fitted_labels)
    baseline_acc = accuracy_at_threshold(baseline_scores, fitted_labels, threshold=0.5)

    # Print results
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"bias = {bias}")
    print()
    print("Held-out metrics:")
    print(f"{'Model':<20} {'AUC':<15} {'Accuracy':<15}")
    print("-" * 50)
    print(f"{'Fitted (logistic)':<20} {fitted_auc:<15.6f} {fitted_acc:<15.6f}")
    print(f"{'Baseline (0.7/0.3)':<20} {baseline_auc:<15.6f} {baseline_acc:<15.6f}")

    return 0


if __name__ == "__main__":
    exit(main())
