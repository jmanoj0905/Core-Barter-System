"""Grid-search UPPER/LOWER cosine thresholds against labeled data.

Reads one or more CSVs shaped like {topic, scope, text, expected_label,
similarity} (synthetic_dataset.csv and/or real_dataset.csv), and finds the
(UPPER, LOWER) pair that maximizes classification accuracy against
expected_label — the same three-way classify() logic main.py uses at
runtime (>= UPPER -> correct, >= LOWER -> weakly_correct, else incorrect).

Usage:
    python calibrate_thresholds.py synthetic_dataset.csv [real_dataset.csv ...]
"""

import csv
import sys
from pathlib import Path

LABELS = ("correct", "weakly_correct", "incorrect")


def classify(similarity: float, upper: float, lower: float) -> str:
    if similarity >= upper:
        return "correct"
    if similarity >= lower:
        return "weakly_correct"
    return "incorrect"


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"Skipping missing file: {p}")
            continue
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if row["expected_label"] not in LABELS:
                    continue
                rows.append({"similarity": float(row["similarity"]), "expected_label": row["expected_label"]})
    return rows


def evaluate(rows: list[dict], upper: float, lower: float) -> tuple[float, dict]:
    correct_count = 0
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    for row in rows:
        predicted = classify(row["similarity"], upper, lower)
        confusion[row["expected_label"]][predicted] += 1
        if predicted == row["expected_label"]:
            correct_count += 1
    accuracy = correct_count / len(rows) if rows else 0.0
    return accuracy, confusion


def grid_search(rows: list[dict]):
    best = None
    step = 0.01
    upper_range = [round(u, 2) for u in [0.30 + i * step for i in range(int((0.90 - 0.30) / step) + 1)]]
    for upper in upper_range:
        lower_range = [round(l, 2) for l in [0.10 + i * step for i in range(int((upper - 0.10) / step))]]
        for lower in lower_range:
            if lower >= upper:
                continue
            accuracy, confusion = evaluate(rows, upper, lower)
            if best is None or accuracy > best[0]:
                best = (accuracy, upper, lower, confusion)
    return best


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    rows = load_rows(sys.argv[1:])
    if not rows:
        print("No labeled rows loaded.")
        return

    print(f"Loaded {len(rows)} labeled examples.")

    current_accuracy, current_confusion = evaluate(rows, upper=0.55, lower=0.35)
    print(f"\nCurrent thresholds (UPPER=0.55, LOWER=0.35): accuracy={current_accuracy:.3f}")

    best_accuracy, best_upper, best_lower, best_confusion = grid_search(rows)
    print(f"Best thresholds found:      UPPER={best_upper}, LOWER={best_lower}: accuracy={best_accuracy:.3f}")

    print("\nConfusion matrix at best thresholds (rows=expected, cols=predicted):")
    header = "expected\\predicted".ljust(20) + "".join(l.ljust(16) for l in LABELS)
    print(header)
    for expected in LABELS:
        row_str = expected.ljust(20) + "".join(str(best_confusion[expected][pred]).ljust(16) for pred in LABELS)
        print(row_str)


if __name__ == "__main__":
    main()
