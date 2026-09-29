"""Evaluate detect_meaning_reversal on the labeled reversal corpus.

detect_meaning_reversal is a pure text function, so no embedding model is
needed. Reads reversal_eval_corpus.json (same directory), prints overall
precision / recall / F1, the confusion counts, a per-category table, and every
misclassified text grouped by category.

Usage:
    cd apps/semantic_analysis/ground_truth && python evaluate_reversal_detection.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reversal_detection import detect_meaning_reversal  # noqa: E402

CORPUS_PATH = Path(__file__).parent / "reversal_eval_corpus.json"
CATEGORIES = ("commitment", "reversal", "distractor", "double_negation")


def load_corpus(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def evaluate_corpus(records: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    by_category: dict[str, dict] = {}
    misclassified: list[dict] = []
    for rec in records:
        predicted = detect_meaning_reversal(rec["text"])
        expected = rec["expected_reversal"]
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1
        cat = by_category.setdefault(rec["category"], {"correct": 0, "total": 0})
        cat["total"] += 1
        if predicted == expected:
            cat["correct"] += 1
        else:
            misclassified.append({**rec, "predicted_reversal": predicted})
    # Zero denominators score 1.0 (nothing predicted / nothing to find);
    # main() flags when this happens so it is not read as a real 100%.
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "by_category": by_category,
        "misclassified": misclassified,
    }


def main(argv=None) -> int:
    records = load_corpus(CORPUS_PATH)
    result = evaluate_corpus(records)
    print(f"Corpus: {len(records)} records (synthetic, LLM-generated; not real session data)")
    print(f"tp={result['tp']} fp={result['fp']} fn={result['fn']} tn={result['tn']}")
    print(f"precision={result['precision']:.3f} recall={result['recall']:.3f} f1={result['f1']:.3f}")
    if result["tp"] + result["fp"] == 0:
        print("note: no positive predictions; precision defaults to 1.0 (zero denominator)")
    if result["tp"] + result["fn"] == 0:
        print("note: no positive labels; recall defaults to 1.0 (zero denominator)")
    print()
    print(f"{'category':<17}{'correct':>8}{'total':>7}{'accuracy':>10}")
    for cat in [c for c in CATEGORIES if c in result["by_category"]] + sorted(
        c for c in result["by_category"] if c not in CATEGORIES
    ):
        s = result["by_category"][cat]
        print(f"{cat:<17}{s['correct']:>8}{s['total']:>7}{s['correct'] / s['total']:>10.1%}")
    print()
    print("double_negation is a known, expected-imperfect category: the detector has no")
    print("double-negation handling, so it is reported to measure the limitation, not as")
    print("a target for 100% accuracy.")
    print()
    print(f"Misclassified ({len(result['misclassified'])}):")
    for cat in sorted({m["category"] for m in result["misclassified"]}):
        print(f"[{cat}]")
        for m in result["misclassified"]:
            if m["category"] == cat:
                kind = "false positive" if m["predicted_reversal"] else "false negative"
                print(f"  {kind}: {m['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
