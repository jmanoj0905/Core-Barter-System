"""Compare frozen-score global and group thresholds without fitting on test rows.

Real evaluation: --calibration cal.csv --evaluation eval.csv --output report.json
Fixture diagnostic: --fixture-diagnostic synthetic_dataset.csv --output report.json
Uses only the standard library; it never loads, trains, or deploys an encoder.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from calibrate_thresholds import LABELS, classify, grid_search

CURRENT = (0.55, 0.35)  # Production baseline at experiment creation, 2026-09-19.
MIN_SUPPORT = 2  # Per boundary class; a guard against unsupported fits, not a power claim.


def contract_key(row):
    return json.dumps([row["topic"], row["scope"]], ensure_ascii=False)


def group_key(row, group_by):
    return row["family"] if group_by == "family" else contract_key(row)


def example_key(row):
    return (contract_key(row), " ".join(row["text"].casefold().split()))


def read_rows(path, *, fixture=False, group_by="topic"):
    rows, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for line, raw in enumerate(csv.DictReader(stream), 2):
            row = {k: (v or "").strip() for k, v in raw.items() if k is not None}
            row["row_id"] = f"{Path(path).name}:{line}"
            required = ["topic", "text", "expected_label", "similarity"]
            if not fixture:
                required.append("session_id")
            if group_by == "family":
                required.append("family")
            for field in required:
                if not row.get(field):
                    raise ValueError(f"{row['row_id']}: missing {field}")
            row.setdefault("scope", "")
            if row["expected_label"] not in LABELS:
                raise ValueError(f"{row['row_id']}: unsupported expected_label")
            try:
                row["similarity"] = float(row["similarity"])
            except ValueError as exc:
                raise ValueError(f"{row['row_id']}: invalid similarity") from exc
            if not math.isfinite(row["similarity"]) or not -1 <= row["similarity"] <= 1:
                raise ValueError(f"{row['row_id']}: similarity must be finite and within [-1, 1]")
            if fixture and row.get("source") != "synthetic":
                raise ValueError("Fixture diagnostic requires source=synthetic on every row")
            if example_key(row) in seen:
                raise ValueError(f"{row['row_id']}: duplicate contract/text; deduplicate first")
            seen.add(example_key(row))
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no labeled rows")
    return rows


def validate_split(calibration, evaluation, group_by):
    if {r["session_id"] for r in calibration} & {r["session_id"] for r in evaluation}:
        raise ValueError("Calibration and evaluation share session_id values")
    if {example_key(r) for r in calibration} & {example_key(r) for r in evaluation}:
        raise ValueError("Calibration and evaluation contain duplicate contract/text examples")
    if group_by == "family":
        families = defaultdict(set)
        for row in calibration + evaluation:
            families[contract_key(row)].add(row["family"])
        if any(len(values) > 1 for values in families.values()):
            raise ValueError("Inconsistent family assignment for the same contract")


def fit(calibration, group_by):
    if set(r["expected_label"] for r in calibration) != set(LABELS):
        raise ValueError("Global calibration requires examples of all three labels")
    _, upper, lower, _ = grid_search(calibration, reference=CURRENT)
    groups = defaultdict(list)
    for row in calibration:
        groups[group_key(row, group_by)].append(row)
    local = {}
    for key, rows in sorted(groups.items()):
        counts = Counter(r["expected_label"] for r in rows)
        if min(counts["correct"], counts["weakly_correct"]) < MIN_SUPPORT:
            continue
        _, local_upper, _, _ = grid_search(rows, fixed_lower=lower, reference=(upper, lower))
        local[key] = {"upper": local_upper, "lower": lower, "label_counts": dict(counts)}
    return {"global": {"upper": upper, "lower": lower}, "groups": local}


def predict(rows, thresholds, group_by, fold=None):
    predictions = []
    global_pair = thresholds["global"]
    for row in rows:
        key = group_key(row, group_by)
        local = thresholds["groups"].get(key, global_pair)
        score = row["similarity"]
        predictions.append({
            "row_id": row["row_id"], "contract": contract_key(row), "group": key,
            "fold": fold, "expected_label": row["expected_label"], "similarity": score,
            "current": classify(score, *CURRENT),
            "global": classify(score, global_pair["upper"], global_pair["lower"]),
            "grouped": classify(score, local["upper"], local["lower"]),
            "global_upper": global_pair["upper"], "global_lower": global_pair["lower"],
            "grouped_upper": local["upper"], "grouped_lower": local["lower"],
            "fallback": key not in thresholds["groups"],
        })
    return predictions


def metrics(predictions, method):
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    for row in predictions:
        confusion[row["expected_label"]][row[method]] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[a][label] for a in LABELS)
        per_class[label] = {
            "support": support, "precision": tp / predicted if predicted else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / (support + predicted) if support + predicted else 0.0,
        }
    correct = sum(confusion[label][label] for label in LABELS)
    boundary_n = sum(per_class[label]["support"] for label in LABELS[:2])
    false_incorrect = sum(confusion[label]["incorrect"] for label in LABELS[:2])
    return {
        "n": len(predictions), "correct": correct, "accuracy": correct / len(predictions),
        "macro_f1": sum(p["f1"] for p in per_class.values()) / 3,
        "boundary_accuracy": (sum(confusion[l][l] for l in LABELS[:2]) / boundary_n
                              if boundary_n else None),
        "false_incorrect_rate": false_incorrect / boundary_n if boundary_n else None,
        "per_class": per_class, "confusion": confusion,
    }


def summarize(predictions):
    improved = sum(p["grouped"] == p["expected_label"] != p["global"] for p in predictions)
    regressed = sum(p["global"] == p["expected_label"] != p["grouped"] for p in predictions)
    by_topic = defaultdict(list)
    for row in predictions:
        by_topic[row["contract"]].append(row)
    methods = ("current", "global", "grouped")
    return {
        "metrics": {method: metrics(predictions, method) for method in methods},
        "per_topic": {key: {m: metrics(rows, m) for m in methods}
                      for key, rows in sorted(by_topic.items())},
        "paired_changes": {"improved": improved, "regressed": regressed,
                           "net_correct": improved - regressed},
        "fallback_count": sum(p["fallback"] for p in predictions),
        "predictions": predictions,
    }


def compare(calibration, evaluation, group_by):
    validate_split(calibration, evaluation, group_by)
    thresholds = fit(calibration, group_by)
    return {"protocol": "separate_calibration_evaluation", "group_by": group_by,
            "calibration_n": len(calibration), "thresholds": thresholds,
            **summarize(predict(evaluation, thresholds, group_by))}


def fixture_diagnostic(rows):
    topics = sorted({r["topic"] for r in rows})
    if len(topics) < 2:
        raise ValueError("Fixture diagnostic requires at least two topics")
    example_predictions, topic_predictions = [], []
    for i, row in enumerate(rows):
        calibration = rows[:i] + rows[i + 1:]
        example_predictions.extend(predict([row], fit(calibration, "topic"), "topic", row["row_id"]))
    for topic in topics:
        calibration = [r for r in rows if r["topic"] != topic]
        evaluation = [r for r in rows if r["topic"] == topic]
        topic_predictions.extend(predict(evaluation, fit(calibration, "topic"), "topic", topic))
    return {
        "protocol": "synthetic_sentence_diagnostic", "production_evidence": False,
        "limitations": [
            "Single-sentence synthetic fixtures, not independently labeled 25-second windows.",
            "Leave-one-example-out shares authors and topics; it is not session-held-out evidence.",
            "Cross-validation folds overlap; 60 predictions do not represent 60 independent topics.",
            "Existing fixture labels have unresolved correct/weakly_correct ambiguity.",
        ],
        "leave_one_example_out": summarize(example_predictions),
        "leave_one_topic_out": summarize(topic_predictions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--fixture-diagnostic", type=Path)
    parser.add_argument("--group-by", choices=("topic", "family"), default="topic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fixture_diagnostic:
        if args.calibration or args.evaluation or args.group_by != "topic":
            parser.error("--fixture-diagnostic is separate from real-data or family evaluation")
        paths = [args.fixture_diagnostic]
    else:
        if not args.calibration or not args.evaluation:
            parser.error("Supply both --calibration and --evaluation")
        paths = [args.calibration, args.evaluation]
    if args.output.resolve() in {p.resolve() for p in paths}:
        parser.error("Output must not overwrite an input dataset")
    try:
        if args.fixture_diagnostic:
            report = fixture_diagnostic(read_rows(args.fixture_diagnostic, fixture=True))
        else:
            report = compare(read_rows(args.calibration, group_by=args.group_by),
                             read_rows(args.evaluation, group_by=args.group_by), args.group_by)
        report["inputs"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                            for p in paths]
        report["configuration"] = {
            "encoder": "unchanged; uses supplied cosine scores",
            "objective": "calibration accuracy", "grid": "-1.00 to 1.01, step 0.01",
            "tie_break": "closest to current thresholds globally; closest to global within group",
            "minimum_group_examples_per_boundary_class": MIN_SUPPORT,
            "current_baseline": {"upper": CURRENT[0], "lower": CURRENT[1]},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")
    summaries = ([('leave_one_example_out', report['leave_one_example_out']),
                  ('leave_one_topic_out', report['leave_one_topic_out'])]
                 if args.fixture_diagnostic else [('held_out', report)])
    for name, summary in summaries:
        print(name)
        for method, values in summary["metrics"].items():
            print(f"  {method}: {values['correct']}/{values['n']} correct; "
                  f"macro-F1={values['macro_f1']:.3f}; "
                  f"weak-F1={values['per_class']['weakly_correct']['f1']:.3f}")
    if args.fixture_diagnostic:
        print("DIAGNOSTIC ONLY: no independent session-window evaluation data was used.")


if __name__ == "__main__":
    main()
