import glob
import json
import os
from collections import defaultdict

from eval_dataset.tools.kappa import session_kappa


def load_session_annotations(annotations_dir: str, session_id: str) -> list[list[str]]:
    files = sorted(glob.glob(os.path.join(annotations_dir, f"{session_id}_*.json")))
    per_annotator_by_index = []
    for path in files:
        with open(path) as f:
            entries = json.load(f)
        per_annotator_by_index.append({e["window_index"]: e["label"] for e in entries})

    common_indices = None
    for by_index in per_annotator_by_index:
        indices = set(by_index.keys())
        common_indices = indices if common_indices is None else (common_indices & indices)
    common_indices = sorted(common_indices) if common_indices else []

    annotator_labels = [
        [by_index[idx] for idx in common_indices] for by_index in per_annotator_by_index
    ]
    return annotator_labels


def report_all_sessions(annotations_dir: str) -> dict[str, float]:
    sessions = defaultdict(int)
    for path in glob.glob(os.path.join(annotations_dir, "*.json")):
        basename = os.path.basename(path).removesuffix(".json")
        session_id = basename.rsplit("_", 1)[0]
        sessions[session_id] += 1

    report = {}
    for session_id, count in sessions.items():
        if count < 2:
            continue
        annotator_labels = load_session_annotations(annotations_dir, session_id)
        if not annotator_labels or not annotator_labels[0]:
            # no common windows across all annotators for this session — skip
            continue
        report[session_id] = session_kappa(annotator_labels)
    return report


def overall_kappa(report: dict[str, float]) -> float:
    values = list(report.values())
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    report = report_all_sessions("eval_dataset/annotations")
    for session_id, kappa in sorted(report.items()):
        flag = "  <-- BELOW 0.7 TARGET" if kappa < 0.7 else ""
        print(f"{session_id}: kappa={kappa:.3f}{flag}")

    overall = overall_kappa(report)
    overall_flag = "  <-- BELOW 0.7 TARGET" if overall < 0.7 else ""
    print(f"\nOverall kappa: {overall:.3f}{overall_flag}")
