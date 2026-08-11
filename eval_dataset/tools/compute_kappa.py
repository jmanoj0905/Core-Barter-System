import glob
import json
import os
from collections import defaultdict

from eval_dataset.tools.kappa import session_kappa


def load_session_annotations(annotations_dir: str, session_id: str) -> list[list[str]]:
    files = sorted(glob.glob(os.path.join(annotations_dir, f"{session_id}_*.json")))
    annotator_labels = []
    for path in files:
        with open(path) as f:
            entries = json.load(f)
        ordered = sorted(entries, key=lambda e: e["window_index"])
        annotator_labels.append([e["label"] for e in ordered])
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
        report[session_id] = session_kappa(annotator_labels)
    return report


if __name__ == "__main__":
    report = report_all_sessions("eval_dataset/annotations")
    for session_id, kappa in sorted(report.items()):
        flag = "  <-- BELOW 0.7 TARGET" if kappa < 0.7 else ""
        print(f"{session_id}: kappa={kappa:.3f}{flag}")
