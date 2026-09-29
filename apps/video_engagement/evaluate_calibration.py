"""Does per-user baseline calibration improve head-pose attention decisions?

Evaluates the personalization transform in isolation on the Head Pose Image
Database (see `public_datasets.py`). For each subject, the frontal frame
(pan == 0 and tilt == 0) plays the role of the calibration frame: its raw
`head_ratio` becomes the subject's baseline. Every other frame with an
unambiguous label (`head_pose_label`) is then classified two ways:

* uncalibrated - `head_deviation` against the fixed 0.5 centre (today's behaviour)
* calibrated   - `head_deviation` against the subject's own frontal `head_ratio`

with the midpoint decision rule `head_deviation < 0.5` => attentive. The fitted
`ATTENTIVE_SCORE_THRESHOLD` is deliberately NOT used: it applies to the fused
score, and this script isolates the head-deviation transform only.

Metric: balanced accuracy = mean(TPR, TNR), where positive = "attentive". A
subject (or the pool) that lacks either class has no defined balanced accuracy;
`balanced_accuracy` returns None then, `evaluate_subject` returns None for such
a subject, and the pooled per-subject mean skips it. Besides that per-subject
mean, a pooled balanced accuracy over all frames together is reported.

Run (from apps/video_engagement, with the service venv):

    venv/bin/python evaluate_calibration.py --corpus datasets/head_pose_image_database
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from public_datasets import head_pose_label, parse_head_pose_filename
from scoring import _deviation_from_baseline, raw_ratios_from_landmarks

DECISION_THRESHOLD = 0.5
UNCALIBRATED_BASELINE = 0.5

_face_mesh = None


def _get_face_mesh():
    """Build the Face Mesh lazily, with the same settings as `main._face_mesh`."""
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp  # lazy: keeps this module importable without mediapipe

        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _face_mesh


def subject_key(path: Path) -> str:
    """Subject id such as "person01" (same form as `convert_head_pose`'s `group`)."""
    return f"person{parse_head_pose_filename(path.name)['person']:02d}"


def group_frames_by_subject(image_paths: list[Path]) -> dict[str, list[Path]]:
    """Group frames by subject, skipping filenames the dataset parser rejects."""
    grouped: dict[str, list[Path]] = {}
    for path in image_paths:
        try:
            key = subject_key(path)
        except ValueError:
            continue
        grouped.setdefault(key, []).append(path)
    return grouped


def _raw_ratios_for_file(path: Path) -> dict | None:
    """Raw head/gaze ratios for one JPEG, or None if unreadable / no face / degenerate."""
    import cv2  # lazy
    import numpy as np

    img = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    result = _get_face_mesh().process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None
    height, width = img.shape[:2]
    face = result.multi_face_landmarks[0]
    landmarks = {i: (lm.x * width, lm.y * height) for i, lm in enumerate(face.landmark)}
    ratios = raw_ratios_from_landmarks(landmarks)
    if ratios["head_ratio"] is None:
        return None
    return ratios


def balanced_accuracy(truth: list[bool], predicted: list[bool]) -> float | None:
    """Mean of TPR and TNR (positive = attentive); None if either class is absent."""
    positives = [p for t, p in zip(truth, predicted) if t]
    negatives = [p for t, p in zip(truth, predicted) if not t]
    if not positives or not negatives:
        return None
    tpr = sum(positives) / len(positives)
    tnr = sum(1 for p in negatives if not p) / len(negatives)
    return (tpr + tnr) / 2.0


def _is_frontal(pose: dict) -> bool:
    return pose["pan"] == 0 and pose["tilt"] == 0


def evaluate_subject(subject_frames: list[Path]) -> dict | None:
    """Balanced accuracy, uncalibrated vs. calibrated, for one subject.

    Returns None when: there is no frontal frame with a detected face to serve
    as the calibration frame; fewer than 2 other frames are labelled and
    detected; or those frames contain only one class (balanced accuracy is
    undefined). The result also carries `outcomes` ([truth, uncalibrated,
    calibrated] per frame) so callers can pool across subjects.
    """
    parsed = []
    for path in subject_frames:
        try:
            parsed.append((path, parse_head_pose_filename(Path(path).name)))
        except ValueError:
            continue

    baseline, baseline_path = None, None
    for path, pose in sorted(parsed, key=lambda item: (item[1]["serie"], item[1]["number"])):
        if not _is_frontal(pose):
            continue
        ratios = _raw_ratios_for_file(path)
        if ratios is not None and ratios["head_ratio"] is not None:
            baseline, baseline_path = ratios, path
            break
    if baseline is None:
        return None

    outcomes = []
    for path, pose in parsed:
        if path == baseline_path:
            continue
        label = head_pose_label(pose["tilt"], pose["pan"])
        if label is None:
            continue
        ratios = _raw_ratios_for_file(path)
        if ratios is None or ratios["head_ratio"] is None:
            continue
        uncalibrated = _deviation_from_baseline(ratios["head_ratio"], UNCALIBRATED_BASELINE)
        calibrated = _deviation_from_baseline(ratios["head_ratio"], baseline["head_ratio"])
        outcomes.append(
            [
                label == "attentive",
                uncalibrated < DECISION_THRESHOLD,
                calibrated < DECISION_THRESHOLD,
            ]
        )
    if len(outcomes) < 2:
        return None

    truth = [o[0] for o in outcomes]
    before = balanced_accuracy(truth, [o[1] for o in outcomes])
    after = balanced_accuracy(truth, [o[2] for o in outcomes])
    if before is None or after is None:
        return None
    return {
        "subject": subject_key(baseline_path),
        "n_frames": len(outcomes),
        "balanced_accuracy_uncalibrated": before,
        "balanced_accuracy_calibrated": after,
        "outcomes": outcomes,
    }


def summarize(results: list[dict]) -> dict:
    """Per-subject mean and pooled (all frames together) balanced accuracy."""
    outcomes = [o for r in results for o in r["outcomes"]]
    truth = [o[0] for o in outcomes]
    summary = {
        "n_subjects": len(results),
        "n_frames": len(outcomes),
        "pooled_frames_balanced_accuracy_uncalibrated": balanced_accuracy(truth, [o[1] for o in outcomes]),
        "pooled_frames_balanced_accuracy_calibrated": balanced_accuracy(truth, [o[2] for o in outcomes]),
        "mean_subject_balanced_accuracy_uncalibrated": None,
        "mean_subject_balanced_accuracy_calibrated": None,
    }
    if results:
        summary["mean_subject_balanced_accuracy_uncalibrated"] = statistics.fmean(
            r["balanced_accuracy_uncalibrated"] for r in results
        )
        summary["mean_subject_balanced_accuracy_calibrated"] = statistics.fmean(
            r["balanced_accuracy_calibrated"] for r in results
        )
    return summary


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True, help="extracted Head Pose Image Database directory")
    parser.add_argument("--output", default=None, help="results JSON (default: evaluation-results.json in the corpus dir)")
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f"ERROR: corpus directory not found: {corpus}", file=sys.stderr)
        return 2
    output = Path(args.output) if args.output else corpus / "evaluation-results.json"

    grouped = group_frames_by_subject(sorted(corpus.rglob("*.jpg")))
    results = []
    for subject in sorted(grouped):
        result = evaluate_subject(grouped[subject])
        if result is not None:
            results.append(result)
    if not results:
        print("ERROR: no subject had a usable frontal frame and both classes", file=sys.stderr)
        return 2

    print(f"{'subject':<10} {'frames':>6} {'uncalibrated':>13} {'calibrated':>11}")
    for r in results:
        print(
            f"{r['subject']:<10} {r['n_frames']:>6} "
            f"{r['balanced_accuracy_uncalibrated']:>13.3f} {r['balanced_accuracy_calibrated']:>11.3f}"
        )
    summary = summarize(results)
    print(f"\nSubjects evaluated: {summary['n_subjects']} of {len(grouped)}; frames: {summary['n_frames']}")
    print(
        "Mean per-subject balanced accuracy: "
        f"uncalibrated {_fmt(summary['mean_subject_balanced_accuracy_uncalibrated'])}, "
        f"calibrated {_fmt(summary['mean_subject_balanced_accuracy_calibrated'])}"
    )
    print(
        "Pooled-frames balanced accuracy:    "
        f"uncalibrated {_fmt(summary['pooled_frames_balanced_accuracy_uncalibrated'])}, "
        f"calibrated {_fmt(summary['pooled_frames_balanced_accuracy_calibrated'])}"
    )

    output.write_text(
        json.dumps(
            {"summary": summary, "subjects": [{k: v for k, v in r.items() if k != "outcomes"} for r in results]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
