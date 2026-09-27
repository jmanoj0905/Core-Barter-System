"""Guided webcam capture for the video-engagement weight experiment.

OPTIONAL validation path. The shipped weights and threshold were fitted on
public datasets (`public_datasets.py`, see
docs/video_engagement/design-choices.md); this tool exists to check that fit
against real frames from THIS rig — the one thing public data cannot do — and
to supply the `gaze_off` and `absent` states no public dataset provided.

Run locally (NOT in Docker — it needs the host camera):

    cd apps/video_engagement
    venv/bin/python collect_attention_frames.py

It walks you through a scripted sequence of labeled attention states, holds
each one for a few seconds while saving JPEG frames, and appends a
`labels.jsonl` manifest that `weight_search.py --corpus` reads back.

Captured frames are pictures of your face: the corpus directory is
gitignored and must never be committed.

## Why these five labels

The score under test fuses three sub-signals (`eyes_open`, `head_deviation`,
`gaze_centered`). A corpus that only contained "attentive" and "not
attentive" could not tell the three weights apart — any weighting would fit
equally well. So the inattentive side is split into one state per
sub-signal, each of which breaks exactly one signal and leaves the other two
looking attentive:

  attentive     all three good — the positive class.
  eyes_closed   breaks `eyes_open` only (head forward, gaze irrelevant).
  looking_away  breaks `head_deviation` only (eyes open, head turned).
  gaze_off      breaks `gaze_centered` only (head forward, eyes to the side).
  absent        no face at all — a detector-coverage control, not a
                scoring class. `video_attention_score` cannot run on a
                frame with no landmarks, so these rows are excluded from
                the weight fit and only used to check that the no-face path
                actually reports no face.

That structure is what makes a per-signal weight identifiable from the data
instead of from taste.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent / "corpus"
MANIFEST_NAME = "labels.jsonl"

# label -> what to do while it records. Order is deliberate: `absent` last so
# you can walk out of frame and end the run without rushing back.
CAPTURE_SCRIPT: list[tuple[str, str]] = [
    ("attentive", "Look straight at the screen, eyes open, head still."),
    ("eyes_closed", "Keep your head facing the screen and CLOSE your eyes."),
    ("looking_away", "Keep your eyes open and TURN YOUR HEAD ~45 degrees to one side."),
    ("gaze_off", "Head facing the screen, but move only your EYES far to one side."),
    ("attentive", "Back to attentive: straight at the screen, eyes open."),
    ("looking_away", "Turn your head ~45 degrees to the OTHER side."),
    ("absent", "Step fully OUT of frame — no face visible at all."),
]

LABELS = sorted({label for label, _ in CAPTURE_SCRIPT})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def frame_filename(label: str, index: int, epoch_ms: int) -> str:
    """Stable, collision-free name so repeated runs append instead of clobber."""
    return f"{label}_{epoch_ms}_{index:04d}.jpg"


def next_frame_index(label_dir: Path) -> int:
    """Highest existing index in a label dir + 1, so re-runs append."""
    if not label_dir.is_dir():
        return 0
    highest = -1
    for path in label_dir.glob("*.jpg"):
        stem_parts = path.stem.rsplit("_", 1)
        if len(stem_parts) == 2 and stem_parts[1].isdigit():
            highest = max(highest, int(stem_parts[1]))
    return highest + 1


def append_manifest_rows(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _countdown(prompt: str, label: str, seconds: int) -> None:
    print(f"\n=== {label.upper()} ===")
    print(f"    {prompt}")
    for remaining in range(seconds, 0, -1):
        print(f"    recording in {remaining}...", flush=True)
        time.sleep(1)


def capture(out_dir: Path, seconds_per_label: float, fps: float, lead_in: int) -> int:
    import cv2  # lazy: this module stays importable without OpenCV installed

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        print(
            "ERROR: could not open camera 0.\n"
            "  - On macOS, grant camera access to your terminal in\n"
            "    System Settings > Privacy & Security > Camera, then retry.\n"
            "  - Close any other app holding the webcam (Zoom, Meet, Photo Booth).\n"
            "  - This script must run on the host, not inside Docker.",
            file=sys.stderr,
        )
        return 2

    manifest_path = out_dir / MANIFEST_NAME
    total_saved = 0
    frame_interval = 1.0 / fps if fps > 0 else 0.0

    try:
        print(f"Corpus: {out_dir}")
        print(f"Holding each state for {seconds_per_label:g}s at ~{fps:g} fps.")
        print("Frames are personal data and stay gitignored. Ctrl-C aborts.\n")

        for step, (label, prompt) in enumerate(CAPTURE_SCRIPT, start=1):
            print(f"[{step}/{len(CAPTURE_SCRIPT)}]", end=" ")
            _countdown(prompt, label, lead_in)

            label_dir = out_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            index = next_frame_index(label_dir)
            epoch_ms = int(time.time() * 1000)

            rows: list[dict] = []
            deadline = time.monotonic() + seconds_per_label
            while time.monotonic() < deadline:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("    (dropped a frame — camera returned nothing)")
                    time.sleep(frame_interval)
                    continue
                encoded_ok, buffer = cv2.imencode(".jpg", frame)
                if not encoded_ok:
                    continue
                name = frame_filename(label, index, epoch_ms)
                (label_dir / name).write_bytes(buffer.tobytes())
                rows.append(
                    {
                        "frame": f"{label}/{name}",
                        "label": label,
                        "timestamp": _timestamp(),
                    }
                )
                index += 1
                time.sleep(frame_interval)

            append_manifest_rows(manifest_path, rows)
            total_saved += len(rows)
            print(f"    saved {len(rows)} frames -> {label_dir}")

    except KeyboardInterrupt:
        print("\nAborted by user; frames already written are kept.", file=sys.stderr)
    finally:
        camera.release()

    print(f"\nDone. {total_saved} frames this run. Manifest: {manifest_path}")
    print("Next: check the fitted weights against these frames with")
    print(f"  venv/bin/python weight_search.py --corpus {out_dir}")
    print("or add it as a third split alongside the public corpora.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="corpus directory (default: apps/video_engagement/corpus/)")
    parser.add_argument("--seconds-per-label", type=float, default=4.0, help="hold time per labeled state")
    parser.add_argument("--fps", type=float, default=5.0, help="frames saved per second")
    parser.add_argument("--lead-in", type=int, default=3, help="countdown seconds before each state")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seconds_per_label <= 0:
        print("ERROR: --seconds-per-label must be positive", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return capture(out_dir, args.seconds_per_label, args.fps, max(0, args.lead_in))
    except ImportError as exc:
        print(f"ERROR: OpenCV is required to capture frames ({exc}).", file=sys.stderr)
        print("Run it with the service venv: venv/bin/python collect_attention_frames.py", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
