"""Turn openly-licensed public face datasets into the labeled-frame corpus
format that `weight_search.py --corpus` reads.

Both converters emit exactly what `collect_attention_frames.py` emits — a
directory of JPEGs plus a `labels.jsonl` manifest — so the fitting code has a
single input format regardless of where the frames came from.

Datasets, licences and what each can actually speak to are documented in
`docs/video_engagement/design-choices.md`. Summary:

* **head_pose** — Head Pose Image Database (Gourier, Hall, Crowley; Pointing
  2004). Usable "for any purpose, provided that the [Pointing'04] article is
  cited". Real photographs, 15 people, discrete pan/tilt per image encoded in
  the filename. Speaks to `head_deviation`. Does NOT speak to `eyes_open`
  (every subject has their eyes open) and does NOT speak to `gaze_centered`
  (subjects rotate head and eyes together toward the target, so the iris stays
  centred in the socket even at pan 90 degrees).

* **closed_open_eyes** — MichalMlodawski/closed-open-eyes on Hugging Face,
  ODC-BY. Full faces labelled `open_eyes` / `closed_eyes`. Speaks to
  `eyes_open`. Does NOT speak to `head_deviation` or `gaze_centered`: head
  pose is uncontrolled and unlabelled in this set.

Run (from apps/video_engagement, with the service venv):

    venv/bin/python public_datasets.py head-pose datasets/head_pose_image_database
    venv/bin/python public_datasets.py closed-open-eyes datasets/closed_open_eyes

Downloaded imagery is gitignored and is never committed.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# person[Id][Serie][Number][Tilt][Pan].jpg — the `Front/` directory spells the
# prefix "personne", the per-subject directories spell it "person".
_HEAD_POSE_NAME = re.compile(
    r"^person(?:ne)?(?P<id>\d{2})(?P<serie>\d)(?P<number>\d+)"
    r"(?P<tilt>[+-]\d+)(?P<pan>[+-]\d+)\.jpg$",
    re.IGNORECASE,
)

# Head-pose labeling rubric, in degrees off centre.
#
# A person genuinely watching a screen still moves their head a little, so a
# small deviation is attentive rather than "looking away". A large deviation is
# unambiguously away. The band in between is real ambiguity — a 30-degree turn
# could be a glance at a second monitor or a shift in the chair — and no rubric
# can call it consistently, so those frames are EXCLUDED from the fit rather
# than pushed into whichever class is convenient.
HEAD_POSE_ATTENTIVE_MAX = 15
HEAD_POSE_AWAY_MIN = 45

EYE_STATE_LABELS = {"open_eyes": "attentive", "closed_eyes": "eyes_closed"}


def parse_head_pose_filename(name: str) -> dict:
    """Extract subject/serie/tilt/pan from a Head Pose Image Database filename."""
    match = _HEAD_POSE_NAME.match(name)
    if not match:
        raise ValueError(f"Not a Head Pose Image Database filename: {name!r}")
    return {
        "person": int(match["id"]),
        "serie": int(match["serie"]),
        "number": int(match["number"]),
        "tilt": int(match["tilt"]),
        "pan": int(match["pan"]),
    }


def head_pose_label(tilt: int, pan: int) -> str | None:
    """Apply the rubric above. None means "ambiguous — exclude from the fit"."""
    deviation = max(abs(tilt), abs(pan))
    if deviation <= HEAD_POSE_ATTENTIVE_MAX:
        return "attentive"
    if deviation >= HEAD_POSE_AWAY_MIN:
        return "looking_away"
    return None


def eye_state_label(dataset_label: str) -> str:
    """Map a closed-open-eyes `Label` value onto this repo's label vocabulary."""
    try:
        return EYE_STATE_LABELS[dataset_label]
    except KeyError:
        raise ValueError(f"Unexpected eye-state label {dataset_label!r}") from None


def write_manifest(out_dir: Path, rows: list[dict]) -> Path:
    """Write labels.jsonl for a corpus (overwrites: converters are deterministic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "labels.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest


def convert_head_pose(dataset_dir: str | Path) -> dict:
    """Label every Head Pose Image Database JPEG in place via its filename.

    The images stay where they are; only labels.jsonl is written, with frame
    paths relative to `dataset_dir`.
    """
    root = Path(dataset_dir)
    rows, skipped = [], 0
    for path in sorted(root.rglob("*.jpg")):
        try:
            pose = parse_head_pose_filename(path.name)
        except ValueError:
            skipped += 1
            continue
        label = head_pose_label(pose["tilt"], pose["pan"])
        if label is None:
            skipped += 1
            continue
        rows.append(
            {
                "frame": str(path.relative_to(root)),
                "label": label,
                "timestamp": None,
                "source": "head_pose_image_database",
                "pan": pose["pan"],
                "tilt": pose["tilt"],
                "person": pose["person"],
                # Grouping key for the held-out evaluation: a subject must not
                # appear in both the fit and the holdout, or the holdout only
                # measures memorisation of that face.
                "group": f"person{pose['person']:02d}",
            }
        )
    manifest = write_manifest(root, rows)
    return {"manifest": str(manifest), "labeled": len(rows), "excluded_ambiguous_or_unparsed": skipped}


def convert_closed_open_eyes(dataset_dir: str | Path, limit_per_class: int | None = None) -> dict:
    """Extract JPEGs from the downloaded parquet shards into a labeled corpus."""
    root = Path(dataset_dir)
    shards = sorted(root.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No .parquet shards in {root}")

    import pyarrow.parquet as pq  # lazy: only this converter needs it

    rows, per_class = [], {}
    for shard in shards:
        table = pq.read_table(shard, columns=["Image_id", "Label", "Image_data"])
        for record in table.to_pylist():
            label = eye_state_label(record["Label"])
            if limit_per_class is not None and per_class.get(label, 0) >= limit_per_class:
                continue
            image = record["Image_data"]
            name = f"{shard.stem}_{record['Image_id']}.jpg"
            frame_dir = root / "frames" / label
            frame_dir.mkdir(parents=True, exist_ok=True)
            (frame_dir / name).write_bytes(image["file"])
            per_class[label] = per_class.get(label, 0) + 1
            rows.append(
                {
                    "frame": f"frames/{label}/{name}",
                    "label": label,
                    "timestamp": None,
                    "source": "closed_open_eyes",
                    "shard": shard.name,
                    "dataset_label": record["Label"],
                    # No subject identity in this set; the shard is the closest
                    # available grouping (images are generated in shard batches).
                    "group": shard.stem,
                }
            )
    manifest = write_manifest(root, rows)
    return {"manifest": str(manifest), "labeled": len(rows), "per_class": per_class}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="dataset", required=True)

    head = sub.add_parser("head-pose", help="label an extracted Head Pose Image Database tree")
    head.add_argument("dataset_dir")

    eyes = sub.add_parser("closed-open-eyes", help="extract frames from downloaded parquet shards")
    eyes.add_argument("dataset_dir")
    eyes.add_argument("--limit-per-class", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.dataset == "head-pose":
            summary = convert_head_pose(args.dataset_dir)
        else:
            summary = convert_closed_open_eyes(args.dataset_dir, args.limit_per_class)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
