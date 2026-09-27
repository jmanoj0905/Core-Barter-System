"""Vendor the AnnoMI motivational-interviewing corpus by pinned manifest.

Unlike eval_dataset/curated/qatd/build_dataset.py (which only verifies
sha256 of files already on disk), this module also fetches each manifest
entry to raw/ on demand before verifying it. Fetching is required because
raw/ is gitignored: no third-party bytes from AnnoMI are ever committed to
this repository (see SOURCE_README.md for why).

No LLM generation, training, or scoring is performed here. This is source
vendoring and loading only.

Run with system python3 from the repository root:
    python3 eval_dataset/curated/annomi/build_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "source_manifest.json"


class ManifestVerificationError(RuntimeError):
    """Raised when a vendored file's sha256 does not match the manifest."""


@dataclass(frozen=True)
class Utterance:
    speaker_role: str  # "therapist" or "client"
    text: str


@dataclass(frozen=True)
class Conversation:
    id: str
    topic: str
    utterances: list = field(default_factory=list)


def read_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_file(entry: dict) -> None:
    """Download one manifest entry to its local_path if not already present."""
    dest = HERE / entry["local_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urllib.request.urlopen(entry["url"]) as response:
        data = response.read()
    dest.write_bytes(data)


def verify_file(entry: dict) -> bytes:
    """Verify a vendored file's sha256 and byte count against the manifest.

    Returns the file's bytes on success. Raises ManifestVerificationError on
    any mismatch, rather than silently accepting a corrupted or altered
    download.
    """
    dest = HERE / entry["local_path"]
    data = dest.read_bytes()
    digest = _sha256(data)
    if digest != entry["sha256"]:
        raise ManifestVerificationError(
            f"{entry['local_path']}: sha256 mismatch "
            f"(expected {entry['sha256']}, got {digest})"
        )
    if len(data) != entry["bytes"]:
        raise ManifestVerificationError(
            f"{entry['local_path']}: byte count mismatch "
            f"(expected {entry['bytes']}, got {len(data)})"
        )
    return data


def fetch_and_verify_all(manifest: dict | None = None) -> None:
    """Fetch (if absent) then verify every file the manifest pins."""
    manifest = manifest or read_manifest()
    for entry in manifest["files"]:
        fetch_file(entry)
        verify_file(entry)


def load_conversations() -> list:
    """Load AnnoMI-simple.csv, grouped into Conversation objects.

    Requires raw/AnnoMI-simple.csv to already be present and verified
    (call fetch_and_verify_all() first, or run this module as a script).

    Utterances are grouped by transcript_id and ordered by utterance_id
    within each conversation. The topic is read verbatim from the CSV's
    `topic` column (never invented) since AnnoMI's topic labels are
    third-party.
    """
    manifest = read_manifest()
    simple_entry = next(
        f for f in manifest["files"] if f["local_path"] == "raw/AnnoMI-simple.csv"
    )
    verify_file(simple_entry)

    csv_path = HERE / simple_entry["local_path"]
    rows_by_transcript: dict[str, list[dict]] = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows_by_transcript.setdefault(row["transcript_id"], []).append(row)

    conversations = []
    for transcript_id, rows in rows_by_transcript.items():
        rows = sorted(rows, key=lambda r: int(r["utterance_id"]))
        topic = rows[0]["topic"]
        utterances = [
            Utterance(speaker_role=row["interlocutor"], text=row["utterance_text"])
            for row in rows
        ]
        conversations.append(
            Conversation(id=transcript_id, topic=topic, utterances=utterances)
        )
    return conversations


def main() -> None:
    manifest = read_manifest()
    fetch_and_verify_all(manifest)
    conversations = load_conversations()
    print(f"Vendored and verified {len(manifest['files'])} files.")
    print(f"Loaded {len(conversations)} conversations.")


if __name__ == "__main__":
    main()
