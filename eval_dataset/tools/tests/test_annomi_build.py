"""Tests for vendoring the AnnoMI source (Task 9).

Covers: the raw data is gitignored, the manifest pins the upstream
revision and per-file sha256, a corrupted local file is rejected rather
than silently accepted, and the loader groups utterance-level CSV rows
into conversations with a topic and both speaker roles.

Tests that require the real fetched CSVs skip cleanly (loud, explicit
pytest.skip — never a silent pass) when raw/ is absent, so this file
stays runnable on a machine that has not fetched AnnoMI.
"""

import hashlib
from pathlib import Path

import pytest

from eval_dataset.curated.annomi.build_dataset import (
    HERE,
    ManifestVerificationError,
    load_conversations,
    read_manifest,
    verify_file,
)

ANNOMI_DIR = Path(__file__).resolve().parents[3] / "eval_dataset" / "curated" / "annomi"


def _raw_present() -> bool:
    manifest = read_manifest()
    return all((ANNOMI_DIR / f["local_path"]).exists() for f in manifest["files"])


def test_gitignore_excludes_raw_data():
    gitignore = (ANNOMI_DIR / ".gitignore").read_text()
    lines = {line.strip() for line in gitignore.splitlines() if line.strip()}
    assert "raw/" in lines
    assert "*.csv" in lines
    assert "__pycache__/" in lines


def test_manifest_pins_revision_and_sha256_per_file():
    manifest = read_manifest()

    assert manifest["repository"] == "https://github.com/uccollab/AnnoMI"
    assert manifest["revision"] == "42936645ec3857a9c84ab296a36a3c34b779ef49"

    assert "license" in manifest
    assert manifest["license"].startswith("NONE")

    assert len(manifest["files"]) == 3
    expected_names = {"raw/AnnoMI-full.csv", "raw/AnnoMI-simple.csv", "raw/README.md"}
    assert {f["local_path"] for f in manifest["files"]} == expected_names

    for entry in manifest["files"]:
        assert manifest["revision"] in entry["url"]
        assert isinstance(entry["sha256"], str)
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)  # must be valid hex
        assert isinstance(entry["bytes"], int)
        assert entry["bytes"] > 0


def test_build_rejects_a_file_whose_sha256_mismatches(tmp_path):
    good_content = b"therapist,client\nhello,hi\n"
    good_digest = hashlib.sha256(good_content).hexdigest()

    corrupted_path = tmp_path / "corrupted.csv"
    corrupted_path.write_bytes(b"this is not the same content at all")

    entry = {
        "local_path": "corrupted.csv",
        "sha256": good_digest,
        "bytes": len(good_content),
    }

    # verify_file resolves local_path relative to HERE (the annomi package
    # directory), so point HERE-relative resolution at tmp_path by writing
    # the file where verify_file will actually look: HERE / local_path.
    # Simpler and equally valid: call the verification logic directly via
    # a manifest-shaped entry against a file we control the path of.
    target = HERE / entry["local_path"]
    try:
        target.write_bytes(corrupted_path.read_bytes())
        with pytest.raises(ManifestVerificationError):
            verify_file(entry)
    finally:
        if target.exists():
            target.unlink()


def test_load_conversations_yields_topic_and_two_roles():
    if not _raw_present():
        pytest.skip(
            "eval_dataset/curated/annomi/raw/ is absent — run "
            "`python3 eval_dataset/curated/annomi/build_dataset.py` to fetch "
            "AnnoMI before running this test. Skipping loudly and explicitly, "
            "not silently passing."
        )

    conversations = load_conversations()

    assert len(conversations) > 0

    roles_seen = set()
    for conversation in conversations:
        assert conversation.id
        assert conversation.topic
        assert isinstance(conversation.topic, str)
        assert conversation.topic.strip() != ""
        assert len(conversation.utterances) > 0
        for utterance in conversation.utterances:
            assert utterance.speaker_role in {"therapist", "client"}
            roles_seen.add(utterance.speaker_role)

    assert roles_seen == {"therapist", "client"}
