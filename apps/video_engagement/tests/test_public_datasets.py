"""Tests for the public-dataset -> labeled-corpus converters.

Pure-python: the parquet reader is lazy-imported, so this file runs under a
system python with no pyarrow, cv2 or mediapipe.
"""

import json

import pytest

from public_datasets import (
    HEAD_POSE_ATTENTIVE_MAX,
    HEAD_POSE_AWAY_MIN,
    convert_head_pose,
    eye_state_label,
    head_pose_label,
    parse_head_pose_filename,
    write_manifest,
)
from weight_search import LABEL_TARGETS, load_corpus_manifest


# --- filename parsing -------------------------------------------------------

def test_parse_head_pose_filename_reads_pan_and_tilt():
    parsed = parse_head_pose_filename("person15292+90+0.jpg")
    assert parsed == {"person": 15, "serie": 2, "number": 92, "tilt": 90, "pan": 0}


def test_parse_head_pose_filename_reads_negative_angles():
    parsed = parse_head_pose_filename("person02135-15+30.jpg")
    assert parsed["tilt"] == -15
    assert parsed["pan"] == 30


def test_parse_head_pose_filename_accepts_the_front_directory_spelling():
    # The Front/ directory spells the prefix "personne", not "person".
    parsed = parse_head_pose_filename("personne06246+0+0.jpg")
    assert parsed["person"] == 6
    assert (parsed["tilt"], parsed["pan"]) == (0, 0)


def test_parse_head_pose_filename_rejects_anything_else():
    with pytest.raises(ValueError, match="Not a Head Pose"):
        parse_head_pose_filename("readme.jpg")


# --- the labeling rubric ----------------------------------------------------

def test_head_pose_label_centre_is_attentive():
    assert head_pose_label(0, 0) == "attentive"


def test_head_pose_label_small_deviation_is_still_attentive():
    # Someone watching a screen does not hold their head perfectly still.
    assert head_pose_label(0, HEAD_POSE_ATTENTIVE_MAX) == "attentive"
    assert head_pose_label(-HEAD_POSE_ATTENTIVE_MAX, 0) == "attentive"


def test_head_pose_label_large_deviation_is_looking_away():
    assert head_pose_label(0, HEAD_POSE_AWAY_MIN) == "looking_away"
    assert head_pose_label(0, -90) == "looking_away"
    assert head_pose_label(90, 0) == "looking_away"


def test_head_pose_label_middle_band_is_excluded_not_guessed():
    # 30 degrees off could be a glance at a second monitor or a shift in the
    # chair; no rubric calls it consistently, so it must not enter the fit.
    assert head_pose_label(0, 30) is None
    assert head_pose_label(30, 0) is None


def test_head_pose_label_uses_the_worse_of_tilt_and_pan():
    assert head_pose_label(0, 90) == "looking_away"
    assert head_pose_label(15, 90) == "looking_away"


def test_head_pose_labels_are_known_to_the_fitter():
    for tilt, pan in ((0, 0), (0, 90)):
        assert head_pose_label(tilt, pan) in LABEL_TARGETS


# --- eye-state label mapping ------------------------------------------------

def test_eye_state_label_maps_onto_the_repo_vocabulary():
    assert eye_state_label("open_eyes") == "attentive"
    assert eye_state_label("closed_eyes") == "eyes_closed"
    assert LABEL_TARGETS[eye_state_label("open_eyes")] == 1.0
    assert LABEL_TARGETS[eye_state_label("closed_eyes")] == 0.0


def test_eye_state_label_rejects_an_unexpected_value():
    with pytest.raises(ValueError, match="Unexpected eye-state label"):
        eye_state_label("squinting")


# --- manifest writing and the head-pose converter ---------------------------

def test_write_manifest_emits_one_json_object_per_line(tmp_path):
    manifest = write_manifest(tmp_path, [{"frame": "a.jpg", "label": "attentive"}])
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"frame": "a.jpg", "label": "attentive"}]


def _fake_head_pose_tree(tmp_path, names):
    person_dir = tmp_path / "Person01"
    person_dir.mkdir(parents=True)
    for name in names:
        (person_dir / name).write_bytes(b"jpeg-ish")
    return tmp_path


def test_convert_head_pose_labels_and_excludes_by_the_rubric(tmp_path):
    root = _fake_head_pose_tree(
        tmp_path,
        [
            "person01146+0+0.jpg",       # attentive
            "person01147+0+15.jpg",      # attentive
            "person01148+0+30.jpg",      # ambiguous -> excluded
            "person01149+0+90.jpg",      # looking_away
            "person01150-60+0.jpg",      # looking_away (tilt)
        ],
    )
    (root / "Person01" / "notes.txt").write_text("ignored", encoding="utf-8")

    summary = convert_head_pose(root)
    assert summary["labeled"] == 4
    assert summary["excluded_ambiguous_or_unparsed"] == 1

    records = load_corpus_manifest(root)
    assert sorted(r["label"] for r in records) == [
        "attentive", "attentive", "looking_away", "looking_away",
    ]
    assert all(r["frame"].is_file() for r in records)


def test_convert_head_pose_output_loads_back_through_the_fitter(tmp_path):
    root = _fake_head_pose_tree(tmp_path, ["person01146+0+0.jpg", "person01149+0+90.jpg"])
    convert_head_pose(root)
    records = load_corpus_manifest(root)
    assert [r["target"] for r in records] == [1.0, 0.0]


def test_convert_head_pose_is_rerunnable_without_duplicating_rows(tmp_path):
    root = _fake_head_pose_tree(tmp_path, ["person01146+0+0.jpg", "person01149+0+90.jpg"])
    convert_head_pose(root)
    convert_head_pose(root)
    assert len(load_corpus_manifest(root)) == 2


def test_convert_closed_open_eyes_needs_shards(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .parquet shards"):
        from public_datasets import convert_closed_open_eyes

        convert_closed_open_eyes(tmp_path)
