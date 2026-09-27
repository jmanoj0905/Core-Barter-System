"""Tests for the capture tool's pure parts.

`collect_attention_frames` lazy-imports cv2, so this file runs under a plain
system python with no OpenCV installed. The camera loop itself is not unit
tested — it needs hardware — but the naming, append and manifest logic that
decides whether a re-run corrupts the corpus is.
"""

import json

import pytest

from collect_attention_frames import (
    CAPTURE_SCRIPT,
    LABELS,
    MANIFEST_NAME,
    append_manifest_rows,
    build_parser,
    frame_filename,
    main,
    next_frame_index,
)
from weight_search import LABEL_TARGETS, load_corpus_manifest


def test_capture_script_labels_match_the_weight_search_label_set():
    # A label the fitter does not know would crash the experiment after the
    # user has already sat through a capture session.
    assert LABELS == sorted(LABEL_TARGETS)


def test_capture_script_covers_each_sub_signal_failure_mode():
    scripted = {label for label, _ in CAPTURE_SCRIPT}
    assert {"attentive", "eyes_closed", "looking_away", "gaze_off", "absent"} <= scripted


def test_capture_script_prompts_are_non_empty():
    assert all(prompt.strip() for _, prompt in CAPTURE_SCRIPT)


def test_frame_filename_is_label_prefixed_and_zero_padded():
    assert frame_filename("attentive", 7, 1700000000000) == "attentive_1700000000000_0007.jpg"


def test_frame_filename_is_unique_per_index():
    names = {frame_filename("gaze_off", i, 1700000000000) for i in range(5)}
    assert len(names) == 5


def test_next_frame_index_is_zero_for_a_missing_dir(tmp_path):
    assert next_frame_index(tmp_path / "nope") == 0


def test_next_frame_index_appends_after_the_highest_existing(tmp_path):
    label_dir = tmp_path / "attentive"
    label_dir.mkdir()
    for index in (0, 1, 12):
        (label_dir / frame_filename("attentive", index, 1700000000000)).write_bytes(b"x")
    assert next_frame_index(label_dir) == 13


def test_next_frame_index_ignores_unparseable_names(tmp_path):
    label_dir = tmp_path / "attentive"
    label_dir.mkdir()
    (label_dir / "stray.jpg").write_bytes(b"x")
    (label_dir / frame_filename("attentive", 3, 1700000000000)).write_bytes(b"x")
    assert next_frame_index(label_dir) == 4


def test_append_manifest_rows_appends_rather_than_truncating(tmp_path):
    manifest = tmp_path / MANIFEST_NAME
    append_manifest_rows(manifest, [{"frame": "attentive/a.jpg", "label": "attentive"}])
    append_manifest_rows(manifest, [{"frame": "absent/b.jpg", "label": "absent"}])
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["label"] for line in lines] == ["attentive", "absent"]


def test_append_manifest_rows_creates_the_corpus_dir(tmp_path):
    manifest = tmp_path / "deeper" / MANIFEST_NAME
    append_manifest_rows(manifest, [{"frame": "attentive/a.jpg", "label": "attentive"}])
    assert manifest.is_file()


def test_manifest_written_here_is_readable_by_the_weight_search_loader(tmp_path):
    """Round-trip: the capture tool's output must load without hand-editing."""
    rows = []
    for label in ("attentive", "eyes_closed", "absent"):
        name = frame_filename(label, 0, 1700000000000)
        frame = tmp_path / label / name
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"jpeg-ish")
        rows.append({"frame": f"{label}/{name}", "label": label, "timestamp": "2026-09-26T10:00:00.000+00:00"})
    append_manifest_rows(tmp_path / MANIFEST_NAME, rows)

    records = load_corpus_manifest(tmp_path)
    assert [r["label"] for r in records] == ["attentive", "eyes_closed", "absent"]
    assert [r["target"] for r in records] == [1.0, 0.0, None]


def test_parser_defaults_point_at_the_project_corpus_dir():
    args = build_parser().parse_args([])
    assert args.out.endswith("apps/video_engagement/corpus")
    assert args.seconds_per_label > 0
    assert args.fps > 0


def test_parser_accepts_out_and_seconds_per_label():
    args = build_parser().parse_args(["--out", "/tmp/x", "--seconds-per-label", "2.5"])
    assert args.out == "/tmp/x"
    assert args.seconds_per_label == pytest.approx(2.5)


def test_main_exits_non_zero_on_a_non_positive_hold_time(tmp_path, capsys):
    assert main(["--out", str(tmp_path), "--seconds-per-label", "0"]) == 2
    assert "seconds-per-label" in capsys.readouterr().err


def test_main_exits_non_zero_when_opencv_is_unavailable(tmp_path, monkeypatch, capsys):
    """No cv2 (system python) or no camera must print a message and fail cleanly,
    never traceback."""
    import collect_attention_frames as module

    def boom(*_args, **_kwargs):
        raise ImportError("No module named 'cv2'")

    monkeypatch.setattr(module, "capture", boom)
    assert module.main(["--out", str(tmp_path)]) == 2
    assert "OpenCV is required" in capsys.readouterr().err
