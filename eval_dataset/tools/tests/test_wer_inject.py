import json
import os
import subprocess
import sys

from eval_dataset.tools.wer_inject import compute_wer, inject_wer


def test_compute_wer_identical_sequences_is_zero():
    assert compute_wer(["the", "cat", "sat"], ["the", "cat", "sat"]) == 0.0


def test_compute_wer_one_substitution_out_of_three():
    assert compute_wer(["the", "cat", "sat"], ["the", "dog", "sat"]) == 1 / 3


def test_inject_wer_hits_target_within_tolerance():
    transcript = {
        "words": [
            {"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
            for i, w in enumerate(
                ("the quick brown fox jumps over the lazy dog while "
                 "the cat sat quietly on the warm windowsill today").split()
            )
        ]
    }
    original_words = [w["text"] for w in transcript["words"]]

    corrupted = inject_wer(transcript, target_wer=0.20, seed=42)
    corrupted_words = [w["text"] for w in corrupted["words"]]

    wer = compute_wer(original_words, corrupted_words)
    assert abs(wer - 0.20) <= 0.02


def test_inject_wer_hits_10_percent_target_within_tolerance():
    transcript = {
        "words": [
            {"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
            for i, w in enumerate(
                ("the quick brown fox jumps over the lazy dog while "
                 "the cat sat quietly on the warm windowsill today").split()
            )
        ]
    }
    original_words = [w["text"] for w in transcript["words"]]

    corrupted = inject_wer(transcript, target_wer=0.10, seed=42)
    corrupted_words = [w["text"] for w in corrupted["words"]]

    wer = compute_wer(original_words, corrupted_words)
    assert abs(wer - 0.10) <= 0.02


def test_inject_wer_hits_30_percent_target_within_tolerance():
    transcript = {
        "words": [
            {"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
            for i, w in enumerate(
                ("the quick brown fox jumps over the lazy dog while "
                 "the cat sat quietly on the warm windowsill today").split()
            )
        ]
    }
    original_words = [w["text"] for w in transcript["words"]]

    corrupted = inject_wer(transcript, target_wer=0.30, seed=42)
    corrupted_words = [w["text"] for w in corrupted["words"]]

    wer = compute_wer(original_words, corrupted_words)
    assert abs(wer - 0.30) <= 0.02


def test_inject_wer_zero_target_returns_unchanged_words():
    transcript = {"words": [{"text": "hi", "start": 0.0, "end": 0.3}]}
    corrupted = inject_wer(transcript, target_wer=0.0, seed=1)
    assert [w["text"] for w in corrupted["words"]] == ["hi"]


def test_cli_generates_all_three_wer_variants(tmp_path):
    repo_root = os.getcwd()
    synthetic_dir = tmp_path / "eval_dataset" / "transcripts" / "synthetic"
    synthetic_dir.mkdir(parents=True)

    transcript = {
        "words": [
            {"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
            for i, w in enumerate(
                ("the quick brown fox jumps over the lazy dog while "
                 "the cat sat quietly on the warm windowsill today").split()
            )
        ]
    }
    wer0_path = tmp_path / "sess_A01_wer0.json"
    wer0_path.write_text(json.dumps(transcript))

    env = dict(os.environ, PYTHONPATH=repo_root)
    result = subprocess.run(
        [sys.executable, "-m", "eval_dataset.tools.wer_inject", str(wer0_path)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    for pct in (10, 20, 30):
        out_path = synthetic_dir / f"sess_A01_wer{pct}.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "words" in data
        assert len(data["words"]) > 0
