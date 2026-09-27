import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval_dataset.tools.run_stt_sweep import check_polly_access, sweep, timed_durations
from eval_dataset.tools.script_parser import Script, Turn

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_module_imports_without_boto3_at_module_scope():
    # Task 13 imports timed_durations from this module while running under
    # apps/semantic_analysis/venv, which does not have boto3 installed. That
    # only works if `import boto3` happens inside functions, never at module
    # scope.
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import eval_dataset.tools.run_stt_sweep; "
        "assert 'boto3' not in sys.modules, 'boto3 imported at module scope'"
        % str(REPO_ROOT)
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_check_polly_access_raises_on_access_denied():
    from botocore.exceptions import ClientError

    client = MagicMock()
    client.describe_voices.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "not authorized"}},
        "DescribeVoices",
    )

    with pytest.raises(PermissionError, match="Polly"):
        check_polly_access(client)


def test_check_polly_access_passes_through_other_errors():
    from botocore.exceptions import ClientError

    client = MagicMock()
    client.describe_voices.side_effect = ClientError(
        {"Error": {"Code": "Throttling", "Message": "slow down"}},
        "DescribeVoices",
    )

    with pytest.raises(ClientError):
        check_polly_access(client)


def test_check_polly_access_succeeds_when_call_does_not_raise():
    client = MagicMock()
    client.describe_voices.return_value = {"Voices": []}
    check_polly_access(client)  # must not raise
    client.describe_voices.assert_called_once()


def _make_script():
    return Script(
        topic="Test topic",
        teacher="A",
        learner="B",
        category="clean",
        turns=[
            Turn(speaker="A", text="hello there friend"),
            Turn(speaker="B", text="hi teacher"),
            Turn(speaker="A", text="lets continue now"),
        ],
    )


def _make_timing():
    # turn 1: [0.0, 1.0), turn 2: [1.6, 2.4), turn 3: [3.0, 4.0)
    return [
        {"speaker": "A", "text": "hello there friend", "start": 0.0, "end": 1.0},
        {"speaker": "B", "text": "hi teacher", "start": 1.6, "end": 2.4},
        {"speaker": "A", "text": "lets continue now", "start": 3.0, "end": 4.0},
    ]


def test_timed_durations_assigns_each_word_to_its_turn():
    script = _make_script()
    timing = _make_timing()
    transcript = {
        "words": [
            {"text": "hello", "start": 0.0, "end": 0.3},
            {"text": "there", "start": 0.3, "end": 0.6},
            {"text": "friend", "start": 0.6, "end": 1.0},
            {"text": "hi", "start": 1.6, "end": 1.9},
            {"text": "teacher", "start": 1.9, "end": 2.4},
            {"text": "lets", "start": 3.0, "end": 3.3},
            {"text": "continue", "start": 3.3, "end": 3.7},
            {"text": "now", "start": 3.7, "end": 4.0},
        ]
    }

    durations = timed_durations(script, transcript, timing)

    assert len(durations) == 3
    assert durations[0] == pytest.approx(1.0)  # 0.3+0.3+0.4
    assert durations[1] == pytest.approx(0.8)  # 0.3+0.5
    assert durations[2] == pytest.approx(1.0)  # 0.3+0.4+0.3


def test_timed_durations_ignores_words_outside_every_turn():
    script = _make_script()
    timing = _make_timing()
    transcript = {
        "words": [
            {"text": "hello", "start": 0.0, "end": 0.3},
            # falls in the inter-turn silence between turn 1 (ends 1.0) and
            # turn 2 (starts 1.6) -- must be ignored, not misattributed.
            {"text": "um", "start": 1.2, "end": 1.4},
            {"text": "hi", "start": 1.6, "end": 1.9},
            # a filler word stamped past the last turn's end.
            {"text": "uh", "start": 4.5, "end": 4.7},
        ]
    }

    durations = timed_durations(script, transcript, timing)

    assert durations[0] == pytest.approx(0.3)
    assert durations[1] == pytest.approx(0.3)
    assert durations[2] == pytest.approx(0.0)
    assert sum(durations) == pytest.approx(0.6)  # the ignored words add nothing


def _write_calibration_script(base_dir: Path, session_id: str) -> Path:
    scripts_dir = base_dir / "eval_dataset" / "scripts" / "calibration" / "clean"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / f"{session_id}.txt"
    path.write_text(
        "TOPIC: Test topic\n"
        "TEACHER: A\n"
        "LEARNER: B\n"
        "CATEGORY: clean\n"
        "\n"
        "A: Hello there, let's begin.\n"
        "B: Sounds good.\n"
    )
    return path


def _fake_transcript():
    words = (
        "the quick brown fox jumps over the lazy dog while "
        "the cat sat quietly on the warm windowsill today"
    ).split()
    return {
        "words": [
            {"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
            for i, w in enumerate(words)
        ]
    }


def _mocked_boto3_client_factory(transcript):
    polly = MagicMock()
    polly.describe_voices.return_value = {"Voices": []}
    polly.synthesize_speech.return_value = {"AudioStream": MagicMock(read=lambda: b"\x00\x00" * 4000)}

    s3 = MagicMock()

    transcribe = MagicMock()
    transcribe.get_transcription_job.return_value = {
        "TranscriptionJob": {
            "TranscriptionJobStatus": "COMPLETED",
            "Transcript": {"TranscriptFileUri": "https://fake-uri/transcript.json"},
        }
    }

    clients = {"polly": polly, "s3": s3, "transcribe": transcribe}

    def client(service_name, *args, **kwargs):
        return clients[service_name]

    return client, clients


def test_wer_variants_are_deterministic_for_a_fixed_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_calibration_script(tmp_path, "sess_CALTEST")

    transcript = _fake_transcript()
    boto3_client, clients = _mocked_boto3_client_factory(transcript)

    with patch("boto3.client", side_effect=boto3_client), \
         patch("eval_dataset.tools.transcribe._download_json", return_value={
             "results": {
                 "items": [
                     {"type": "pronunciation", "alternatives": [{"content": w["text"]}],
                      "start_time": str(w["start"]), "end_time": str(w["end"])}
                     for w in transcript["words"]
                 ]
             }
         }), \
         patch("eval_dataset.tools.transcribe.time.sleep"):
        sweep(["sess_CALTEST"], seed=7)

    variant_dir = tmp_path / "eval_dataset" / "transcripts" / "synthetic"
    first_run = {
        pct: json.loads((variant_dir / f"sess_CALTEST_wer{pct}.json").read_text())
        for pct in (10, 20, 30)
    }

    # rerun from scratch (fresh output dir, same seed) must reproduce the
    # exact same variants -- sweep must thread the seed through rather than
    # reseeding per call.
    import shutil
    shutil.rmtree(tmp_path / "eval_dataset" / "audio")
    shutil.rmtree(tmp_path / "eval_dataset" / "transcripts")

    with patch("boto3.client", side_effect=boto3_client), \
         patch("eval_dataset.tools.transcribe._download_json", return_value={
             "results": {
                 "items": [
                     {"type": "pronunciation", "alternatives": [{"content": w["text"]}],
                      "start_time": str(w["start"]), "end_time": str(w["end"])}
                     for w in transcript["words"]
                 ]
             }
         }), \
         patch("eval_dataset.tools.transcribe.time.sleep"):
        sweep(["sess_CALTEST"], seed=7)

    second_run = {
        pct: json.loads((variant_dir / f"sess_CALTEST_wer{pct}.json").read_text())
        for pct in (10, 20, 30)
    }

    assert first_run == second_run


def test_sweep_skips_a_session_whose_wav_already_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_calibration_script(tmp_path, "sess_CALTEST")

    audio_dir = tmp_path / "eval_dataset" / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "sess_CALTEST.wav").write_bytes(b"already-here")

    transcript = _fake_transcript()
    boto3_client, clients = _mocked_boto3_client_factory(transcript)

    with patch("boto3.client", side_effect=boto3_client):
        sweep(["sess_CALTEST"], seed=0)

    clients["polly"].synthesize_speech.assert_not_called()
    clients["s3"].upload_file.assert_not_called()
    clients["transcribe"].start_transcription_job.assert_not_called()

    # only the Polly access gate ran (describe_voices), nothing else.
    clients["polly"].describe_voices.assert_called_once()

    # the wav content is untouched.
    assert (audio_dir / "sess_CALTEST.wav").read_bytes() == b"already-here"
