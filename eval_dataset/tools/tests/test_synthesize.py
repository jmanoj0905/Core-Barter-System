import json
import wave
from unittest.mock import MagicMock
import io

from eval_dataset.tools.script_parser import Script, Turn
from eval_dataset.tools.synthesize import synthesize_script, VOICE_MAP


def _fake_wav_bytes(duration_seconds=0.5, framerate=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.set_nchannels_or_defaults = None
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(framerate)
        f.writeframes(b"\x00\x00" * int(framerate * duration_seconds))
    return buf.getvalue()


def test_synthesize_script_calls_polly_per_turn_with_correct_voice(tmp_path):
    script = Script(
        topic="Python", teacher="A", learner="B", category="clean",
        turns=[Turn(speaker="A", text="Hello there."), Turn(speaker="B", text="Hi!")],
    )
    mock_client = MagicMock()
    mock_client.synthesize_speech.return_value = {"AudioStream": io.BytesIO(_fake_wav_bytes())}

    wav_path = tmp_path / "out.wav"
    timing_path = tmp_path / "timing.json"
    synthesize_script(script, mock_client, str(wav_path), str(timing_path))

    calls = mock_client.synthesize_speech.call_args_list
    assert calls[0].kwargs["VoiceId"] == VOICE_MAP["A"]
    assert calls[0].kwargs["Text"] == "Hello there."
    assert calls[1].kwargs["VoiceId"] == VOICE_MAP["B"]
    assert calls[1].kwargs["Text"] == "Hi!"
    assert calls[0].kwargs["OutputFormat"] == "pcm"

    assert wav_path.exists()
    timing = json.loads(timing_path.read_text())
    assert len(timing) == 2
    assert timing[0]["speaker"] == "A"
    assert timing[1]["start"] > timing[0]["start"]
