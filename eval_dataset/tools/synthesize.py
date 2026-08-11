import json
import wave
import io

from eval_dataset.tools.script_parser import Script

VOICE_MAP = {"A": "Joanna", "B": "Matthew"}
PAUSE_SECONDS = 0.6
SAMPLE_RATE = 8000  # Polly "pcm" output format is 16-bit signed LE, 8kHz or 16kHz


def synthesize_turn(polly_client, text: str, voice_id: str) -> bytes:
    response = polly_client.synthesize_speech(
        Text=text, VoiceId=voice_id, OutputFormat="pcm", Engine="neural",
        SampleRate=str(SAMPLE_RATE),
    )
    return response["AudioStream"].read()


def synthesize_script(script: Script, polly_client, output_wav_path: str, timing_json_path: str) -> None:
    for turn in script.turns:
        if turn.speaker not in VOICE_MAP:
            raise ValueError(
                f"unknown speaker {turn.speaker!r}, expected one of {sorted(VOICE_MAP)}"
            )

    pcm_chunks = []
    timing = []
    cursor = 0.0
    silence_frame = b"\x00\x00" * int(SAMPLE_RATE * PAUSE_SECONDS)

    for turn in script.turns:
        voice_id = VOICE_MAP[turn.speaker]
        pcm = synthesize_turn(polly_client, turn.text, voice_id)
        duration = len(pcm) / 2 / SAMPLE_RATE  # 16-bit samples
        timing.append({"speaker": turn.speaker, "text": turn.text, "start": cursor, "end": cursor + duration})
        cursor += duration + PAUSE_SECONDS
        pcm_chunks.append(pcm)
        pcm_chunks.append(silence_frame)

    with wave.open(output_wav_path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(b"".join(pcm_chunks))

    with open(timing_json_path, "w") as f:
        json.dump(timing, f, indent=2)


if __name__ == "__main__":
    import sys
    import boto3
    from eval_dataset.tools.script_parser import parse_script

    script_path = sys.argv[1]
    script = parse_script(script_path)
    session_id = script_path.split("/")[-1].removesuffix(".txt")
    client = boto3.client("polly")
    synthesize_script(
        script, client,
        f"eval_dataset/audio/{session_id}.wav",
        f"eval_dataset/audio/{session_id}_timing.json",
    )
