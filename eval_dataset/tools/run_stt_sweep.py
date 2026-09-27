"""Wires synthesize -> transcribe -> wer_inject into a per-session sweep.

Ruling R3 (task-12-brief): `boto3` (and `botocore`) are installed in system
python3 but NOT in `apps/semantic_analysis/venv`, which is where
sentence-transformers lives. Task 13 imports `timed_durations` from this
module while running under that venv, so `import boto3` (and any botocore
import) must happen INSIDE the functions that call AWS, never at module
scope — mirrors the import-light discipline in
`apps/semantic_analysis/windowing.py`.
"""

import json
import os

from eval_dataset.tools.script_parser import Script


def check_polly_access(polly_client) -> None:
    """Gate, not a nicety: make one minimal, cheap Polly call before spending
    any synthesis calls on the sweep.

    `docs/superpowers/specs/2026-08-11-eval-dataset-design.md` records the
    Polly IAM permission as never having been added, so this runs first and
    aborts the whole sweep on failure rather than letting 25 sessions fail
    one at a time mid-run.
    """
    from botocore.exceptions import ClientError

    try:
        polly_client.describe_voices()
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "AccessDenied":
            raise PermissionError(
                "Polly access denied: the AWS credentials in use are missing "
                "the polly:DescribeVoices (and likely polly:SynthesizeSpeech) "
                "IAM permission required by eval_dataset/tools/synthesize.py. "
                "Add it before running the sweep — see "
                "docs/superpowers/specs/2026-08-11-eval-dataset-design.md "
                "('Cloud services required')."
            ) from e
        raise


def timed_durations(script: Script, transcript: dict, timing: list[dict]) -> list[float]:
    """Per-turn teacher-speech durations measured from a real transcript.

    `transcribe.py` emits `{"words": [{text, start, end}]}` with no speaker
    labels. Diarization isn't required because we synthesize the audio:
    `synthesize.py` already writes a `{speaker, text, start, end}` timing
    record per turn. Each transcribed word is attributed to the turn whose
    `[start, end)` interval contains the word's start timestamp, and turn
    durations are the sum of its attributed words' `(end - start)`.

    A word landing inside no turn's interval — inter-turn silence, or a
    filler word `wer_inject.py` stamps past a turn's end — is ignored, not
    misattributed to a neighboring turn.

    The result is positionally aligned with `script.turns`, exactly like
    `synthetic_durations` in `eval_dataset/tools/replay.py` — `replay()`
    consumes either interchangeably.
    """
    durations = [0.0] * len(script.turns)

    for word in transcript["words"]:
        word_start = word["start"]
        for idx, turn_timing in enumerate(timing):
            if turn_timing["start"] <= word_start < turn_timing["end"]:
                durations[idx] += word["end"] - word["start"]
                break

    return durations


def _script_path_for_session(session_id: str) -> str:
    import glob

    matches = sorted(glob.glob(f"eval_dataset/scripts/calibration/*/{session_id}.txt"))
    if not matches:
        raise FileNotFoundError(
            f"no script found for session_id={session_id!r} under "
            "eval_dataset/scripts/calibration/"
        )
    if len(matches) > 1:
        raise ValueError(f"multiple scripts found for session_id={session_id!r}: {matches}")
    return matches[0]


def sweep(session_ids: list[str], seed: int = 0) -> None:
    """Run the full synthesize -> transcribe -> wer_inject pipeline for each
    session id, skipping any session whose `.wav` already exists so reruns
    (after a partial failure) are cheap and don't re-spend Polly/Transcribe
    calls.
    """
    import time

    import boto3

    from eval_dataset.tools.script_parser import parse_script
    from eval_dataset.tools.synthesize import synthesize_script
    from eval_dataset.tools.transcribe import transcribe_audio
    from eval_dataset.tools.wer_inject import inject_wer

    polly_client = boto3.client("polly")
    check_polly_access(polly_client)

    s3_client = boto3.client("s3")
    transcribe_client = boto3.client("transcribe")
    bucket = os.environ.get("AWS_S3_BUCKET", "core-barter-audio-tmp")

    os.makedirs("eval_dataset/audio", exist_ok=True)
    os.makedirs("eval_dataset/transcripts/raw", exist_ok=True)
    os.makedirs("eval_dataset/transcripts/synthetic", exist_ok=True)

    for session_id in session_ids:
        wav_path = f"eval_dataset/audio/{session_id}.wav"
        if os.path.exists(wav_path):
            print(f"skip {session_id}: {wav_path} already exists")
            continue

        script_path = _script_path_for_session(session_id)
        script = parse_script(script_path)

        timing_path = f"eval_dataset/audio/{session_id}_timing.json"
        synthesize_script(script, polly_client, wav_path, timing_path)

        job_name = f"{session_id}-{int(time.time())}"
        transcript = transcribe_audio(s3_client, transcribe_client, wav_path, bucket, job_name)

        wer0_path = f"eval_dataset/transcripts/raw/{session_id}_wer0.json"
        with open(wer0_path, "w") as f:
            json.dump(transcript, f, indent=2)

        for target_wer, pct in ((0.10, 10), (0.20, 20), (0.30, 30)):
            corrupted = inject_wer(transcript, target_wer=target_wer, seed=seed)
            out_path = f"eval_dataset/transcripts/synthetic/{session_id}_wer{pct}.json"
            with open(out_path, "w") as f:
                json.dump(corrupted, f, indent=2)


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="sweep all 25 calibration sessions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("session_ids", nargs="*")
    args = parser.parse_args()

    if args.all:
        ids = sorted(
            os.path.basename(p).removesuffix(".txt")
            for p in glob.glob("eval_dataset/scripts/calibration/*/sess_CAL*.txt")
        )
    else:
        ids = args.session_ids

    sweep(ids, seed=args.seed)
