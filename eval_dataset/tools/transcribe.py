import time
import urllib.request
import json


def _download_json(uri: str) -> dict:
    with urllib.request.urlopen(uri) as response:
        return json.loads(response.read())


def _reshape_transcript(aws_payload: dict) -> dict:
    words = []
    for item in aws_payload["results"]["items"]:
        if item["type"] != "pronunciation":
            continue
        words.append({
            "text": item["alternatives"][0]["content"],
            "start": float(item["start_time"]),
            "end": float(item["end_time"]),
        })
    return {"words": words}


def transcribe_audio(
    s3_client, transcribe_client, wav_path: str, bucket: str, job_name: str,
    max_polls: int = 120,
) -> dict:
    s3_key = f"{job_name}.wav"
    s3_client.upload_file(wav_path, bucket, s3_key)

    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": f"s3://{bucket}/{s3_key}"},
        MediaFormat="wav",
        LanguageCode="en-US",
    )

    try:
        status = None
        for _ in range(max_polls):
            response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            status = response["TranscriptionJob"]["TranscriptionJobStatus"]
            if status in ("COMPLETED", "FAILED"):
                break
            time.sleep(5)
        else:
            raise TimeoutError(
                f"Transcribe job {job_name} did not finish within {max_polls} polls"
            )

        if status == "FAILED":
            raise RuntimeError(f"Transcribe job {job_name} failed")

        uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        aws_payload = _download_json(uri)
        return _reshape_transcript(aws_payload)
    finally:
        s3_client.delete_object(Bucket=bucket, Key=s3_key)


if __name__ == "__main__":
    import sys
    import boto3
    import os

    wav_path = sys.argv[1]
    base_name = wav_path.split("/")[-1].removesuffix(".wav")
    # append a timestamp so re-running after a failure doesn't collide with
    # the previous (failed) Transcribe job name (ConflictException).
    job_name = f"{base_name}-{int(time.time())}"
    bucket = os.environ.get("AWS_S3_BUCKET", "core-barter-audio-tmp")
    s3 = boto3.client("s3")
    transcribe = boto3.client("transcribe")
    result = transcribe_audio(s3, transcribe, wav_path, bucket, job_name)
    with open(f"eval_dataset/transcripts/raw/{base_name}_wer0.json", "w") as f:
        json.dump(result, f, indent=2)
