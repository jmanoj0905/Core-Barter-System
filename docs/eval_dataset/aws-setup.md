# AWS Setup for Eval Dataset Tools

The `eval_dataset/tools/` package needs AWS Polly (text-to-speech), AWS
Transcribe (speech-to-text), and an S3 bucket as scratch storage between
them. This is the same credential convention already used by
`apps/audio_pipeline` — nothing new to invent, just extend what's already
there.

## What I need from you

1. **An AWS account/IAM user with programmatic access** — either your
   existing one (if `apps/audio_pipeline` is already configured) or a new
   IAM user scoped to this project.
2. **Access key + secret key** for that IAM user (see "IAM policy" below for
   exactly which permissions to attach).
3. **Confirmation of the AWS region** to use — default is `ap-south-1`
   (Mumbai), matching the rest of the project. Polly and Transcribe are
   available there; if you'd rather use a different region, let me know.
4. **An S3 bucket name** for scratch audio storage — default
   `core-barter-audio-tmp` (same bucket `apps/audio_pipeline` already uses).
   If that bucket doesn't exist yet, either create it or tell me and I'll
   walk through `aws s3 mb`.

Once you have the access key + secret key, paste them into a `.env` file at
the repo root (never commit this file — it's already covered by the
project's `.gitignore` pattern for `apps/`, double-check `eval_dataset/`
doesn't need its own):

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-south-1
AWS_S3_BUCKET=core-barter-audio-tmp
```

These are the exact same variable names `apps/audio_pipeline` reads (see
root `CLAUDE.md` → STT Configuration), and `eval_dataset/tools/transcribe.py`
reads `AWS_S3_BUCKET` with the same default. `eval_dataset/tools/synthesize.py`
and `transcribe.py` both instantiate `boto3.client(...)` with no explicit
region argument, so boto3 picks up `AWS_REGION` from the environment
automatically — just make sure it's exported in your shell or `.env` before
running any of the CLI tools.

## IAM policy to attach

Minimum permissions needed — Polly (synthesize), Transcribe (start/poll
jobs), and S3 (scratch bucket read/write/delete, limited to that one
bucket):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PollySynthesize",
      "Effect": "Allow",
      "Action": ["polly:SynthesizeSpeech"],
      "Resource": "*"
    },
    {
      "Sid": "TranscribeJobs",
      "Effect": "Allow",
      "Action": [
        "transcribe:StartTranscriptionJob",
        "transcribe:GetTranscriptionJob"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ScratchBucketAccess",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::core-barter-audio-tmp/*"
    },
    {
      "Sid": "ScratchBucketList",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::core-barter-audio-tmp"
    }
  ]
}
```

Replace `core-barter-audio-tmp` in the `Resource` ARNs if you pick a
different bucket name. `polly:SynthesizeSpeech` and the two `transcribe:*`
actions don't support resource-level restriction, hence `Resource: "*"` for
those — this is normal for these two services, not an overly broad grant.

Transcribe also needs read access to the S3 object it's transcribing — the
policy above already covers that via `ScratchBucketAccess`, since Transcribe
reads from the same bucket the wav was uploaded to.

## What each tool actually calls

- `eval_dataset/tools/synthesize.py` → `polly_client.synthesize_speech(...)`
  with `Engine="neural"`, `OutputFormat="pcm"`, voices `Joanna` (speaker A)
  and `Matthew` (speaker B). Confirm your AWS account has neural-engine
  access in the chosen region (it does by default in most regions,
  including `ap-south-1`, but flagging in case your account has any Polly
  service restrictions).
- `eval_dataset/tools/transcribe.py` → uploads the synthesized `.wav` to
  `s3://<bucket>/<job_name>.wav`, starts a Transcribe job, polls (capped —
  gives up with a `TimeoutError` after ~10 minutes rather than hanging
  forever), downloads the transcript JSON, reshapes it, and deletes the S3
  scratch object when done.

## Running it end-to-end

Once your teammates' scripts are in `eval_dataset/scripts/person_X/`, from
the repo root:

```bash
pip install -r eval_dataset/requirements.txt

python -m eval_dataset.tools.synthesize eval_dataset/scripts/person_A/sess_A01.txt
python -m eval_dataset.tools.transcribe eval_dataset/audio/sess_A01.wav
python -m eval_dataset.tools.wer_inject eval_dataset/transcripts/raw/sess_A01_wer0.json

uvicorn eval_dataset.tools.annotation_app.main:app --reload
# open http://localhost:8000 to annotate

python -m eval_dataset.tools.compute_kappa
```

## Known limitation (already noted, not a blocker)

If `transcribe_audio`'s `start_transcription_job` call itself fails (AWS
throttling, a job-name collision, etc. — rare, and the `__main__` CLI
already suffixes job names with a timestamp to avoid collisions on retry),
the already-uploaded S3 scratch object won't get cleaned up automatically.
Not a correctness issue for the dataset — just means an occasional manual
`aws s3 rm` in the scratch bucket if a job genuinely fails to start. Not
worth hardening further unless it becomes a recurring annoyance.

## Cost note

40 sessions × ~15-20 turns × Polly neural pricing, plus 40 sessions'
worth of Transcribe audio (5-8 min each ≈ 4-5 hours total audio) — this is
well within AWS free-tier-adjacent costs for Polly and low-double-digit
dollars for Transcribe at most. Nothing here should run up a meaningful
bill, but flagging since it's the first time this project has made live
calls to either service outside `apps/audio_pipeline`'s existing Transcribe
usage.
