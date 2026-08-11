# Eval Dataset Tools

All commands below must be run from the repo root (paths and output
locations are hardcoded relative to `eval_dataset/`).

Build order: write a script in `eval_dataset/scripts/person_X/sess_XNN.txt`
(format in `docs/superpowers/specs/2026-08-11-eval-dataset-design.md`), then
run:

    python -m eval_dataset.tools.synthesize eval_dataset/scripts/person_A/sess_A01.txt
    python -m eval_dataset.tools.transcribe eval_dataset/audio/sess_A01.wav
    python -m eval_dataset.tools.wer_inject eval_dataset/transcripts/raw/sess_A01_wer0.json

The last command writes `sess_A01_wer{10,20,30}.json` variants to
`eval_dataset/transcripts/synthetic/`.

Then annotate via the web app:

    uvicorn eval_dataset.tools.annotation_app.main:app --reload
    # open http://localhost:8000

Requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` env vars
(same as `apps/audio_pipeline`), plus Polly + Transcribe + S3 IAM permissions.
