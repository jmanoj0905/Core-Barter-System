# Eval Dataset Tools

Build order: write a script in `scripts/person_X/sess_XNN.txt` (format in
`docs/superpowers/specs/2026-08-11-eval-dataset-design.md`), then run:

    python -m eval_dataset.tools.synthesize scripts/person_A/sess_A01.txt
    python -m eval_dataset.tools.transcribe audio/sess_A01.wav
    python -m eval_dataset.tools.wer_inject transcripts/raw/sess_A01_wer0.json

Then annotate via the web app:

    uvicorn eval_dataset.tools.annotation_app.main:app --reload
    # open http://localhost:8000

Requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` env vars
(same as `apps/audio_pipeline`), plus Polly + Transcribe + S3 IAM permissions.
