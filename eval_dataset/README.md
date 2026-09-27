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

## STT sweep — `run_stt_sweep.py`

Wires `synthesize.py` -> `transcribe.py` -> `wer_inject.py` into one sweep
over the 25 `eval_dataset/scripts/calibration/*/sess_CAL*.txt` sessions:

    python -m eval_dataset.tools.run_stt_sweep --all

or for specific sessions:

    python -m eval_dataset.tools.run_stt_sweep sess_CAL01 sess_CAL02

`sweep()` first calls `check_polly_access()`, which makes one cheap
`describe_voices` call and aborts the entire run on `AccessDenied` rather
than letting 25 sessions fail one at a time mid-run. For each session it
then: synthesizes audio + turn timing via Polly, transcribes via AWS
Transcribe (WER-0 ground truth), and writes three WER-corrupted variants
(10/20/30%) at a fixed seed. A session whose `.wav` already exists is
skipped entirely (including the Polly gate's per-session work), so a rerun
after a partial failure doesn't re-spend Polly/Transcribe calls.

Expected artifact layout once the sweep has run over all 25 sessions:

- `eval_dataset/audio/sess_CAL{01..25}.wav` (25 files)
- `eval_dataset/audio/sess_CAL{01..25}_timing.json` (25 files)
- `eval_dataset/transcripts/raw/sess_CAL{01..25}_wer0.json` (25 files)
- `eval_dataset/transcripts/synthetic/sess_CAL{01..25}_wer{10,20,30}.json`
  (75 files)

WER-0 transcripts are committed as artifacts because AWS Transcribe is not
reproducible across runs (spec §6.4 / "WER injection" above) — there is no
way to regenerate the same ground-truth transcript later. The 10/20/30
variants are *not* committed: `wer_inject.py` is deterministic for a fixed
seed, so they are fully regenerable from the committed WER-0 file at the
seed recorded alongside the sweep.

**The real sweep has not been executed as part of this work.** Per
"Cloud services required" above, the Polly IAM permission needed by
`synthesize.py` was never added to the AWS credentials this repo uses —
`check_polly_access()` exists specifically to gate on that before spending
any Transcribe/S3 calls. No `.wav`, timing, or transcript files exist in
the repo for the calibration corpus as a result. `timed_durations()` (word
-> turn attribution by timestamp containment against `synthesize.py`'s
per-turn timing file, since `transcribe.py` emits no speaker labels) is
implemented and unit-tested against synthetic fixtures, ready for Task 13
to consume once the sweep has actually produced real transcripts.

## Running the tests

`python3 -m pytest eval_dataset/tools/tests/ -q` runs everything except the
`sentence_transformers`-dependent tests (`test_conformance_oracle.py`,
`test_embed_windows.py`), which skip under system `python3`. Run those under
`apps/semantic_analysis/venv/bin/python -m pytest eval_dataset/tools/tests/ -q`
instead — that venv lacks `boto3`, so the 5 boto3-dependent tests in
`test_run_stt_sweep.py` skip there instead. Both commands exit 0,
green-with-skips; each skip reason names the interpreter that runs it.
