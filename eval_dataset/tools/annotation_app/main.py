import json
import os
import glob

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from eval_dataset.tools.window_chunker import chunk_into_windows

AUDIO_DIR = "eval_dataset/audio"
TRANSCRIPTS_DIR = "eval_dataset/transcripts/raw"
ANNOTATIONS_DIR = "eval_dataset/annotations"

app = FastAPI()


@app.get("/sessions")
def list_sessions():
    files = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*_wer0.json")))
    session_ids = [os.path.basename(f).removesuffix("_wer0.json") for f in files]
    return [{"session_id": sid} for sid in session_ids]


@app.get("/sessions/{session_id}/windows")
def get_windows(session_id: str):
    with open(os.path.join(TRANSCRIPTS_DIR, f"{session_id}_wer0.json")) as f:
        transcript = json.load(f)
    windows = chunk_into_windows(transcript)
    return [{"index": w.index, "start": w.start, "end": w.end, "text": w.text} for w in windows]


@app.post("/sessions/{session_id}/annotate")
def save_annotation(session_id: str, annotator: str, labels: list[dict]):
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    out_path = os.path.join(ANNOTATIONS_DIR, f"{session_id}_{annotator}.json")
    with open(out_path, "w") as f:
        json.dump(labels, f, indent=2)
    return {"status": "saved"}


if os.path.isdir(AUDIO_DIR):
    app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
