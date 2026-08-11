import json
import wave
from fastapi.testclient import TestClient


def _write_test_fixtures(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    raw_dir = tmp_path / "transcripts" / "raw"
    ann_dir = tmp_path / "annotations"
    audio_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    with wave.open(str(audio_dir / "sess_A01.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\x00\x00" * 8000)

    transcript = {"words": [{"text": "Hello", "start": 0.0, "end": 0.4}, {"text": "there.", "start": 0.5, "end": 0.9}]}
    (raw_dir / "sess_A01_wer0.json").write_text(json.dumps(transcript))

    import eval_dataset.tools.annotation_app.main as main_module
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(audio_dir))
    monkeypatch.setattr(main_module, "TRANSCRIPTS_DIR", str(raw_dir))
    monkeypatch.setattr(main_module, "ANNOTATIONS_DIR", str(ann_dir))
    return main_module, ann_dir


def test_list_sessions_returns_sessions_with_transcripts(tmp_path, monkeypatch):
    main_module, _ = _write_test_fixtures(tmp_path, monkeypatch)
    client = TestClient(main_module.app)

    response = client.get("/sessions")

    assert response.status_code == 200
    assert response.json() == [{"session_id": "sess_A01"}]


def test_get_windows_returns_chunked_transcript(tmp_path, monkeypatch):
    main_module, _ = _write_test_fixtures(tmp_path, monkeypatch)
    client = TestClient(main_module.app)

    response = client.get("/sessions/sess_A01/windows")

    assert response.status_code == 200
    assert response.json() == [{"index": 0, "start": 0.0, "end": 5.0, "text": "Hello there."}]


def test_post_annotation_saves_file(tmp_path, monkeypatch):
    main_module, ann_dir = _write_test_fixtures(tmp_path, monkeypatch)
    client = TestClient(main_module.app)

    response = client.post(
        "/sessions/sess_A01/annotate?annotator=bob",
        json=[{"window_index": 0, "label": "correct"}],
    )

    assert response.status_code == 200
    assert response.json() == {"status": "saved"}
    saved = json.loads((ann_dir / "sess_A01_bob.json").read_text())
    assert saved == [{"window_index": 0, "label": "correct"}]
