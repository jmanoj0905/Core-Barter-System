from eval_dataset.tools.window_chunker import chunk_into_windows, Window


def test_chunk_into_windows_splits_by_word_start_time():
    transcript = {
        "words": [
            {"text": "Alright,", "start": 0.0, "end": 0.4},
            {"text": "let's", "start": 0.4, "end": 0.7},
            {"text": "start", "start": 4.9, "end": 5.2},
            {"text": "with", "start": 5.3, "end": 5.5},
            {"text": "variables.", "start": 9.8, "end": 10.3},
        ]
    }

    windows = chunk_into_windows(transcript, window_seconds=5.0)

    assert windows == [
        Window(index=0, start=0.0, end=5.0, text="Alright, let's start"),
        Window(index=1, start=5.0, end=10.0, text="with variables."),
    ]


def test_chunk_into_windows_empty_transcript_returns_empty_list():
    assert chunk_into_windows({"words": []}) == []
