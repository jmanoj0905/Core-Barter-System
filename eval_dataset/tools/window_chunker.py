from dataclasses import dataclass


@dataclass
class Window:
    index: int
    start: float
    end: float
    text: str


def chunk_into_windows(transcript: dict, window_seconds: float = 5.0) -> list[Window]:
    words = transcript["words"]
    if not words:
        return []

    max_end = max(w["end"] for w in words)
    num_windows = int(max_end // window_seconds) + 1

    windows = []
    for i in range(num_windows):
        w_start = i * window_seconds
        w_end = w_start + window_seconds
        words_in_window = [w["text"] for w in words if w_start <= w["start"] < w_end]
        if words_in_window:
            windows.append(Window(index=i, start=w_start, end=w_end, text=" ".join(words_in_window)))

    return windows
