import importlib.util
from pathlib import Path

from eval_dataset.tools.embed_windows import WindowScore, score_windows
from eval_dataset.tools.replay import ReplayWindow

_WINDOWING_PATH = (
    Path(__file__).resolve().parents[3] / "apps" / "semantic_analysis" / "windowing.py"
)
_spec = importlib.util.spec_from_file_location("_test_windowing", _WINDOWING_PATH)
_windowing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_windowing)

classify = _windowing.classify
topic_text = _windowing.topic_text


def make_window(index, text):
    return ReplayWindow(
        index=index,
        text=text,
        raw_text=text,
        teacher_seconds_start=0.0,
        teacher_seconds_end=25.0,
        contributions=(),
    )


def test_identical_text_scores_near_one():
    topic = "Node.js event loop"
    scope = "non-blocking I/O"
    windows = [make_window(1, topic_text(topic, scope))]

    scores = score_windows(windows, topic, scope)

    assert len(scores) == 1
    assert scores[0].similarity > 0.99


def test_classification_uses_production_thresholds():
    topic = "Node.js event loop"
    scope = "non-blocking I/O"
    windows = [
        make_window(1, topic_text(topic, scope)),
        make_window(2, "The weather in Antarctica is extremely cold and windy."),
        make_window(3, "Node uses an event loop to handle asynchronous callbacks."),
    ]

    scores = score_windows(windows, topic, scope)

    assert len(scores) == 3
    for score in scores:
        assert score.classification == classify(score.similarity)


def test_scope_is_included_in_the_topic_embedding():
    topic = "Node.js event loop"
    windows = [make_window(1, "Node uses an event loop to handle asynchronous callbacks.")]

    scores_a = score_windows(windows, topic, "non-blocking I/O and callbacks")
    scores_b = score_windows(windows, topic, "unrelated cooking recipes and desserts")

    assert scores_a[0].similarity != scores_b[0].similarity


def test_empty_window_list_returns_empty():
    assert score_windows([], "t", "s") == []
