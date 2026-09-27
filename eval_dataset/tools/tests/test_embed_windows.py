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


def make_window(index, text, raw_text=None):
    return ReplayWindow(
        index=index,
        text=text,
        raw_text=text if raw_text is None else raw_text,
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


def test_embeds_cleaned_text_not_raw_text():
    # raw_text carries filler words and off-topic content that .text would
    # have had stripped by clean_text() during replay; the two must diverge
    # measurably so embedding the wrong field is actually detectable.
    topic = "Node.js event loop"
    scope = "non-blocking I/O and callbacks"
    text = "Node uses an event loop to handle asynchronous callbacks and non-blocking I/O efficiently."
    raw_text = (
        "Well, um, so, you know, the weather has been really nice and sunny "
        "lately, and honestly I went for a long walk by the beach, basically, "
        "literally had a great time, actually, okay."
    )

    mixed_window = make_window(1, text, raw_text=raw_text)
    text_only_window = make_window(1, text)  # raw_text defaults to text
    raw_only_window = make_window(1, raw_text)  # .text is the raw, off-topic string

    mixed_score = score_windows([mixed_window], topic, scope)[0]
    text_only_score = score_windows([text_only_window], topic, scope)[0]
    raw_only_score = score_windows([raw_only_window], topic, scope)[0]

    # Sanity: the two source texts really do diverge enough to be distinguishable.
    assert abs(text_only_score.similarity - raw_only_score.similarity) > 0.05

    # score_windows must embed `.text`, never `.raw_text`.
    assert mixed_score.similarity == text_only_score.similarity
    assert mixed_score.similarity != raw_only_score.similarity
