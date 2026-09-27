import subprocess, sys
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent.parent

def test_windowing_imports_without_heavy_dependencies():
    # The harness imports this module outside the service venv.
    code = (
        "import sys; sys.path.insert(0, %r); import windowing; "
        "assert 'sentence_transformers' not in sys.modules; "
        "assert 'torch' not in sys.modules; assert 'fastapi' not in sys.modules"
        % str(APP_DIR)
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

def test_classify_boundaries():
    import windowing as w
    assert w.classify(0.36) == "correct"
    assert w.classify(0.359) == "weakly_correct"
    assert w.classify(0.14) == "weakly_correct"
    assert w.classify(0.139) == "incorrect"

def test_topic_text_matches_production_format():
    import windowing as w
    assert w.topic_text("Node.js basics", "event loop") == "Node.js basics. event loop"
    assert w.topic_text("Node.js basics", "") == "Node.js basics. "

def test_window_is_ready_at_threshold():
    import windowing as w
    assert w.window_is_ready(25.0) is True
    assert w.window_is_ready(24.9) is False

def test_has_enough_content_at_min_tokens():
    import windowing as w
    assert w.has_enough_content("one two three") is True
    assert w.has_enough_content("one two") is False
