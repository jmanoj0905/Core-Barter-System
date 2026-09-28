"""The bridge is the harness's only loader of production `windowing.py`.

These tests guard the property the bridge exists for: one loaded copy, shared
by every caller. If a call site grows its own `spec_from_file_location` shim
again, `test_no_other_call_site_loads_windowing` fails.
"""

import re
import sys
from pathlib import Path

from eval_dataset.tools import replay, windowing_bridge

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRIDGE = _REPO_ROOT / "eval_dataset" / "tools" / "windowing_bridge.py"
_THIS_FILE = Path(__file__).resolve()  # holds the pattern it searches for

# Directories that may reach into the service directory at all.
_HARNESS_DIRS = (
    _REPO_ROOT / "eval_dataset" / "tools",
    _REPO_ROOT / "apps" / "semantic_analysis" / "ground_truth",
)

_SHIM_RE = re.compile(r"spec_from_file_location")


def test_bridge_points_at_the_production_module():
    assert windowing_bridge.WINDOWING_PATH == (
        _REPO_ROOT / "apps" / "semantic_analysis" / "windowing.py"
    )
    assert windowing_bridge.WINDOWING_PATH.is_file()


def test_constants_match_the_production_file():
    source = windowing_bridge.WINDOWING_PATH.read_text()
    assert f"UPPER = {windowing_bridge.UPPER}" in source
    assert f"LOWER = {windowing_bridge.LOWER}" in source
    assert f"MIN_CONTENT_TOKENS = {windowing_bridge.MIN_CONTENT_TOKENS}" in source


def test_module_is_loaded_once_and_cached():
    first = windowing_bridge.windowing
    assert sys.modules["_windowing"] is first
    # A second load must reuse the cached module, not execute the file again.
    assert windowing_bridge._load() is first


def test_call_sites_share_the_bridge_objects():
    # `replay` re-exports by assignment; the objects must be the bridge's own,
    # not equal-looking functions from a second load of the same file.
    assert replay.clean_text is windowing_bridge.clean_text
    assert replay.window_is_ready is windowing_bridge.window_is_ready
    assert replay.has_enough_content is windowing_bridge.has_enough_content


def test_no_other_call_site_loads_windowing():
    offenders = []
    for directory in _HARNESS_DIRS:
        for path in directory.rglob("*.py"):
            if path in (_BRIDGE, _THIS_FILE) or "venv" in path.parts:
                continue
            text = path.read_text()
            if _SHIM_RE.search(text) and "windowing" in text:
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], (
        "these files load windowing.py themselves instead of importing "
        f"eval_dataset.tools.windowing_bridge: {offenders}"
    )
