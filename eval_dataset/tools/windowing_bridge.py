"""The one place the offline harness loads production `windowing.py`.

`apps/semantic_analysis` is not an importable package from the repo root — it
is a service directory with its own venv — so the harness reaches production
windowing logic through `importlib.util.spec_from_file_location`. That shim
used to be copy-pasted at four call sites (`replay`, `embed_windows`,
`rho_calibration`, and a test), which loaded the module four independent times
under four different names. Four live copies of a module whose whole purpose is
that production and the harness *cannot* diverge is the opposite of the point:
patch one in a test and the other three keep the old constants.

So the load happens exactly once, here, and every caller imports the names from
this module. The loaded module is registered in `sys.modules` under
`_windowing`, so a re-import of this bridge reuses it rather than re-executing
the file.

Import-light by construction: `windowing.py` imports only `re`, and this module
adds nothing heavier.
"""

import importlib.util
import sys
from pathlib import Path

WINDOWING_PATH = (
    Path(__file__).resolve().parents[2] / "apps" / "semantic_analysis" / "windowing.py"
)

_MODULE_NAME = "_windowing"


def _load():
    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, WINDOWING_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[_MODULE_NAME]
        raise
    return module


windowing = _load()

# Re-exported so callers read as if `windowing` were a normal import. Every
# public name in the production module is listed: a caller that needs one of
# these must not reach for a second loader to get it.
UPPER = windowing.UPPER
LOWER = windowing.LOWER
FILLER_WORDS = windowing.FILLER_WORDS
FILLER_PHRASES = windowing.FILLER_PHRASES
MIN_CONTENT_TOKENS = windowing.MIN_CONTENT_TOKENS
WINDOW_DURATION_THRESHOLD = windowing.WINDOW_DURATION_THRESHOLD

clean_text = windowing.clean_text
classify = windowing.classify
topic_text = windowing.topic_text
window_is_ready = windowing.window_is_ready
has_enough_content = windowing.has_enough_content

__all__ = [
    "WINDOWING_PATH",
    "windowing",
    "UPPER",
    "LOWER",
    "FILLER_WORDS",
    "FILLER_PHRASES",
    "MIN_CONTENT_TOKENS",
    "WINDOW_DURATION_THRESHOLD",
    "clean_text",
    "classify",
    "topic_text",
    "window_is_ready",
    "has_enough_content",
]
