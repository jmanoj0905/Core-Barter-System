"""Structural gold labels for eval windows.

A script's author declares which dialogue turns were digressions via the
DIGRESSION header (parsed into Script.digression_turns). This module maps
that declared intent onto each replayed window's label. It is intentionally
mechanism-free: it never looks at a window's cosine or classification, only
at which turns contributed how many seconds to the window. Consulting the
model's own output here would make the calibration circular.

Spec §5.2 requires the [0.4, 0.6] exclusion band because a window straddling
roughly 50/50 has no honest ground truth: rounding it either way would inject
label noise exactly at the calibration threshold this module's callers are
trying to fit. A future maintainer should not "simplify" this to a >= 0.5
cutoff — that would manufacture false certainty precisely where the labels
are least trustworthy.
"""

from eval_dataset.tools.replay import ReplayWindow
from eval_dataset.tools.script_parser import Script

# Closed exclusion band: shares in [0.4, 0.6] are not labellable. A small
# float tolerance keeps this robust against the fact that shares are computed
# from float seconds division (e.g. 2/5 or 3/5 computed from thirds can land
# a few epsilons off 0.4/0.6). Exclusion is the safe direction on either side
# of that tolerance, so the tolerance only ever widens the excluded band.
_BAND_LOW = 0.4
_BAND_HIGH = 0.6
_EPS = 1e-9


def gold_label(window: ReplayWindow, script: Script) -> str | None:
    """Derive the gold label for a window from the script's declared digressions.

    Returns "on_topic" if the window's off-topic share is < 0.4, "off_topic"
    if > 0.6, and None (excluded from calibration) if the share falls in the
    closed band [0.4, 0.6] (with float tolerance), or if the window has zero
    total contribution seconds and therefore no defined share.
    """
    total_seconds = sum(c.seconds for c in window.contributions)

    # A window with zero total seconds has no defined off-topic share. This
    # should never arise in practice (a window with no contributions is never
    # emitted by the windower), but guard it explicitly rather than letting a
    # ZeroDivisionError propagate: not labellable, so excluded.
    if total_seconds == 0:
        return None

    off_topic_seconds = sum(
        c.seconds for c in window.contributions if c.turn_index in script.digression_turns
    )
    share = off_topic_seconds / total_seconds

    if share < _BAND_LOW - _EPS:
        return "on_topic"
    if share > _BAND_HIGH + _EPS:
        return "off_topic"
    return None
