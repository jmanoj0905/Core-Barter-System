"""Offline replay windower — reproduces production `semantic_analysis` windowing.

Walks a parsed `Script`'s turns in speaking order, accumulating only teacher
speech into a buffer exactly the way `apps/semantic_analysis/main.py` does:
excess seconds are discarded (not carried forward) when a window closes, the
window counter increments before the content check (so a skipped window still
consumes an index), and a trailing non-empty buffer is flushed as a final
window after the last turn.

This module does windowing only — no cosine, no embedding, no scoring.
"""

from dataclasses import dataclass

from eval_dataset.tools.script_parser import Script
from eval_dataset.tools.windowing_bridge import (
    clean_text,
    has_enough_content,
    window_is_ready,
)


@dataclass(frozen=True)
class TurnContribution:
    turn_index: int  # 1-indexed, matching DIGRESSION
    seconds: float


@dataclass(frozen=True)
class ReplayWindow:
    index: int  # production window_counter value
    text: str  # cleaned
    raw_text: str
    teacher_seconds_start: float
    teacher_seconds_end: float
    contributions: tuple[TurnContribution, ...]


def synthetic_durations(script: Script) -> list[float]:
    """0.4 s/word (150 wpm) per turn, positionally aligned with script.turns."""
    return [len(turn.text.split()) * 0.4 for turn in script.turns]


def replay(script: Script, durations: list[float]) -> list[ReplayWindow]:
    """Replay a script into the windows production would have produced.

    Only turns whose speaker equals `script.teacher` feed the buffer. A turn
    is atomic — never split across windows. When the buffer closes,
    accumulated_seconds resets to 0.0 (excess is discarded, never carried
    forward). The window counter increments before the has_enough_content
    check, so a skipped window still consumes an index. Does not mutate
    `script`.
    """
    windows: list[ReplayWindow] = []
    window_counter = 0

    buffered_raw_texts: list[str] = []
    buffered_contributions: list[TurnContribution] = []
    accumulated_seconds = 0.0
    teacher_clock = 0.0

    def close_window() -> None:
        nonlocal window_counter, buffered_raw_texts, buffered_contributions
        nonlocal accumulated_seconds, teacher_clock

        window_counter += 1
        raw_text = " ".join(buffered_raw_texts)
        cleaned = clean_text(raw_text)

        window_seconds = sum(c.seconds for c in buffered_contributions)
        start = teacher_clock
        end = teacher_clock + window_seconds

        if has_enough_content(cleaned):
            windows.append(
                ReplayWindow(
                    index=window_counter,
                    text=cleaned,
                    raw_text=raw_text,
                    teacher_seconds_start=start,
                    teacher_seconds_end=end,
                    contributions=tuple(buffered_contributions),
                )
            )

        teacher_clock = end
        buffered_raw_texts = []
        buffered_contributions = []
        accumulated_seconds = 0.0

    for turn_index, turn in enumerate(script.turns, start=1):
        if turn.speaker != script.teacher:
            continue

        seconds = durations[turn_index - 1]
        buffered_raw_texts.append(turn.text)
        buffered_contributions.append(TurnContribution(turn_index=turn_index, seconds=seconds))
        accumulated_seconds += seconds

        if window_is_ready(accumulated_seconds):
            close_window()

    if buffered_raw_texts:
        close_window()

    return windows
