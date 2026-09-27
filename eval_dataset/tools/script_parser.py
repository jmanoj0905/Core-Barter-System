from dataclasses import dataclass

VALID_CATEGORIES = {
    "clean",
    "gradual_drift",
    "adversarial",
    "code_switch",
    "silence",
    "depressed_baseline",
    "abstract_on_topic",
    "interleaved_abuse",
}

# Categories whose DIGRESSION set is required to be non-empty when
# require_digression_labels=True (the drift categories: the corpus's gold
# labels live on these turns).
_DRIFT_CATEGORIES = {"gradual_drift", "adversarial", "interleaved_abuse"}

# Categories whose DIGRESSION set is required to be empty when
# require_digression_labels=True (on-topic / no-drift categories).
_NO_DRIFT_CATEGORIES = {"clean", "silence", "depressed_baseline", "abstract_on_topic"}

# code_switch is deliberately left unconstrained in both directions — existing
# scripts use it and this plan does not re-label them.


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class Script:
    topic: str
    teacher: str
    learner: str
    category: str
    turns: list[Turn]
    scope: str = ""
    author: str = ""
    digression_turns: frozenset[int] = frozenset()
    source: str | None = None


def _parse_digression(value: str, turns: list[Turn], teacher: str) -> frozenset[int]:
    """Expand a DIGRESSION header value into a set of 1-indexed turn numbers.

    Numbering is 1-indexed over ALL dialogue turns (teacher and learner
    alike), inclusive ranges, comma-separated: "5-9, 14" -> {5,6,7,8,9,14}.
    Every named index must fall within 1..len(turns) and must name a turn
    spoken by the teacher (only teacher speech reaches a window).
    """
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s.strip()), int(end_s.strip())
            indices = range(start, end + 1)
        else:
            indices = [int(part)]
        for idx in indices:
            if idx < 1 or idx > len(turns):
                raise ValueError(
                    f"DIGRESSION names turn {idx}, but the script only has "
                    f"{len(turns)} dialogue turns"
                )
            speaker = turns[idx - 1].speaker
            if speaker != teacher:
                raise ValueError(
                    f"DIGRESSION names turn {idx}, but that turn is spoken by "
                    f"the learner ({speaker!r}), not the teacher — only "
                    "teacher speech reaches a window"
                )
            result.add(idx)
    return frozenset(result)


def parse_script(path: str, *, require_digression_labels: bool = False) -> Script:
    """Parse a corpus script file.

    require_digression_labels gates the category<->DIGRESSION *presence*
    rules (drift categories must declare a non-empty DIGRESSION set;
    on-topic categories must declare none). It defaults to False so that
    scripts written before this header field existed — and the two
    original topic-pool scripts — keep parsing; the calibration corpus
    tooling (tasks 8 and 11) passes True to enforce gold labels are present.
    Index-range and teacher-speaker validation on DIGRESSION are
    unconditional: they are authoring errors whenever the line is present
    at all, regardless of this flag.
    """
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    header = {}
    turn_lines = []
    in_turns = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            in_turns = True
            continue
        if not in_turns and ":" in stripped and stripped.split(":", 1)[0].isupper():
            key, value = stripped.split(":", 1)
            header[key.strip()] = value.strip()
        elif stripped:
            turn_lines.append(stripped)

    category = header["CATEGORY"]
    if category not in VALID_CATEGORIES:
        raise ValueError(f"invalid category: {category!r}, must be one of {VALID_CATEGORIES}")

    turns = []
    for line in turn_lines:
        speaker, text = line.split(":", 1)
        turns.append(Turn(speaker=speaker.strip(), text=text.strip()))

    if not turns:
        raise ValueError(
            "script has no dialogue turns — check for a missing blank line "
            "between the header and the first turn"
        )

    teacher = header["TEACHER"]

    digression_value = header.get("DIGRESSION", "")
    if digression_value:
        digression_turns = _parse_digression(digression_value, turns, teacher)
    else:
        digression_turns = frozenset()

    if require_digression_labels:
        if category in _DRIFT_CATEGORIES and not digression_turns:
            raise ValueError(
                f"category {category!r} requires DIGRESSION to declare at "
                "least one turn"
            )
        if category in _NO_DRIFT_CATEGORIES and digression_turns:
            raise ValueError(
                f"category {category!r} must not declare DIGRESSION turns"
            )

    return Script(
        topic=header["TOPIC"],
        teacher=teacher,
        learner=header["LEARNER"],
        category=category,
        turns=turns,
        scope=header.get("SCOPE", ""),
        author=header.get("AUTHOR", ""),
        digression_turns=digression_turns,
        source=header.get("SOURCE"),
    )
