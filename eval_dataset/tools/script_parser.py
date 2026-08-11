from dataclasses import dataclass

VALID_CATEGORIES = {"clean", "gradual_drift", "adversarial", "code_switch", "silence"}


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


def parse_script(path: str) -> Script:
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

    return Script(
        topic=header["TOPIC"],
        teacher=header["TEACHER"],
        learner=header["LEARNER"],
        category=category,
        turns=turns,
    )
