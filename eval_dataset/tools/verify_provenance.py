"""Verify that a script excerpted from real dialogue is genuinely verbatim.

Task 11 splices spans of AnnoMI dialogue into calibration scripts. The
cleaner agent that does the splicing is supposed to be a checkable
*extractor* — it selects character ranges out of real source records and
never writes new words. This module is the check. A script's provenance
file names, for every dialogue line, exactly which source record and
character range it was lifted from; `verify()` re-derives each line from
its citation and raises if anything does not line up.

Guiding principle (Review Focus #5 of the plan): every failure path must
raise `ProvenanceError`, never return cleanly. In particular, a substring
test against a *missing or empty* source record must not be allowed to
pass by accident — `"anything" in ""` is `False`, so a careless
implementation could "correctly" reject an empty source by luck while
happily returning `None` (success) for a genuinely missing one. Both cases
are checked explicitly below, before any substring comparison runs.

Whitespace normalisation policy (deliberate, and worth stating plainly):
we collapse runs of whitespace to a single space and strip leading/
trailing whitespace before comparing text — real transcripts carry
irregular spacing (double spaces, stray tabs, trailing newlines) and the
script format puts one turn per physical line, so some whitespace
variance between the source CSV and the authored script line is expected
and harmless. We do NOT lowercase, and we do NOT strip or normalise
punctuation. Either of those would let a reworded or "cleaned up" line
pass as verbatim, which is exactly the failure this module exists to
catch. Whitespace collapsing is the minimum needed to make verbatim
excerpting workable; anything more is a hole.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from eval_dataset.tools.script_parser import parse_script

# record_id -> full source text (all utterances of that conversation,
# concatenated in order). Populated lazily by _default_source_loader so
# that importing this module never touches the network or requires the
# (gitignored) AnnoMI corpus to be present.
_conversation_text_cache: dict[str, str] | None = None


class ProvenanceError(Exception):
    """Raised whenever a script's claimed provenance cannot be verified.

    Never raised generically — every call site names the script, the
    offending line/span, and what was expected, since Task 11 uses these
    messages as the debugging surface for re-extracting failing scripts.
    """


@dataclass(frozen=True)
class Provenance:
    dataset: str
    records: tuple[str, ...]  # source conversation ids, in splice order
    spans: tuple[tuple[str, int, int], ...]  # (record_id, char_start, char_end)
    seams: tuple[int, ...]  # turn indices where a splice occurs


def _normalize(text: str) -> str:
    """Collapse whitespace runs to a single space; strip the ends.

    Deliberately does nothing else — no lowercasing, no punctuation
    stripping. See the module docstring for why.
    """
    return re.sub(r"\s+", " ", text).strip()


def _load_provenance(provenance_path: str) -> Provenance:
    path = Path(provenance_path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ProvenanceError(
            f"provenance file not found: {provenance_path}"
        ) from exc

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProvenanceError(
            f"provenance file {provenance_path} is not valid JSON: {exc}"
        ) from exc

    try:
        return Provenance(
            dataset=raw["dataset"],
            records=tuple(raw["records"]),
            spans=tuple(tuple(span) for span in raw["spans"]),
            seams=tuple(raw["seams"]),
        )
    except KeyError as exc:
        raise ProvenanceError(
            f"provenance file {provenance_path} is missing required field {exc}"
        ) from exc


def _default_source_loader(record_id: str) -> str | None:
    """Look up a record's full text from the real AnnoMI loader (Task 9).

    Returns the concatenated utterance text for `record_id`, or None if no
    conversation with that id was loaded. Lazily imports and calls
    load_conversations() so that merely importing this module (or running
    unit tests that inject their own loader) never requires the gitignored
    raw/ corpus to be present.
    """
    global _conversation_text_cache
    if _conversation_text_cache is None:
        from eval_dataset.curated.annomi.build_dataset import load_conversations

        _conversation_text_cache = {
            conversation.id: "\n".join(u.text for u in conversation.utterances)
            for conversation in load_conversations()
        }
    return _conversation_text_cache.get(record_id)


SourceLoader = Callable[[str], "str | None"]


def verify(
    script_path: str,
    provenance_path: str,
    *,
    source_loader: SourceLoader | None = None,
) -> None:
    """Verify that every line of the script at `script_path` is a verbatim
    excerpt from the source record(s) cited in `provenance_path`.

    Raises ProvenanceError on any discrepancy. Returns None on success.

    `source_loader(record_id) -> str | None` supplies the full source text
    for a record: None (or an exception) means "no such record" (missing),
    "" means "record exists but has no text" (empty) — both are checked
    explicitly and raise, never fall through to a vacuous pass. Defaults
    to the real AnnoMI loader; tests inject a synthetic one so they never
    need the fetched corpus.
    """
    script = parse_script(script_path)
    provenance = _load_provenance(provenance_path)
    loader = source_loader or _default_source_loader

    turns = script.turns
    spans = provenance.spans

    if len(spans) != len(turns):
        raise ProvenanceError(
            f"{script_path}: provenance cites {len(spans)} span(s) but the "
            f"script has {len(turns)} dialogue turn(s) — every dialogue "
            "line must be covered by exactly one cited span, and this "
            "script has an uncited (or over-cited) line"
        )

    # --- resolve and validate every cited source record up front -------
    source_cache: dict[str, str] = {}

    def get_source(record_id: str) -> str:
        if record_id in source_cache:
            return source_cache[record_id]
        try:
            text = loader(record_id)
        except Exception as exc:  # noqa: BLE001 - re-raised as ProvenanceError
            raise ProvenanceError(
                f"{script_path}: source record {record_id!r} could not be "
                f"loaded: {exc}"
            ) from exc
        if text is None:
            raise ProvenanceError(
                f"{script_path}: source record {record_id!r} is missing — "
                "no such record was found by the source loader; a "
                "provenance citation must name a record that actually "
                "exists"
            )
        if text == "":
            raise ProvenanceError(
                f"{script_path}: source record {record_id!r} has empty "
                "source text — an empty source cannot verify any span "
                "cited against it"
            )
        source_cache[record_id] = text
        return text

    # --- records field must match the de-duplicated splice order of spans
    observed_order: list[str] = []
    for record_id, _start, _end in spans:
        if not observed_order or observed_order[-1] != record_id:
            observed_order.append(record_id)
    if tuple(observed_order) != provenance.records:
        raise ProvenanceError(
            f"{script_path}: provenance.records {provenance.records!r} does "
            f"not match the splice order implied by spans "
            f"{tuple(observed_order)!r}"
        )

    # --- seams must fall exactly on genuine record-transition boundaries
    actual_transitions = {
        idx
        for idx in range(1, len(spans))
        if spans[idx][0] != spans[idx - 1][0]
    }
    for seam in provenance.seams:
        if not (1 <= seam <= len(turns) - 1):
            raise ProvenanceError(
                f"{script_path}: seam at turn index {seam} is out of range "
                f"— a seam must name an interior turn boundary between "
                f"1 and {len(turns) - 1} (there is no turn before the "
                "first line or after the last), so this does not fall on "
                "a turn boundary"
            )
    declared_seams = set(provenance.seams)
    if declared_seams != actual_transitions:
        spurious = sorted(declared_seams - actual_transitions)
        missing = sorted(actual_transitions - declared_seams)
        detail_parts = []
        if spurious:
            detail_parts.append(
                f"declared seam(s) {spurious} do not fall on an actual "
                "record transition (the spans on both sides cite the same "
                "record, so the seam falls inside a contiguous span, not "
                "on a turn boundary between two different sources)"
            )
        if missing:
            detail_parts.append(
                f"turn boundary(ies) {missing} are an actual record "
                "transition but are not declared as a seam"
            )
        raise ProvenanceError(
            f"{script_path}: seams {sorted(declared_seams)} do not match "
            f"the record transitions implied by spans "
            f"{sorted(actual_transitions)} — " + "; ".join(detail_parts)
        )

    # --- every line must be a verbatim excerpt of its cited span --------
    for turn_idx, (turn, span) in enumerate(zip(turns, spans)):
        record_id, start, end = span
        source_text = get_source(record_id)

        if not (0 <= start < end <= len(source_text)):
            raise ProvenanceError(
                f"{script_path}: span for turn {turn_idx + 1} "
                f"({record_id!r}, {start}, {end}) is out of bounds for "
                f"source text of length {len(source_text)}"
            )

        expected = _normalize(turn.text)
        excerpt = _normalize(source_text[start:end])

        if excerpt != expected:
            raise ProvenanceError(
                f"{script_path}: turn {turn_idx + 1} ({turn.speaker!r}: "
                f"{turn.text!r}) does not match the text cited at "
                f"({record_id!r}, {start}, {end}) in the source record, "
                f"which reads {source_text[start:end]!r} — the line is "
                "not a verbatim excerpt of its citation"
            )

        if expected not in _normalize(source_text):
            # Defence in depth: should be unreachable given the exact-span
            # check above (the excerpt is sliced from source_text), but
            # kept explicit rather than assumed, per the module's
            # "never pass vacuously" principle.
            raise ProvenanceError(
                f"{script_path}: turn {turn_idx + 1} text is not found "
                f"verbatim anywhere in source record {record_id!r}"
            )
