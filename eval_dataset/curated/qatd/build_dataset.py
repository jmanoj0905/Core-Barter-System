"""Build the reviewed QATD selection; no LLM generation or training is performed.

Run with apps/semantic_analysis/venv/bin/python from the repository root.
The selected conversations, edits, contracts, and splits are in curation_plan.json.
"""

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read_json(path):
    return json.loads(path.read_text())


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def normalize(text, replacements):
    text = text.replace("➗", " divided by ").replace("✖", " times ")
    for original, replacement in replacements.items():
        text = re.sub(r"\b" + re.escape(original) + r"\b", replacement, text, flags=re.I)
    text = "".join(c for c in text if unicodedata.category(c) not in {"So", "Sk"}
                   and c not in "\ufe0f\u200d")
    text = re.sub(r"[:;=]-?[)(D]|<3", "", text)
    text = " ".join(text.split()).strip(" \\")
    return re.sub(r"\s+([,!.?])", r"\1", text)


def load_sessions(plan):
    manifest = read_json(HERE / "source_manifest.json")
    assert manifest["revision"] == plan["source_revision"]
    for file in manifest["files"]:
        content = (HERE / file["local_path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == file["sha256"], file["local_path"]
    originals = defaultdict(list)
    for split in ("train", "test"):
        with (HERE / "raw" / f"{split}.csv").open(newline="", encoding="utf-8-sig") as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                row["source_csv_line"] = line
                originals[(split, row["InterventionId"])].append(row)
    sessions = []
    question_splits = defaultdict(set)
    for selection in plan["sessions"]:
        source = originals[(selection["source_split"], selection["intervention_id"])]
        assert source, selection
        question_ids = {r["QuestionId_DQ"] for r in source}
        assert len(question_ids) == 1
        question_id = next(iter(question_ids))
        question_splits[question_id].add(selection["split"])
        selected = [r for r in source if selection["start_sequence"] <= int(r["MessageSequence"])
                    <= selection["end_sequence"]]
        assert len({r["TutorId"] for r in selected if r["IsTutor"] == "1"}) == 1
        turns = [{"role": "learner", "text": "Please help me work through this question: " + selection["problem"],
                  "origin": "agent_added_problem_context", "source_sequence": None}]
        for row in selected:
            original = row["MessageString"]
            text = normalize(original, selection.get("replacements", {}))
            if not text:
                continue
            turns.append({"role": "teacher" if row["IsTutor"] == "1" else "learner",
                          "text": text, "source_text": original,
                          "source_sequence": int(row["MessageSequence"]),
                          "source_csv_line": row["source_csv_line"],
                          "origin": "source_normalized" if text != original else "source_verbatim"})
        assert {t["role"] for t in turns} == {"teacher", "learner"}
        sessions.append({
            "session_id": "qatd-" + selection["intervention_id"],
            "source_intervention_id": selection["intervention_id"],
            "source_question_id": question_id, "source_split": selection["source_split"],
            "split": selection["split"], "contract_id": selection["contract"],
            "source_url": manifest["repository"] + "/blob/" + manifest["revision"]
                          + "/anchored-dialogues/" + selection["source_split"] + ".csv",
            "selection_reason": selection["selection_reason"], "turns": turns,
            "adaptations": ["Select a contiguous teaching excerpt; omit opening/closing platform logistics.",
                            "Standardize roles to one teacher and one learner.",
                            "Normalize whitespace/emojis and apply the recorded name replacements.",
                            "Add a learner problem statement adapted from source question metadata."],
            "replacements": selection.get("replacements", {}),
            "license": "CC-BY-NC-SA-4.0",
        })
    assert all(len(splits) == 1 for splits in question_splits.values()), "Question leaks across splits"
    assert len({s["session_id"] for s in sessions}) == len(sessions)
    return sessions


def window_session(session, plan):
    windows, buffer, seconds = [], [], 0.0
    turn_start = 0
    for index, turn in enumerate(session["turns"]):
        if turn["role"] == "teacher":
            buffer.append(turn)
            seconds += len(turn["text"].split()) * 60 / plan["words_per_minute"]
        if seconds >= plan["teacher_window_seconds"] or index == len(session["turns"]) - 1:
            if not buffer:
                continue
            text = " ".join(t["text"] for t in buffer)
            words = len(text.split())
            windows.append({
                "window_id": f"{session['session_id']}-w{len(windows) + 1:02}",
                "session_id": session["session_id"], "split": session["split"],
                "source_question_id": session["source_question_id"],
                "source_contract_id": session["contract_id"],
                "teacher_text": text,
                "teacher_source_sequences": [t["source_sequence"] for t in buffer],
                "context_turns": [{"role": t["role"], "text": t["text"]}
                                  for t in session["turns"][max(0, turn_start - 4):index + 1]],
                "estimated_teacher_seconds": round(seconds, 2), "teacher_word_count": words,
                "timing_source": "estimated_from_text_150_words_per_minute_not_recorded_audio",
                "is_final_partial_window": seconds < plan["teacher_window_seconds"],
                "scored": words >= plan["minimum_scored_teacher_words"],
                "exclusion_reason": None if words >= plan["minimum_scored_teacher_words"]
                                    else "Short final fragment retained for context, excluded from the scope challenge.",
            })
            buffer, seconds, turn_start = [], 0.0, index + 1
    return windows


def make_examples(windows, plan):
    examples = []
    contracts = plan["contracts"]
    for window in windows:
        if not window["scored"]:
            continue
        source_id = window["source_contract_id"]
        source = contracts[source_id]
        conditions = [
            (source_id, "correct", "original_lesson_scope",
             "The excerpt advances the source lesson skill, including clarification and scaffolding."),
            (source["adjacent_contract"], "weakly_correct", "adjacent_scope_counterfactual",
             "Same declared subject, but this excerpt teaches the other skill outside the target scope; no bridge to the target task is supplied."),
            (source["unrelated_contract"], "incorrect", "different_subject_counterfactual",
             "The excerpt teaches a different declared subject and does not advance the target contract. Shared arithmetic alone does not establish relevance."),
        ]
        for target_id, label, variant, reason in conditions:
            target = contracts[target_id]
            examples.append({
                **window, "example_id": window["window_id"] + "--" + target_id,
                "source": "qatd_human_dialogue_adapted_scope_challenge",
                "contract_id": target_id, "topic": target["topic"], "scope": target["scope"],
                "family": target["family"], "text": window["teacher_text"],
                "expected_label": label, "variant": variant, "label_rationale": reason,
                "label_author": plan["label_author"], "label_status": plan["label_status"],
                "natural_drift_observation": False, "license": "CC-BY-NC-SA-4.0",
            })
    return examples


def score(examples):
    # Reuse the live text cleaning function without starting the service.
    sys.path.insert(0, str(ROOT / "apps/semantic_analysis"))
    from main import clean_text
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    texts = sorted({clean_text(row["text"]) for row in examples}
    contracts = sorted({f"{row['topic']}. {row['scope']}" for row in examples})
    text_embeddings = dict(zip(texts, model.encode(texts, convert_to_tensor=True)))
    contract_embeddings = dict(zip(contracts, model.encode(contracts, convert_to_tensor=True)))
    for row in examples:
        row["similarity"] = round(float(util.cos_sim(
            text_embeddings[clean_text(row["text"])],
            contract_embeddings[f"{row['topic']}. {row['scope']}"]).item()), 6)


def main():
    plan = read_json(HERE / "curation_plan.json")
    sessions = load_sessions(plan)
    windows = [w for session in sessions for w in window_session(session, plan)]
    examples = make_examples(windows, plan)
    score(examples)
    write_jsonl(HERE / "sessions.jsonl", sessions)
    write_jsonl(HERE / "windows.jsonl", windows)
    write_jsonl(HERE / "examples.jsonl", examples)
    fields = ["source", "topic", "scope", "text", "expected_label", "similarity", "session_id",
              "window_id", "family", "source_question_id", "example_id", "variant", "label_status",
              "estimated_teacher_seconds", "timing_source"]
    for split in ("calibration", "evaluation"):
        rows = [r for r in examples if r["split"] == split]
        with (HERE / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    preview = ["# Adapted one-on-one tutoring sessions", "",
               "Real QATD conversations; excerpts and edits are recorded in curation_plan.json.",
               "The initial learner problem prompt is agent-added. Remaining turns come from the cited session.",
               "License: CC BY-NC-SA 4.0. Attribution: Eedi; Matthew Zent, Digory Smith, Simon Woodhead.", ""]
    for session in sessions:
        preview += [f"## {session['session_id']} — {session['contract_id']} ({session['split']})", "",
                    f"[Source CSV]({session['source_url']}); question {session['source_question_id']}.",
                    session["selection_reason"], ""]
        for turn in session["turns"]:
            marker = f"source turn {turn['source_sequence']}" if turn["source_sequence"] else "added problem context"
            preview += [f"**{turn['role'].title()}** ({marker}): {turn['text']}", ""]
    (HERE / "dialogue_preview.md").write_text("\n".join(preview))
    summary = {
        "source_sessions": len(sessions), "source_questions": len({s["source_question_id"] for s in sessions}),
        "contracts": len(plan["contracts"]), "families": len({c["family"] for c in plan["contracts"].values()}),
        "windows_retained": len(windows), "scored_windows": sum(w["scored"] for w in windows),
        "excluded_short_windows": sum(not w["scored"] for w in windows),
        "examples": len(examples), "label_status": plan["label_status"],
        "splits": {split: {"sessions": sum(s["split"] == split for s in sessions),
                           "examples": sum(r["split"] == split for r in examples),
                           "labels": dict(Counter(r["expected_label"] for r in examples if r["split"] == split))}
                   for split in ("calibration", "evaluation")},
        "independent_human_gold": False, "recorded_audio_timestamps": False,
        "natural_drift_benchmark": False,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
