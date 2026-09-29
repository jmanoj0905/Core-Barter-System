import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ground_truth"))
from evaluate_reversal_detection import load_corpus, evaluate_corpus


def test_load_corpus_reads_json(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps([
        {"text": "I will pay you.", "category": "commitment", "expected_reversal": False},
    ]))
    records = load_corpus(corpus_path)
    assert records == [{"text": "I will pay you.", "category": "commitment", "expected_reversal": False}]


def test_evaluate_corpus_computes_precision_recall_and_by_category():
    records = [
        {"text": "I will not pay you.", "category": "reversal", "expected_reversal": True},
        {"text": "I will pay you.", "category": "commitment", "expected_reversal": False},
        {"text": "I am not sure about the weather, but I will pay you the full amount.",
         "category": "distractor", "expected_reversal": False},
    ]
    result = evaluate_corpus(records)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["by_category"]["reversal"]["correct"] == 1
    assert result["by_category"]["commitment"]["correct"] == 1
    assert result["by_category"]["distractor"]["correct"] == 1


def test_evaluate_corpus_counts_confusion_and_misclassified():
    records = [
        {"text": "I will not pay you.", "category": "reversal", "expected_reversal": True},
        {"text": "I am paying you tomorrow.", "category": "commitment", "expected_reversal": False},
        {"text": "I will not refuse to pay you.", "category": "double_negation", "expected_reversal": False},
        {"text": "I am not paying you.", "category": "reversal", "expected_reversal": True},
    ]
    result = evaluate_corpus(records)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert {m["text"] for m in result["misclassified"]} == {
        "I will not refuse to pay you.", "I am not paying you.",
    }


def test_evaluate_corpus_zero_denominators_default_to_one():
    records = [{"text": "I will pay you.", "category": "commitment", "expected_reversal": False}]
    result = evaluate_corpus(records)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
