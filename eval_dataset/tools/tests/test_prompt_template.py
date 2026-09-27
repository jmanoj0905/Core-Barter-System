from pathlib import Path

from eval_dataset.tools.authoring import CATEGORY_DESCRIPTIONS

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "authoring" / "prompt_template.md"
)

FORBIDDEN = ["cosine", "similarity", "embedding", "threshold", "rho", "RHO",
             "0.36", "0.14", "classifier", "sentence-bert", "sbert",
             "window", "warning", "escrow", "drift score"]


def test_template_leaks_no_mechanism_vocabulary():
    text = TEMPLATE_PATH.read_text().lower()
    for term in FORBIDDEN:
        assert term.lower() not in text, f"blinding leak: {term}"


def test_category_descriptions_leak_nothing():
    for name, desc in CATEGORY_DESCRIPTIONS.items():
        for term in FORBIDDEN:
            assert term.lower() not in desc.lower(), f"{name} leaks {term}"


def test_every_corpus_category_has_a_description():
    assert set(CATEGORY_DESCRIPTIONS) == {
        "gradual_drift", "adversarial", "depressed_baseline",
        "abstract_on_topic", "interleaved_abuse"}


def test_template_has_all_placeholders():
    text = TEMPLATE_PATH.read_text()
    placeholders = [
        "{topic}", "{scope}", "{category_description}",
        "{teacher_word_target}", "{teacher_letter}", "{learner_letter}",
    ]
    for placeholder in placeholders:
        assert text.count(placeholder) == 1, (
            f"{placeholder} appears {text.count(placeholder)} times, expected exactly 1"
        )


def test_depressed_baseline_and_abstract_on_topic_forbid_digression():
    for name in ("depressed_baseline", "abstract_on_topic"):
        desc = CATEGORY_DESCRIPTIONS[name].lower()
        assert "never" in desc or "no " in desc or "without" in desc, (
            f"{name} description must explicitly instruct staying on topic throughout"
        )
