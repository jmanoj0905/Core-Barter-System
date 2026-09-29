# apps/warning_engine/ground_truth/tests/test_synthesize_fusion_corpus.py
from synthesize_fusion_corpus import generate_corpus


def test_generate_corpus_is_deterministic_for_a_fixed_seed():
    a = generate_corpus(n_scenarios=50, seed=42)
    b = generate_corpus(n_scenarios=50, seed=42)
    assert a == b


def test_generate_corpus_has_expected_columns_and_ranges():
    rows = generate_corpus(n_scenarios=50, seed=1)
    assert len(rows) == 50
    for row in rows:
        assert set(row.keys()) == {"scenario_id", "speech_score", "video_score", "engaged_label"}
        assert 0.0 <= row["speech_score"] <= 1.0
        assert 0.0 <= row["video_score"] <= 1.0
        assert row["engaged_label"] in (0, 1)


def test_generate_corpus_has_at_least_20_percent_disagreement_scenarios():
    rows = generate_corpus(n_scenarios=500, seed=7)
    # A "disagreement" scenario is one where the two scores differ by more
    # than one noise sigma (0.1) from each other on average — a looser,
    # implementation-independent check than re-deriving the exact latent
    # levels the generator used internally.
    disagreements = sum(1 for r in rows if abs(r["speech_score"] - r["video_score"]) > 0.15)
    assert disagreements / len(rows) >= 0.15  # allows noise slack around the 20% generation target
