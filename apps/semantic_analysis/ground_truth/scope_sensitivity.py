"""Does the encoder actually read the scope clause, or only the topic?

Each counterfactual utterance in topics.SCOPE_COUNTERFACTUALS is genuine
teaching content on one topic, labeled under two different scopes for that same
topic. If the model reads scope, an utterance labeled `correct` under scope A
and `weakly_correct` under scope B must score higher against A than against B.

This is a PAIRED, THRESHOLD-FREE diagnostic: it compares two similarities for
the same text, so no calibration is involved and no threshold can flatter the
result. Chance is 50%. It answers a different question from accuracy — accuracy
asks "is the boundary in the right place", this asks "is there any scope signal
for a boundary to use".

Usage:
    cd apps/semantic_analysis/ground_truth && python scope_sensitivity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import clean_text  # noqa: E402  (same cleaning as the live service)

from topics import SCOPE_COUNTERFACTUALS  # noqa: E402

RANK = {"correct": 2, "weakly_correct": 1, "incorrect": 0}


def main():
    from sentence_transformers import SentenceTransformer, util

    print("Loading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    wins = ties = losses = 0
    margins = []
    print(f"\n{'topic':<34} {'sim(A)':>8} {'sim(B)':>8} {'delta':>8}  expected")
    for entry in SCOPE_COUNTERFACTUALS:
        emb_a = model.encode(f"{entry['topic']}. {entry['scope_a']}", convert_to_tensor=True)
        emb_b = model.encode(f"{entry['topic']}. {entry['scope_b']}", convert_to_tensor=True)
        for pair in entry["pairs"]:
            emb = model.encode(clean_text(pair["text"]), convert_to_tensor=True)
            sim_a = float(util.cos_sim(emb, emb_a).item())
            sim_b = float(util.cos_sim(emb, emb_b).item())

            # Which scope do the labels say this utterance fits better?
            expected = RANK[pair["under_a"]] - RANK[pair["under_b"]]
            observed = sim_a - sim_b
            if expected == 0:
                verdict = "n/a"
            elif (observed > 0) == (expected > 0):
                wins += 1
                verdict = "A" if expected > 0 else "B"
                margins.append(abs(observed))
            else:
                losses += 1
                verdict = ("A" if expected > 0 else "B") + " MISSED"
                margins.append(-abs(observed))
            print(f"{entry['topic'][:33]:<34} {sim_a:8.4f} {sim_b:8.4f} {observed:+8.4f}  {verdict}")

    total = wins + losses
    print(f"\nPaired scope preference: {wins}/{total} correct ({wins / total:.0%}); chance is 50%.")
    if margins:
        mean_margin = sum(margins) / len(margins)
        print(f"Mean signed margin toward the labeled scope: {mean_margin:+.4f}")
    print(
        f"\n{total} paired comparisons over {len(SCOPE_COUNTERFACTUALS)} topics is a probe, "
        "not a powered test.\nDIAGNOSTIC ONLY: never fit thresholds on these rows "
        "(design_decisions.md, D7)."
    )


if __name__ == "__main__":
    main()
