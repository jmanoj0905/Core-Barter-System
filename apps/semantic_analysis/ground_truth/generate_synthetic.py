"""Build a synthetic labeled dataset for threshold calibration.

Runs each hand-labeled example in topics.py through the same embedding
model and text cleaning the live service uses, records the cosine
similarity against its topic, and writes one row per example to
synthetic_dataset.csv. Output columns match export_real_labels.py so the
two can be concatenated for calibrate_thresholds.py.

Usage:
    cd apps/semantic_analysis/ground_truth && python generate_synthetic.py
"""

import csv
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import clean_text  # noqa: E402  (reuse the live cleaning logic)

from topics import TOPICS  # noqa: E402

OUTPUT_PATH = Path(__file__).parent / "synthetic_dataset.csv"


def main():
    print("Loading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    rows = []
    for entry in TOPICS:
        topic_text = f"{entry['topic']}. {entry['scope']}"
        topic_embedding = model.encode(topic_text, convert_to_tensor=True)

        for expected_label in ("correct", "weakly_correct", "incorrect"):
            for text in entry[expected_label]:
                cleaned = clean_text(text)
                embedding = model.encode(cleaned, convert_to_tensor=True)
                similarity = float(util.cos_sim(embedding, topic_embedding).item())
                rows.append({
                    "source": "synthetic",
                    "topic": entry["topic"],
                    "scope": entry["scope"],
                    "text": text,
                    "expected_label": expected_label,
                    "similarity": round(similarity, 4),
                })

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "topic", "scope", "text", "expected_label", "similarity"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} labeled rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
