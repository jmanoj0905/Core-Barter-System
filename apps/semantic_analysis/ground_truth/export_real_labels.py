"""Export human-labeled window_results from the backend DB as calibration data.

Only rows with a non-null human_label count — those are the ones a
participant actually rated via the referee "rate this" feedback button
(see POST /session/{barter_id}/window/{window_id}/feedback). The model's
own `classification` is not used as ground truth here; `human_label` is.

Output columns match generate_synthetic.py so the two can be concatenated
for calibrate_thresholds.py. Note: real rows don't carry the raw cosine
similarity of the *cleaned* text at export time consistently with the
synthetic set unless the backend already stored it — this script reuses
the `cosine_similarity` the live service computed and logged at window
time (window_results.cosine_similarity), not a recomputation.

Usage:
    python export_real_labels.py [path/to/barter.db] [output.csv]
"""

import csv
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "backend" / "barter.db"
DEFAULT_OUTPUT = Path(__file__).parent / "real_dataset.csv"

QUERY = """
SELECT
    sc.topic,
    sc.scope,
    wr.text_content,
    wr.human_label,
    wr.cosine_similarity
FROM window_results wr
JOIN session_contracts sc ON sc.barter_session_id = wr.barter_session_id
WHERE wr.human_label IS NOT NULL
"""


def main():
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not db_path.exists():
        print(f"No DB found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    rows = conn.execute(QUERY).fetchall()
    conn.close()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "topic", "scope", "text", "expected_label", "similarity"])
        for topic, scope, text, human_label, similarity in rows:
            writer.writerow(["real", topic, scope, text, human_label, round(similarity, 4)])

    print(f"Wrote {len(rows)} human-labeled rows to {output_path}")


if __name__ == "__main__":
    main()
