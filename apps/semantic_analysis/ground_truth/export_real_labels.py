"""Export human-labeled window_results from the backend DB as calibration data.

Only rows with a human_label in the three-class task count — those are the ones a
participant actually rated via the referee "rate this" feedback button
(see POST /session/{barter_id}/window/{window_id}/feedback). The model's
own `classification` is not used as ground truth here; `human_label` is.

Output adds session_id and window_id to the generate_synthetic.py columns so
threshold_experiment.py can reject shared calibration/evaluation sessions.
The fourth label, out_of_scope, is excluded rather than silently remapped.
Note: real rows don't carry the raw cosine
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
from contextlib import closing
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "backend" / "barter.db"
DEFAULT_OUTPUT = Path(__file__).parent / "real_dataset.csv"

QUERY = """
SELECT
    sc.topic,
    sc.scope,
    wr.text_content,
    wr.human_label,
    wr.cosine_similarity,
    wr.barter_session_id,
    wr.window_number
FROM window_results wr
JOIN session_contracts sc ON sc.barter_session_id = wr.barter_session_id
WHERE wr.human_label IN ('correct', 'weakly_correct', 'incorrect')
ORDER BY wr.barter_session_id, wr.window_number
"""


def main():
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not db_path.exists():
        print(f"No DB found at {db_path}")
        return

    if db_path.resolve() == output_path.resolve():
        raise SystemExit("Output must not overwrite the database")
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(window_results)")}
        if "human_label" not in columns:
            raise SystemExit("Cannot export: window_results has no human_label column. "
                             "Use a database with the feedback schema and human-rated windows.")
        rows = conn.execute(QUERY).fetchall()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "topic", "scope", "text", "expected_label", "similarity",
                         "session_id", "window_id"])
        for topic, scope, text, human_label, similarity, session_id, window_id in rows:
            writer.writerow(["real", topic, scope, text, human_label, round(similarity, 4),
                             session_id, window_id])

    print(f"Wrote {len(rows)} human-labeled rows to {output_path}")


if __name__ == "__main__":
    main()
