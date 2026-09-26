import csv
import sqlite3
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("export_real_labels.py")


def test_export_preserves_sessions_and_uses_human_labels(tmp_path):
    database, output = tmp_path / "barter.db", tmp_path / "labels.csv"
    with sqlite3.connect(database) as conn:
        conn.executescript("""
            CREATE TABLE session_contracts (barter_session_id INTEGER, topic TEXT, scope TEXT);
            CREATE TABLE window_results (barter_session_id INTEGER, window_number INTEGER,
                text_content TEXT, human_label TEXT, cosine_similarity REAL, classification TEXT);
            INSERT INTO session_contracts VALUES (7, 'Topic', 'Scope');
            INSERT INTO window_results VALUES (7, 1, 'Human-rated window', 'weakly_correct', .7, 'correct');
            INSERT INTO window_results VALUES (7, 2, 'Unrated window', NULL, .9, 'correct');
            INSERT INTO window_results VALUES (7, 3, 'Outside three-label task', 'out_of_scope', .1, 'incorrect');
        """)
    result = subprocess.run([sys.executable, str(SCRIPT), str(database), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["expected_label"] == "weakly_correct"
    assert rows[0]["session_id"] == "7"
    assert rows[0]["window_id"] == "1"


def test_old_database_reports_missing_labels_without_creating_output(tmp_path):
    database, output = tmp_path / "old.db", tmp_path / "labels.csv"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE window_results (id INTEGER)")
    result = subprocess.run([sys.executable, str(SCRIPT), str(database), str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "human_label" in result.stderr
    assert not output.exists()
