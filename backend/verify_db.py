"""Verify that all tables exist and seed data is correct."""

import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()

EXPECTED_TABLES = [
    "users",
    "barter_sessions",
    "session_contracts",
    "window_results",
    "warnings",
    "verdicts",
]


def verify():
    db_url = os.environ["DATABASE_URL_SYNC"]
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Check tables exist
    cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    existing = {row[0] for row in cur.fetchall()}

    missing = [t for t in EXPECTED_TABLES if t not in existing]
    if missing:
        print(f"FAIL: Missing tables: {missing}")
        sys.exit(1)
    print(f"OK: All {len(EXPECTED_TABLES)} tables exist.")

    # Check seed data
    cur.execute("SELECT username, trust_score FROM users ORDER BY username")
    rows = cur.fetchall()
    expected_users = [("alice", 0.30), ("bob", 0.30)]

    if len(rows) < 2:
        print(f"FAIL: Expected at least 2 users, found {len(rows)}")
        sys.exit(1)

    for username, score in expected_users:
        match = [r for r in rows if r[0] == username]
        if not match:
            print(f"FAIL: User '{username}' not found.")
            sys.exit(1)
        if abs(match[0][1] - score) > 0.001:
            print(f"FAIL: User '{username}' trust_score={match[0][1]}, expected {score}")
            sys.exit(1)

    print("OK: Seed data verified (alice=0.30, bob=0.30).")
    cur.close()
    conn.close()
    print("All checks passed.")


if __name__ == "__main__":
    verify()
