"""Run SQL migration files in order against the database."""

import glob
import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()


def run_migrations():
    # Convert async URL to sync sqlite URL
    db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:////data/barter.db")
    # sqlite+aiosqlite:////data/barter.db -> /data/barter.db
    db_path = db_url.replace("sqlite+aiosqlite://", "").replace("sqlite://", "")
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    cur = conn.cursor()

    migration_dir = os.path.join(os.path.dirname(__file__), "migrations")
    sql_files = sorted(glob.glob(os.path.join(migration_dir, "*.sql")))

    for sql_file in sql_files:
        name = os.path.basename(sql_file)
        print(f"Running {name}...")
        with open(sql_file) as f:
            cur.executescript(f.read())
        print(f"  Done.")

    cur.close()
    conn.close()
    print("All migrations completed.")


if __name__ == "__main__":
    run_migrations()