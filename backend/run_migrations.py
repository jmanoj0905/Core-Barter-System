"""Run SQL migration files in order against the database."""

import glob
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def run_migrations():
    db_url = os.environ["DATABASE_URL_SYNC"]
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()

    migration_dir = os.path.join(os.path.dirname(__file__), "migrations")
    sql_files = sorted(glob.glob(os.path.join(migration_dir, "*.sql")))

    for sql_file in sql_files:
        name = os.path.basename(sql_file)
        print(f"Running {name}...")
        with open(sql_file) as f:
            cur.execute(f.read())
        print(f"  Done.")

    cur.close()
    conn.close()
    print("All migrations completed.")


if __name__ == "__main__":
    run_migrations()
