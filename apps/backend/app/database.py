from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base

engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


if engine.dialect.name == "sqlite":

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, connection_record):
        # SQLAlchemy foreign keys alone don't enforce anything in SQLite —
        # each connection must turn it on explicitly (ISSUE-015).
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Columns added to an existing table after it was first created. create_all()
# below creates missing TABLES only — it never alters a table that already
# exists — so a database built before a column was declared in models.py keeps
# working right up until the first query touches that column. The human-label
# columns hit exactly that: the feedback endpoint wrote them for weeks against
# databases that had no such columns. Each entry is (table, column, DDL type).
_EXPECTED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("window_results", "human_label", "VARCHAR(20)"),
    ("window_results", "labeled_by_user_id", "INTEGER REFERENCES users(id)"),
    ("window_results", "labeled_at", "DATETIME"),
    ("warnings", "advisory", "BOOLEAN DEFAULT 0"),
)


def _add_missing_columns(connection) -> list[str]:
    """Add any declared-but-absent columns. Idempotent; safe on every startup.

    SQLite has no ADD COLUMN IF NOT EXISTS, so each column is checked against
    PRAGMA table_info first. Only additive, nullable columns belong here — this
    is not a migration framework and must never drop, rename, or retype
    anything. Tables that do not exist yet are skipped; create_all() makes them
    with the full column set already in place.
    """
    from sqlalchemy import text

    added = []
    for table, column, ddl_type in _EXPECTED_COLUMNS:
        exists = connection.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:t"),
            {"t": table},
        ).first()
        if not exists:
            continue
        present = {row[1] for row in connection.execute(text(f"PRAGMA table_info({table})"))}
        if column in present:
            continue
        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
        added.append(f"{table}.{column}")
    return added


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if engine.dialect.name == "sqlite":
            added = await conn.run_sync(_add_missing_columns)
            if added:
                print(f"[db] added missing columns: {', '.join(added)}")


async def get_db():
    async with async_session() as session:
        yield session
