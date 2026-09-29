-- Migration 009: advisory flag on warnings
--
-- A severe off-topic warning used to force a session's verdict to DISPUTE and
-- destroy a teacher's escrow. That veto is gone (see 84acc6b, 8e7a4b8); this
-- column instead records a severe warning as advisory — a record for a later
-- adjudicator to find, not a payout veto. app/models.py declares
-- Warning.advisory, and app/database.py's _EXPECTED_COLUMNS lists it, but
-- SQLAlchemy's create_all() only creates missing TABLES, never missing
-- columns on a table that already exists. Any database created before this
-- file therefore has no `advisory` column until it is added here or at
-- startup.
--
-- SQLite has no ADD COLUMN IF NOT EXISTS. Re-running this file on an
-- already-migrated database raises "duplicate column name" — harmless but
-- noisy. The startup path in app/database.py (_add_missing_columns) applies
-- the same column idempotently, so this file exists for parity with the
-- migration lineage; prefer the startup path for an existing database.

ALTER TABLE warnings ADD COLUMN advisory BOOLEAN DEFAULT 0;
