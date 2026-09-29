-- Migration 011: meaning_reversal_detected on window_results
--
-- Audit trail for the meaning-reversal check in semantic_analysis: records
-- whether a window was forced to `incorrect` (or found to contain a reversal)
-- by that check, so it can be told apart from a genuine low-cosine
-- `incorrect`. app/models.py declares WindowResult.meaning_reversal_detected
-- and app/database.py's _EXPECTED_COLUMNS lists it, but SQLAlchemy's
-- create_all() only creates missing TABLES, never missing columns on a table
-- that already exists.
--
-- Deliberately NO `DEFAULT 0`: rows written before this column existed stay
-- NULL, meaning "predates the check / unknown". Backfilling 0 would falsely
-- assert those windows were checked and found clean.
--
-- SQLite has no ADD COLUMN IF NOT EXISTS. Re-running this file on an
-- already-migrated database raises "duplicate column name" — harmless but
-- noisy. The startup path in app/database.py (_add_missing_columns) applies
-- the same column idempotently; prefer it for an existing database.

ALTER TABLE window_results ADD COLUMN meaning_reversal_detected BOOLEAN;
