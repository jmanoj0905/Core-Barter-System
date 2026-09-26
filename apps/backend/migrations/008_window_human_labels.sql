-- Migration 008: human ground-truth labels on window_results
--
-- app/models.py has declared human_label / labeled_by_user_id / labeled_at
-- since the feedback endpoint was added, and POST
-- /session/{id}/window/{n}/feedback writes all three — but no migration ever
-- created the columns, and SQLAlchemy's create_all() only creates missing
-- TABLES, never missing columns on a table that already exists. Any database
-- created before this file therefore fails that endpoint with
-- "no such column: window_results.human_label", which is why
-- ground_truth/export_real_labels.py had nothing to export.
--
-- SQLite has no ADD COLUMN IF NOT EXISTS. Re-running this file on an
-- already-migrated database raises "duplicate column name" — harmless but
-- noisy. The startup path in app/database.py applies the same three columns
-- idempotently, so this file exists for parity with the migration lineage;
-- prefer the startup path for existing databases.

ALTER TABLE window_results ADD COLUMN human_label VARCHAR(20);
ALTER TABLE window_results ADD COLUMN labeled_by_user_id INTEGER REFERENCES users(id);
ALTER TABLE window_results ADD COLUMN labeled_at DATETIME;

-- Exports filter on human_label IS NOT NULL and the labeled set stays small
-- relative to the table, so a partial index keeps that scan cheap.
CREATE INDEX IF NOT EXISTS idx_window_results_human_label
    ON window_results(human_label) WHERE human_label IS NOT NULL;
