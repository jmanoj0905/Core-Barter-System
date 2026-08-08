-- Confirmations table for tracking per-user session completions
CREATE TABLE IF NOT EXISTS confirmations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(barter_session_id, user_id)
);

-- Add drift_summary JSON storage to verdicts
-- SQLite: Check if column exists before adding
ALTER TABLE verdicts ADD COLUMN drift_summary TEXT;

-- Add window_ids to warnings (comma-separated window IDs that triggered warning)
ALTER TABLE warnings ADD COLUMN window_ids TEXT;