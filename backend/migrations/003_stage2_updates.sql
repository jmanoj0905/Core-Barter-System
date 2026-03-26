-- Confirmations table for tracking per-user session completions
CREATE TABLE IF NOT EXISTS confirmations (
    id SERIAL PRIMARY KEY,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    confirmed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(barter_session_id, user_id)
);

-- Add drift_summary JSON storage to verdicts
ALTER TABLE verdicts ADD COLUMN IF NOT EXISTS drift_summary TEXT;

-- Add window_ids to warnings (comma-separated window IDs that triggered warning)
ALTER TABLE warnings ADD COLUMN IF NOT EXISTS window_ids TEXT;
