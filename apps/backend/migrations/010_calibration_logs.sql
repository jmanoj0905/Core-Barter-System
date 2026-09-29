-- Migration 010: Gaze calibration audit log
-- Adds calibration_logs (one row per calibration attempt: outcome plus
-- old/new per-user baseline ratios)

CREATE TABLE IF NOT EXISTS calibration_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    outcome TEXT NOT NULL,
    old_baseline_head_ratio FLOAT,
    old_baseline_gaze_ratio FLOAT,
    new_baseline_head_ratio FLOAT,
    new_baseline_gaze_ratio FLOAT,
    recalibration_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_calibration_logs_barter ON calibration_logs(barter_session_id);
