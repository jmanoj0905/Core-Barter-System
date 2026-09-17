-- Migration 007: Video Engagement
-- Adds video_engagement_results (raw per-window video-only scores) and
-- engagement_score_log (fused speech+video engagement history)

CREATE TABLE IF NOT EXISTS video_engagement_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    window_start FLOAT NOT NULL,
    window_end FLOAT NOT NULL,
    video_attention_score FLOAT NOT NULL,
    backend_used TEXT NOT NULL,
    raw_signals TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_video_engagement_barter ON video_engagement_results(barter_session_id);

CREATE TABLE IF NOT EXISTS engagement_score_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    user_id INTEGER REFERENCES users(id),
    speech_engagement_score FLOAT,
    video_attention_score FLOAT,
    fused_engagement_score FLOAT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_engagement_log_barter ON engagement_score_log(barter_session_id);
