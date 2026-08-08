-- Add teacher/learner role tracking to session contracts
-- SQLite: add columns with NOT NULL and default values
ALTER TABLE session_contracts ADD COLUMN teacher_user_id INTEGER REFERENCES users(id) DEFAULT 1;
ALTER TABLE session_contracts ADD COLUMN learner_user_id INTEGER REFERENCES users(id) DEFAULT 2;