-- Add teacher/learner role tracking to session contracts
ALTER TABLE session_contracts ADD COLUMN IF NOT EXISTS teacher_user_id INTEGER REFERENCES users(id);
ALTER TABLE session_contracts ADD COLUMN IF NOT EXISTS learner_user_id INTEGER REFERENCES users(id);
UPDATE session_contracts SET teacher_user_id = 1, learner_user_id = 2 WHERE teacher_user_id IS NULL;
ALTER TABLE session_contracts ALTER COLUMN teacher_user_id SET NOT NULL;
ALTER TABLE session_contracts ALTER COLUMN learner_user_id SET NOT NULL;
