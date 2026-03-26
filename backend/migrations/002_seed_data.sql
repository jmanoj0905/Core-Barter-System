INSERT INTO users (username, trust_score)
VALUES
    ('alice', 0.30),
    ('bob', 0.30)
ON CONFLICT (username) DO NOTHING;
