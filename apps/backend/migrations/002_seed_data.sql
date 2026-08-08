INSERT INTO users (username, trust_score)
VALUES
    ('alice', 1.0),
    ('bob', 1.0)
ON CONFLICT (username) DO NOTHING;
