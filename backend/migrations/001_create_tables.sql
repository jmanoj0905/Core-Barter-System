CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    trust_score FLOAT NOT NULL DEFAULT 0.30,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS barter_sessions (
    id SERIAL PRIMARY KEY,
    user1_id INTEGER NOT NULL REFERENCES users(id),
    user2_id INTEGER NOT NULL REFERENCES users(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_contracts (
    id SERIAL PRIMARY KEY,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    topic VARCHAR(200) NOT NULL,
    scope TEXT,
    allowed_concepts TEXT,
    disallowed_concepts TEXT,
    agreed_duration_seconds INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS window_results (
    id SERIAL PRIMARY KEY,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    window_number INTEGER NOT NULL,
    classification VARCHAR(20) NOT NULL,
    cosine_similarity FLOAT NOT NULL,
    text_content TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS warnings (
    id SERIAL PRIMARY KEY,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS verdicts (
    id SERIAL PRIMARY KEY,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id) UNIQUE,
    verdict_type VARCHAR(20) NOT NULL,
    on_topic_percentage FLOAT NOT NULL,
    warning_count INTEGER NOT NULL,
    duration_check VARCHAR(10) NOT NULL,
    confirmation_check VARCHAR(10) NOT NULL,
    trust_delta_user1 FLOAT NOT NULL,
    trust_delta_user2 FLOAT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
