-- Migration 006: Escrow System
-- Adds wallet, escrow, and credit_transaction tables

-- Wallets table: tracks credit balance per user
CREATE TABLE IF NOT EXISTS wallets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id),
    available_balance INTEGER NOT NULL DEFAULT 999999,
    locked_balance INTEGER NOT NULL DEFAULT 0,
    total_earned INTEGER NOT NULL DEFAULT 0,
    total_spent INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_wallets_user_id ON wallets(user_id);

-- Escrows table: tracks locked credits per session
CREATE TABLE IF NOT EXISTS escrows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barter_session_id INTEGER NOT NULL REFERENCES barter_sessions(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    amount INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'locked',
    locked_at TEXT DEFAULT (datetime('now')),
    released_at TEXT,
    release_type VARCHAR(20)
);

CREATE INDEX IF NOT EXISTS idx_escrows_session ON escrows(barter_session_id);
CREATE INDEX IF NOT EXISTS idx_escrows_user ON escrows(user_id);
CREATE INDEX IF NOT EXISTS idx_escrows_status ON escrows(status);

-- Credit transactions: immutable ledger of all credit movements
CREATE TABLE IF NOT EXISTS credit_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    barter_session_id INTEGER REFERENCES barter_sessions(id),
    transaction_type VARCHAR(30) NOT NULL,
    amount INTEGER NOT NULL,
    balance_after INTEGER NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_credit_transactions_user ON credit_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_credit_transactions_session ON credit_transactions(barter_session_id);
CREATE INDEX IF NOT EXISTS idx_credit_transactions_type ON credit_transactions(transaction_type);

-- Create wallets for existing users with initial 999999 credits
INSERT OR IGNORE INTO wallets (user_id, available_balance, locked_balance, total_earned, total_spent)
SELECT id, 999999, 0, 0, 0
FROM users
WHERE id NOT IN (SELECT user_id FROM wallets);