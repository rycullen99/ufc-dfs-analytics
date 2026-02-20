-- Canonical fighter identity tables
CREATE TABLE IF NOT EXISTS canonical_fighter (
    canonical_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    weight_class TEXT
);

CREATE TABLE IF NOT EXISTS fighter_alias (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id INTEGER NOT NULL REFERENCES canonical_fighter(canonical_id),
    alias_name TEXT NOT NULL,
    alias_source TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    UNIQUE(alias_name, alias_source)
);

CREATE INDEX IF NOT EXISTS idx_fighter_alias_name ON fighter_alias(alias_name);
CREATE INDEX IF NOT EXISTS idx_fighter_alias_canonical ON fighter_alias(canonical_id);
