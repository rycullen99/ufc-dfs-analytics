-- Supabase dimension tables
CREATE TABLE IF NOT EXISTS fighter_dim (
    canonical_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    weight_class TEXT
);

CREATE TABLE IF NOT EXISTS event_dim (
    date_id INTEGER PRIMARY KEY,
    event_date DATE,
    n_contests INTEGER
);

CREATE TABLE IF NOT EXISTS contest_dim (
    contest_id INTEGER PRIMARY KEY,
    date_id INTEGER REFERENCES event_dim(date_id),
    contest_name TEXT,
    entry_cost REAL,
    contest_size INTEGER,
    multi_entry_max INTEGER,
    cash_line REAL
);

CREATE TABLE IF NOT EXISTS source_dim (
    source_id SERIAL PRIMARY KEY,
    source_name TEXT NOT NULL UNIQUE,
    description TEXT
);

INSERT INTO source_dim (source_name, description) VALUES
    ('dfs', 'DraftKings DFS platform'),
    ('odds', 'BestFightOdds closing lines'),
    ('model', 'Pipeline model predictions')
ON CONFLICT DO NOTHING;
