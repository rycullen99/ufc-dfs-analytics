-- Sharp user signals
CREATE TABLE IF NOT EXISTS user_skill_scores (
    username TEXT PRIMARY KEY,
    n_entries INTEGER NOT NULL,
    n_cashes INTEGER NOT NULL,
    raw_rate REAL,
    posterior_mean REAL,
    skill_percentile REAL,
    is_sharp INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sharp_signals (
    date_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    field_exposure_pct REAL,
    sharp_exposure_pct REAL,
    sharp_vs_field_delta REAL,
    n_sharp_lineups INTEGER,
    sharp_confidence REAL,
    PRIMARY KEY (date_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_sharp_date ON sharp_signals(date_id);
CREATE INDEX IF NOT EXISTS idx_uss_sharp ON user_skill_scores(is_sharp);
