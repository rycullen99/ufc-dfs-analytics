-- Supabase fact tables
CREATE TABLE IF NOT EXISTS fighter_event_snapshot (
    date_id INTEGER REFERENCES event_dim(date_id),
    canonical_id INTEGER REFERENCES fighter_dim(canonical_id),
    salary INTEGER,
    card_position INTEGER,
    card_section TEXT,
    ownership_actual REAL,
    actual_points REAL,
    implied_prob REAL,
    odds_quality_tier TEXT,
    PRIMARY KEY (date_id, canonical_id)
);

CREATE TABLE IF NOT EXISTS ownership_predictions (
    date_id INTEGER REFERENCES event_dim(date_id),
    canonical_id INTEGER REFERENCES fighter_dim(canonical_id),
    predicted_mean REAL,
    predicted_std REAL,
    model_version TEXT,
    PRIMARY KEY (date_id, canonical_id)
);

CREATE TABLE IF NOT EXISTS sharp_signals (
    date_id INTEGER REFERENCES event_dim(date_id),
    canonical_id INTEGER REFERENCES fighter_dim(canonical_id),
    field_exposure_pct REAL,
    sharp_exposure_pct REAL,
    sharp_vs_field_delta REAL,
    sharp_confidence REAL,
    PRIMARY KEY (date_id, canonical_id)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    backtest_id SERIAL PRIMARY KEY,
    model_name TEXT,
    run_date TIMESTAMP DEFAULT NOW(),
    n_folds INTEGER,
    mean_mae REAL,
    holdout_mae REAL,
    ci_lower REAL,
    ci_upper REAL,
    config JSONB
);
