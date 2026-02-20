-- Backtest result storage
CREATE TABLE IF NOT EXISTS backtest_results (
    backtest_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    run_date TEXT NOT NULL,
    n_folds INTEGER,
    mean_mae REAL,
    mean_rmse REAL,
    holdout_mae REAL,
    holdout_rmse REAL,
    ci_lower REAL,
    ci_upper REAL,
    config_json TEXT
);
