-- Ownership prediction outputs
CREATE TABLE IF NOT EXISTS ownership_predictions (
    date_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    predicted_mean REAL NOT NULL,
    predicted_std REAL,
    model_version TEXT,
    PRIMARY KEY (date_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_op_date ON ownership_predictions(date_id);
