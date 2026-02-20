-- Per-fighter odds quality tiers
CREATE TABLE IF NOT EXISTS fighter_odds_quality (
    date_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    odds_quality_tier TEXT NOT NULL,
    PRIMARY KEY (date_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_foq_tier ON fighter_odds_quality(odds_quality_tier);
