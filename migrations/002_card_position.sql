-- Card position backfill table
CREATE TABLE IF NOT EXISTS fighter_card_position (
    date_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    card_position INTEGER NOT NULL,
    card_section TEXT NOT NULL,
    PRIMARY KEY (date_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_fcp_date ON fighter_card_position(date_id);
