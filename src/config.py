"""Central configuration for the UFC DFS analytics pipeline."""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
DB_PATH = Path("/Users/ryancullen/Desktop/DFS/Databases/resultsdb_ufc.db")
CSV_ODDS_PATH = Path("/Users/ryancullen/projects/fightodds-scraper/ufc_all_odds_2020_plus.csv")
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
FREEROLL_CONTEST_ID = 142690805
SALARY_CAP = 50_000
FIGHTERS_PER_LINEUP = 6

# ─── Walk-Forward CV ─────────────────────────────────────────────────────────
MIN_TRAINING_EVENTS = 20
HOLDOUT_EVENTS = 10

# ─── Salary tiers ────────────────────────────────────────────────────────────
SALARY_TIER_BINS = [0, 7000, 7500, 8000, 8500, 9000, 15000]
SALARY_TIER_LABELS = ["tier_1", "tier_2", "tier_3", "tier_4", "tier_5", "tier_6"]

# ─── Card position labels ────────────────────────────────────────────────────
CARD_SECTIONS = {
    1: "main_event",
    2: "co_main",
    3: "main_card",
    4: "main_card",
    5: "main_card",
    6: "prelim",
    7: "prelim",
}

# ─── Leakage blocklist ───────────────────────────────────────────────────────
LEAKY_FEATURES = frozenset({
    "ownership",
    "actual_points",
    "actual_fpts",
    "is_cashing",
    "payout",
    "lineup_rank",
    "points_percentile",
    "was_optimal",
    "cashed",
})

# ─── Known fighter aliases (odds → DFS canonical) ────────────────────────────
KNOWN_ALIASES = {
    "nina nunes": "nina ansaroff",
    "brogan walker sanchez": "brogan walker",
    "marcos rogerio": "marcos rogerio de lima",
    "elizeu zaleski": "elizeu zaleski dos santos",
    "ariane da silva": "ariane lipski da silva",
    "ariane lipski": "ariane lipski da silva",
    "serghei spivac": "sergey spivak",
    "silvana gomez juarez": "silvana juarez",
}
