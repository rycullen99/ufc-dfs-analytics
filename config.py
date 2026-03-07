"""
Configuration constants for UFC DFS Backtesting System.
========================================================

WHY A SEPARATE CONFIG FILE?
    Instead of scattering magic numbers throughout the code, we centralize them
    here. This means if we discover that our slate size bins should change (say
    we get more 15-fight cards), we change ONE number here and every module
    that imports it picks up the change automatically.

WHAT'S IN HERE:
    1. Paths         - Where the database and output files live
    2. Thresholds    - Statistical minimums for trusting results
    3. Regimes       - How we slice contests (slate size, field size, fee)
    4. Fighter dims  - Weight class tiers, ownership bands, salary tiers
    5. Export tags    - Labels for the final skill output

PROGRAMMING CONCEPTS USED:
    - pathlib.Path   - Modern Python file path handling (replaces os.path)
    - Dict/List      - Type hints so you know what each constant holds
    - float("inf")   - Infinity, used as "everything above X" in bin edges
    - Lambda funcs   - Small inline functions (used in NHL, not needed here yet)
"""

from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================
# pathlib.Path is Python's modern way to handle file paths.
# Path.home() returns /Users/ryancullen on macOS.
# The "/" operator joins path segments — cleaner than os.path.join().

def _resolve_db_path() -> Path:
    """
    Try multiple known locations for the UFC results database.

    WHY: The DB might live in different spots depending on which machine
    or folder structure you're using. This tries each candidate in order
    and returns the first one that actually exists on disk.

    PATTERN: This is called a "fallback chain" — try option A, then B, etc.
    """
    candidates = [
        Path.home() / "Desktop" / "resultsdb_ufc.db",
        Path.home() / "Desktop" / "UFC DFS" / "resultsdb_ufc.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # If none exist, return the first as default (will error on use)
    return candidates[0]


# __file__ is a special Python variable = the path to THIS file (config.py)
# .resolve() turns it into an absolute path
# .parent goes up one directory (from config.py to ufc-dfs-analytics/)
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = _resolve_db_path()
OUTPUT_DIR = BASE_DIR / "outputs"


# =============================================================================
# STATISTICAL THRESHOLDS
# =============================================================================
# These control how much data we need before we trust a finding.
#
# Think of it like sample size in a poll: asking 5 people isn't reliable,
# asking 5,000 people is. Same idea with DFS lineups.

BOOTSTRAP_N = 10_000          # Number of resamples for confidence intervals
                               # (10K is standard — gives stable CI estimates)

CONFIDENCE_LEVEL = 0.95        # 95% confidence = "we'd see this result 95% of
                               # the time if we re-ran the analysis"

MIN_SAMPLES_PER_CELL = 100    # Minimum lineups in a bucket before we report it
                               # (below this, random noise dominates)

MIN_SAMPLES_REGIME = 5_000    # Minimum lineups for a regime to be "TIER 1"

MIN_CONTESTS_REGIME = 20      # Minimum contests for regime validity
                               # (avoids one wild card skewing everything)

FDR_ALPHA = 0.05              # False Discovery Rate threshold
                               # When testing many rules at once, some will
                               # appear significant by chance. FDR corrects for
                               # this. 0.05 = accept 5% false positive rate.


# =============================================================================
# CONTEST CLASSIFICATION
# =============================================================================
# This is the KEY hierarchy we discovered:
#
#   1. ALL 359 complete contests are 150-max GPPs (proven by duplication data)
#   2. Within those, Slate Size is the primary split
#   3. Then Field Size (causes ownership reversal at 50K boundary)
#   4. Then Entry Fee (adjusts magnitude of contrarian edge)
#
# The LIMITED ($555 Knockout) contests are a SEPARATE ecosystem with only
# partial lineup data — we track them but can't do full ROI analysis.

# --- Slate Size (PRIMARY dimension) ---
# Number of fights on the card. Determines how many unique lineup combinations
# exist, which drives whether chalk or contrarian strategies work better.
SLATE_SIZE_BINS = [0, 11, 13, float("inf")]
SLATE_SIZE_LABELS = ["SHORT", "STANDARD", "FULL"]
# SHORT (10-11):  Fewer combos → lineups converge → chalk works (175-200% own)
# STANDARD (12-13): Sweet spot → moderate contrarian (100-150% own)
# FULL (14-15):  Most combos → unique builds possible → contrarian edge (100-125%)

# --- Field Size (SECONDARY — causes ownership REVERSAL) ---
# How many total entries are in the contest.
# <50K and 50K+ play DIFFERENTLY for ownership strategy.
FIELD_SIZE_BINS = [0, 10_000, 30_000, 50_000, float("inf")]
FIELD_SIZE_LABELS = ["small", "medium", "large", "mega"]

# --- Entry Fee (TERTIARY — adjusts magnitude) ---
ENTRY_FEE_BINS = [0, 3, 8, 20, float("inf")]
ENTRY_FEE_LABELS = ["MICRO", "LOW", "MID", "HIGH"]
# MICRO ($1-3):  Largest fields, softest competition, moderate contrarian
# LOW ($5-8):    Flat — ownership matters less
# MID ($10-20):  Chalk-leaning (softer fields, favorites deliver)
# HIGH ($25-30): Sharpest fields, STRONGEST contrarian edge (100-125% = 2.59x)


# =============================================================================
# CONTEST NAME PATTERNS
# =============================================================================
# Since multi_entry_max is NULL for all 359 complete contests, we identify
# contest types by their names. These patterns were validated against the
# actual contest data.
#
# str.contains() in pandas uses regex by default, so we use the | operator
# to match ANY of these patterns.

CONTEST_150MAX_PATTERNS = [
    "150",        # "150 Max Entry", "150 Entry Max"
    "mini",       # "mini-MAX"
    "MEGA",       # "MEGA mini-MAX"
    "Micro",      # "Micro Special", "Micro-Special"
]

# These are the DK contest "brands" — each has a fee tier:
CONTEST_BRAND_MAP = {
    "Special":    "HIGH",     # $25-30, PPV numbered events (UFC 300, etc.)
    "Throwdown":  "MID",      # $15-25, flagship weekly GPP
    "Hook":       "LOW",      # $8
    "Arm Bar":    "LOW",      # $5
    "Haymaker":   "MICRO",    # $3
    "Jab":        "MICRO",    # $1
}


# =============================================================================
# WELL-SCRAPED CONTEST FILTER
# =============================================================================
# Not all scraped contests have complete payout data. We identified 106 of 359
# contests where: ABS(1.0 - SUM(payout * user_count) / total_prizes) < 0.05
#
# This means the total payouts we scraped match DK's advertised prize pool
# within 5%. For ROI calculations, we MUST use this filter — otherwise
# incomplete payout data inflates ROI above 1.0 (which is impossible given
# DK's ~13% rake).

PAYOUT_TOLERANCE = 0.05  # 5% tolerance for payout completeness check


# =============================================================================
# FIGHTER DIMENSIONS
# =============================================================================

# --- Weight Class Tiers (from backtested FPTS/$1K and dud rates) ---
# Tier determines expected value per salary dollar and bust probability.
WEIGHT_CLASS_TIERS = {
    "S": [                        # Best: 4.41 FPTS/$1K, 37.8% dud rate
        "Women's Strawweight",
        "Welterweight",
        "Women's Flyweight",
    ],
    "A": [                        # Strong: 4.14 FPTS/$1K, 39.3% dud rate
        "Women's Bantamweight",
        "Lightweight",
    ],
    "B": [                        # Neutral: 3.90 FPTS/$1K, 42.8% dud rate
        "Featherweight",
        "Light Heavyweight",
        "Middleweight",
    ],
    "C": [                        # Risky: 3.69 FPTS/$1K, 45.4% dud rate
        "Heavyweight",
    ],
    "F": [                        # Avoid stacking: 3.36 FPTS/$1K, 52.3% dud rate
        "Bantamweight",
        "Flyweight",
    ],
}

# Reverse lookup: weight class name → tier letter
# dict comprehension: for each tier/classes pair, for each class, map class→tier
WEIGHT_CLASS_TO_TIER = {
    wc: tier
    for tier, classes in WEIGHT_CLASS_TIERS.items()
    for wc in classes
}

# --- Salary Tiers ---
# Fighter salary ranges on DraftKings ($6,600 - $10,600)
SALARY_TIER_BINS = [0, 7000, 7500, 8000, 8500, 9000, 9500, float("inf")]
SALARY_TIER_LABELS = ["<$7K", "$7-7.5K", "$7.5-8K", "$8-8.5K", "$8.5-9K", "$9-9.5K", "$9.5K+"]

# --- Ownership Bands ---
OWNERSHIP_BINS = [0, 3, 8, 15, 25, 35, 50, float("inf")]
OWNERSHIP_LABELS = ["<3%", "3-8%", "8-15%", "15-25%", "25-35%", "35-50%", "50%+"]

# --- Total Lineup Ownership ---
TOTAL_OWNERSHIP_BINS = [0, 100, 125, 150, 175, 200, float("inf")]
TOTAL_OWNERSHIP_LABELS = ["<100%", "100-125%", "125-150%", "150-175%", "175-200%", "200%+"]

# --- Salary Remaining (from $50K cap) ---
SALARY_REMAINING_BINS = [-1, 0, 500, 1000, 2000, float("inf")]
SALARY_REMAINING_LABELS = ["$0 (maxed)", "$1-500", "$500-1K", "$1K-2K", "$2K+"]

# --- Salary Spread (max fighter salary - min fighter salary) ---
SALARY_SPREAD_BINS = [0, 500, 1000, 1500, 2000, float("inf")]
SALARY_SPREAD_LABELS = ["Very Tight (<$500)", "Tight ($500-1K)", "Moderate ($1-1.5K)",
                         "Wide ($1.5-2K)", "Stars & Scrubs (>$2K)"]

# --- Implied Probability Bands (from Odds API closing lines) ---
PROB_BAND_BINS = [0, 0.30, 0.45, 0.55, 0.70, 1.01]
PROB_BAND_LABELS = ["Heavy Dog", "Mod Dog", "Toss-up", "Mod Fav", "Heavy Fav"]

# --- Line Movement (open-to-close probability shift) ---
LINE_MOVEMENT_BINS = [-1.0, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 1.0]
LINE_MOVEMENT_LABELS = ["Huge Drift", "Med Drift", "Slight Drift",
                         "Stable", "Small Steam", "Med Steam", "Huge Steam"]

# --- Favorite Count per Lineup ---
FAVORITE_COUNT_BINS = [-1, 0, 1, 2, 3, 4, 5, 6]
FAVORITE_COUNT_LABELS = ["0", "1", "2", "3", "4", "5", "6"]


# =============================================================================
# ROI CALCULATION
# =============================================================================
# DraftKings takes ~13% rake (house edge). This means the AVERAGE player
# gets back ~$0.87 per $1 wagered. A "winning" strategy has ROI > ~0.87.
# ROI of 1.0 = break even. ROI > 1.0 = profit.
#
# CRITICAL: ROI must be weighted by user_count (number of entries per unique
# lineup). Simple averaging per unique lineup inflates ROI because it treats
# a lineup entered once the same as one entered 150 times.
#
# Formula: SUM(payout * user_count) / SUM(entry_cost * user_count)

DK_RAKE = 0.13  # DraftKings' approximate rake percentage
BREAKEVEN_ROI = 1.0 - DK_RAKE  # ~0.87 — the "house" baseline


# =============================================================================
# EXPORT SETTINGS
# =============================================================================

# Confidence tiers for the skill output
CONFIDENCE_TIERS = {
    "TIER_1": {"min_lineups": 100_000, "label": "High confidence"},
    "TIER_2": {"min_lineups": 10_000,  "label": "Strong recommendation"},
    "TIER_3": {"min_lineups": 1_000,   "label": "Directional only"},
}

# Tags for rules in the skill
RULE_TAGS = [
    "core",       # High confidence, always apply
    "leverage",   # Situational edge (certain regimes only)
    "fade",       # Anti-pattern to avoid
    "reversal",   # Rule that FLIPS between regimes
]
