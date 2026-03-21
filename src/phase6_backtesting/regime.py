"""
Contest regime classification for ROI backtesting.
Extends src/config.py with backtesting-specific bins and contest taxonomy.

KEY FINDING: All 359 complete contests are 150-max GPPs (NULL multi_entry_max).
The $555 Knockout series (LIMITED) has explicit multi_entry_max but incomplete data.
Primary split hierarchy: Slate Size → Field Size → Entry Fee
"""

import re

import pandas as pd
import numpy as np

# Slate size: number of fights determines available lineup combos
SLATE_BINS   = [0, 11, 13, float("inf")]
SLATE_LABELS = ["SHORT", "STANDARD", "FULL"]
# SHORT (≤11): fewer combos → lineups converge → chalk works (175-200% ownership)
# STANDARD (12-13): moderate contrarian sweet spot (100-150%)
# FULL (14+): most combos → contrarian edge (100-125%)

# Field size: causes ownership strategy REVERSAL at 50K boundary
FIELD_BINS   = [0, 10_000, 30_000, 50_000, float("inf")]
FIELD_LABELS = ["small", "medium", "large", "mega"]

# Entry fee: adjusts magnitude of contrarian edge within 150-max contests
FEE_BINS   = [0, 3, 8, 20, float("inf")]
FEE_LABELS = ["MICRO", "LOW", "MID", "HIGH"]
# HIGH ($25-30): sharpest fields, strongest contrarian edge (100-125% = 2.59x ROI)
# MID ($10-20): chalk-leaning (softer fields, favorites deliver, 200%+ = 1.90x)
# LOW ($5-8): flat — ownership matters less
# MICRO ($1-3): largest fields, moderate contrarian

# Payout completeness threshold for unbiased ROI
# Exclude contests where scraped payouts differ >6% from advertised prize pool
# Widened from 5% after lineup resolution pass adjusted some user_counts
PAYOUT_TOLERANCE = 0.06

# Total ownership bands (sum of 6 fighter ownership percentages)
TOTAL_OWN_BINS   = [0, 100, 125, 150, 175, 200, float("inf")]
TOTAL_OWN_LABELS = ["<100%", "100-125%", "125-150%", "150-175%", "175-200%", "200%+"]

# Salary remaining from $50K cap
SAL_REM_BINS   = [-1, 0, 500, 1000, 2000, float("inf")]
SAL_REM_LABELS = ["$0 (maxed)", "$1-500", "$500-1K", "$1K-2K", "$2K+"]

# Favorite count per lineup (fighters with >50% implied probability)
# right=True on these bins: (-1,0]=0, (0,1]=1, ... (5,6]=6
FAV_COUNT_BINS   = [-1, 0, 1, 2, 3, 4, 5, 6]
FAV_COUNT_LABELS = ["0", "1", "2", "3", "4", "5", "6"]

# Implied probability sum across all 6 fighters
PROB_SUM_BINS   = [0, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, float("inf")]
PROB_SUM_LABELS = ["<2.5", "2.5-2.75", "2.75-3.0", "3.0-3.25", "3.25-3.5", "3.5-3.75", "3.75+"]

# own_prob_ratio: avg(fighter_ownership / implied_win_prob)
# <0.50 = fighters owned well below market expectation → value
OWN_PROB_BINS   = [0, 0.50, 0.75, 1.0, 1.25, float("inf")]
OWN_PROB_LABELS = ["<0.50", "0.50-0.75", "0.75-1.0", "1.0-1.25", "1.25+"]

# Duplication: how many contests the same lineup was entered across
DUPE_BINS   = [0, 1, 5, 20, 100, float("inf")]
DUPE_LABELS = ["1 (unique)", "2-5", "6-20", "21-100", "101+"]

# Fight count bands (granular — replaces 3-bucket slate_size for rule derivation)
# 10-11 merged (7 dates, TIER 2), 15+ merged (7 dates, TIER 2)
FIGHT_COUNT_BINS   = [0, 11, 12, 13, 14, float("inf")]
FIGHT_COUNT_LABELS = ["10-11", "12", "13", "14", "15+"]

# Contest families: recurring DK contest brands with distinct characteristics
# Regex patterns match contest_name from the contests table
CONTEST_FAMILY_PATTERNS = [
    (re.compile(r"mini-MAX", re.IGNORECASE), "mini-MAX"),
    (re.compile(r"Throwdown", re.IGNORECASE), "Throwdown"),
    (re.compile(r"Arm Bar", re.IGNORECASE), "Arm Bar"),
    (re.compile(r"Hook(?!\s*&)", re.IGNORECASE), "Hook"),
    (re.compile(r"Haymaker", re.IGNORECASE), "Haymaker"),
    (re.compile(r"Special", re.IGNORECASE), "Special"),
]


def add_regime_columns(df: pd.DataFrame, salary_col: str = "total_salary") -> pd.DataFrame:
    """
    Add derived regime columns to a lineups or contests DataFrame.
    Requires columns: fight_count, contest_size, entry_cost, total_ownership,
                      total_salary, favorite_count, implied_prob_sum, own_prob_ratio,
                      lineup_count.
    """
    df = df.copy()

    if "fight_count" in df.columns:
        df["slate_size"] = pd.cut(df["fight_count"], bins=SLATE_BINS, labels=SLATE_LABELS, right=True)
        df["fight_count_band"] = pd.cut(df["fight_count"], bins=FIGHT_COUNT_BINS,
                                        labels=FIGHT_COUNT_LABELS, right=True)

    if "multi_entry_max" in df.columns:
        df["contest_archetype"] = df["multi_entry_max"].apply(_classify_archetype)

    if "contest_size" in df.columns:
        df["field_size"] = pd.cut(df["contest_size"], bins=FIELD_BINS, labels=FIELD_LABELS, right=True)

    if "entry_cost" in df.columns:
        df["fee_tier"] = pd.cut(df["entry_cost"], bins=FEE_BINS, labels=FEE_LABELS, right=True)

    if "total_ownership" in df.columns:
        df["total_own_band"] = pd.cut(
            df["total_ownership"], bins=TOTAL_OWN_BINS, labels=TOTAL_OWN_LABELS, right=True
        )

    if salary_col in df.columns:
        df["salary_remaining"] = 50_000 - df[salary_col]
        df["salary_remaining_band"] = pd.cut(
            df["salary_remaining"], bins=SAL_REM_BINS, labels=SAL_REM_LABELS, right=True
        )

    if "favorite_count" in df.columns:
        df["fav_count_band"] = pd.cut(
            df["favorite_count"], bins=FAV_COUNT_BINS, labels=FAV_COUNT_LABELS, right=True
        )

    if "implied_prob_sum" in df.columns:
        df["prob_sum_band"] = pd.cut(
            df["implied_prob_sum"], bins=PROB_SUM_BINS, labels=PROB_SUM_LABELS, right=True
        )

    if "own_prob_ratio" in df.columns:
        df["own_prob_band"] = pd.cut(
            df["own_prob_ratio"], bins=OWN_PROB_BINS, labels=OWN_PROB_LABELS, right=True
        )

    if "lineup_count" in df.columns:
        df["dupe_band"] = pd.cut(
            df["lineup_count"], bins=DUPE_BINS, labels=DUPE_LABELS, right=True
        )

    if "contest_name" in df.columns:
        df["contest_family"] = df["contest_name"].apply(_classify_contest_family)

    return df


def _classify_archetype(max_entries) -> str:
    """Classify contest by max entry count."""
    if pd.isna(max_entries):
        return "UNLIMITED"
    max_entries = int(max_entries)
    if max_entries == 1:
        return "SE"
    if max_entries <= 20:
        return "LIMITED"
    if max_entries <= 150:
        return "150-Max"
    return "UNLIMITED"


def _classify_contest_family(name: str) -> str:
    """Classify a contest name into its family using regex patterns."""
    if not isinstance(name, str):
        return "Other"
    for pattern, family in CONTEST_FAMILY_PATTERNS:
        if pattern.search(name):
            return family
    return "Other"
