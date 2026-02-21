"""
UFC DFS Fighter Rules by Archetype Analysis

Tests whether fighter-level construction rules interact with contest archetype.
Key question: Do weight class tiers, line movement signals, and gender rules
change depending on contest type? Or are they universal?

Sections:
  1. Weight Class x Archetype (fighter-level)
  2. Gender x Archetype (fighter-level)
  3. Line Movement x Archetype (fighter-level)
  4. Probability Band x Archetype (fighter-level)
  5. Card Position x Archetype (fighter-level)
  6. Weight Class Composition x Archetype (lineup-level, batched)
  7. Field Size x Line Movement Interaction (UNLIMITED only)
  8. Effective Entries x Ownership Strategy
"""

import sqlite3
import numpy as np
import pandas as pd
import logging
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

DB_PATH = "/Users/ryancullen/Desktop/resultsdb_ufc.db"
OUTPUT_DIR = Path("/Users/ryancullen/ufc-dfs-analytics/data/analysis")
FREEROLL_ID = 142690805

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
SEPARATOR = "=" * 70
SUBSEP = "-" * 50

# ---------------------------------------------------------------------------
# Weight class tiers from prior analysis
# ---------------------------------------------------------------------------

WEIGHT_CLASS_TIERS = {
    "Women's Strawweight": "S", "W-Strawweight": "S",
    "Women's Flyweight": "S", "W-Flyweight": "S",
    "Featherweight": "S",
    "Welterweight": "A", "Heavyweight": "A", "Middleweight": "A",
    "Lightweight": "B", "Light Heavyweight": "B",
    "Bantamweight": "F", "Flyweight": "F",
}

WOMENS_WEIGHT_CLASSES = {
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
    "W-Strawweight", "W-Flyweight", "W-Bantamweight", "W-Featherweight",
}

PROB_BANDS = [
    ("Heavy Fav (70%+)", 0.70, 1.01),
    ("Mod Fav (55-70%)", 0.55, 0.70),
    ("Toss-up (45-55%)", 0.45, 0.55),
    ("Mod Dog (30-45%)", 0.30, 0.45),
    ("Heavy Dog (<30%)", 0.00, 0.30),
]

CARD_SECTION_ORDER = {
    "main_event": 0, "co_main": 1, "main_card": 2,
    "prelim": 3, "early_prelim": 4,
}

CARD_SECTION_DISPLAY = {
    "main_event": "Main Event",
    "co_main": "Co-Main",
    "main_card": "Main Card",
    "prelim": "Prelim",
    "early_prelim": "Early Prelim",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_contest(multi_entry_max):
    if multi_entry_max == 1:
        return "SE"
    elif multi_entry_max is not None and 3 <= multi_entry_max <= 18:
        return "LIMITED"
    elif multi_entry_max == 20:
        return "20-Max"
    elif multi_entry_max == 150:
        return "150-Max"
    else:
        return "UNLIMITED"


def field_size_bucket(contest_size):
    if contest_size < 5000:
        return "Small (<5K)"
    elif contest_size < 20000:
        return "Medium (5K-20K)"
    elif contest_size < 50000:
        return "Large (20K-50K)"
    else:
        return "Mega (50K+)"


def assign_prob_band(prob):
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return None
    for label, lo, hi in PROB_BANDS:
        if lo <= prob < hi:
            return label
    if prob >= 0.70:
        return "Heavy Fav (70%+)"
    return None


def assign_line_move_dir(line_move):
    if line_move is None or (isinstance(line_move, float) and np.isnan(line_move)):
        return "unknown"
    if line_move > 0.02:
        return "steamed"
    elif line_move < -0.02:
        return "drifted"
    else:
        return "stable"


def get_tier(wc):
    return WEIGHT_CLASS_TIERS.get(wc, "?") if wc else "?"


def is_womens(wc):
    return wc in WOMENS_WEIGHT_CLASSES if wc else False


def sample_flag(n, threshold=50):
    return " [SMALL SAMPLE]" if n < threshold else ""


def section_header(title):
    log.info(f"\n{SEPARATOR}")
    log.info(f"  {title}")
    log.info(SEPARATOR)


def subsection_header(title):
    log.info(f"\n{SUBSEP}")
    log.info(f"  {title}")
    log.info(SUBSEP)


def convert_for_json(obj):
    """Convert numpy/pandas types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_contests(conn):
    """Load contests with archetype classification."""
    df = pd.read_sql_query(f"""
        SELECT contest_id, date_id, entry_cost, contest_size,
               multi_entry_max, cash_line, total_prizes
        FROM contests
        WHERE contest_id != {FREEROLL_ID}
    """, conn)

    df["archetype"] = df["multi_entry_max"].apply(classify_contest)
    df["field_bucket"] = df["contest_size"].apply(field_size_bucket)

    log.info(f"Loaded {len(df)} contests (excl freeroll)")
    log.info(f"Archetype distribution:\n{df['archetype'].value_counts().to_string()}\n")
    return df


def load_fighters(conn, contests_df):
    """
    Load fighter-level data: players joined with player_features (for weight_class)
    and contests (for date_id and archetype).

    Deduplicates by (date_id, full_name) to avoid counting the same fighter
    multiple times across contests on the same date. Picks the row from the
    largest contest (most representative ownership).
    """
    log.info("Loading fighter data (players + player_features + contests)...")

    df = pd.read_sql_query(f"""
        SELECT
            p.player_id,
            p.contest_id,
            p.full_name,
            p.salary,
            p.ownership,
            p.actual_points,
            pf.weight_class,
            c.date_id,
            c.multi_entry_max,
            c.contest_size,
            c.entry_cost
        FROM players p
        JOIN contests c ON p.contest_id = c.contest_id
        LEFT JOIN player_features pf ON p.player_id = pf.player_id
            AND p.contest_id = pf.contest_id
        WHERE c.contest_id != {FREEROLL_ID}
    """, conn)

    df["archetype"] = df["multi_entry_max"].apply(classify_contest)

    # Deduplicate: for each (date_id, full_name), keep the row from the largest contest
    df = df.sort_values("contest_size", ascending=False)
    df = df.drop_duplicates(subset=["date_id", "full_name"], keep="first")

    # Derived columns
    df["salary_value"] = np.where(
        df["salary"] > 0,
        df["actual_points"] / (df["salary"] / 1000),
        0,
    )
    df["is_ceiling"] = df["salary_value"] >= 8.0
    df["is_dud"] = df["actual_points"] == 0
    df["tier"] = df["weight_class"].apply(get_tier)
    df["is_womens"] = df["weight_class"].apply(is_womens)
    df["gender"] = np.where(df["is_womens"], "Women", "Men")

    log.info(f"Loaded {len(df):,} fighter-events (deduplicated by date_id + full_name)")
    log.info(f"Weight class coverage: {df['weight_class'].notna().sum():,} / {len(df):,}")
    return df


def load_odds_fighters(conn, contests_df):
    """
    Load fighter-level data with odds from fighter_odds_enriched.
    Deduplicates by (date_id, dfs_name).
    """
    log.info("Loading odds-enriched fighter data...")

    df = pd.read_sql_query(f"""
        SELECT
            foe.date_id,
            foe.canonical_id,
            foe.player_id,
            foe.dfs_name,
            foe.salary,
            foe.actual_ownership,
            foe.actual_points,
            foe.open_prob,
            foe.close_prob,
            foe.line_move,
            foe.is_favorite,
            foe.close_n_books,
            fcp.card_section,
            pf.weight_class
        FROM fighter_odds_enriched foe
        LEFT JOIN fighter_card_position fcp
            ON foe.date_id = fcp.date_id AND foe.player_id = fcp.player_id
        LEFT JOIN player_features pf
            ON foe.player_id = pf.player_id
            AND pf.contest_id IN (
                SELECT contest_id FROM contests
                WHERE date_id = foe.date_id AND contest_id != {FREEROLL_ID}
                LIMIT 1
            )
        WHERE foe.player_id IS NOT NULL
          AND foe.actual_points IS NOT NULL
    """, conn)

    # Add archetype: pick the largest contest on each date for archetype assignment
    # Actually for odds analysis, we need per-archetype splits so we expand later
    df["salary_value"] = np.where(
        df["salary"] > 0,
        df["actual_points"] / (df["salary"] / 1000),
        0,
    )
    df["is_ceiling"] = df["salary_value"] >= 8.0
    df["is_dud"] = df["actual_points"] == 0
    df["line_move_dir"] = df["line_move"].apply(assign_line_move_dir)
    df["prob_band"] = df["close_prob"].apply(assign_prob_band)
    df["tier"] = df["weight_class"].apply(get_tier)
    df["is_womens"] = df["weight_class"].apply(is_womens)
    df["gender"] = np.where(df["is_womens"], "Women", "Men")

    log.info(f"Loaded {len(df):,} odds-enriched fighter-events")
    log.info(f"Dates with odds: {df['date_id'].nunique()}")
    return df


def expand_fighters_by_archetype(fighters_df, contests_df):
    """
    For each fighter-event, create one row per archetype present on that date.
    This lets us compute fighter stats per archetype.
    """
    # Get distinct (date_id, archetype) pairs
    date_archetypes = contests_df[["date_id", "archetype"]].drop_duplicates()

    # Merge: each fighter on a date gets one row per archetype on that date
    expanded = fighters_df.merge(date_archetypes, on="date_id", how="inner", suffixes=("_orig", ""))

    # If fighters_df already has an 'archetype' column, drop the original
    if "archetype_orig" in expanded.columns:
        expanded = expanded.drop(columns=["archetype_orig"])

    return expanded


# ---------------------------------------------------------------------------
# Section 1: Weight Class x Archetype
# ---------------------------------------------------------------------------

def analyze_weight_class_by_archetype(fighters_df, contests_df):
    section_header("1. WEIGHT CLASS x ARCHETYPE (Fighter-Level)")
    log.info("Does tier ranking change by archetype?")
    log.info("Deduped by (date_id, full_name). Uses largest-contest row.\n")

    expanded = expand_fighters_by_archetype(fighters_df, contests_df)
    wc_data = expanded[expanded["weight_class"].notna() & (expanded["weight_class"] != "Unknown")]

    results = {}
    archetypes = ["UNLIMITED", "LIMITED", "SE", "20-Max", "150-Max"]

    for arch in archetypes:
        arch_data = wc_data[wc_data["archetype"] == arch]
        if arch_data.empty:
            continue

        agg = arch_data.groupby("weight_class").agg(
            n=("actual_points", "count"),
            avg_fpts=("actual_points", "mean"),
            avg_value=("salary_value", "mean"),
            ceiling_pct=("is_ceiling", "mean"),
            dud_pct=("is_dud", "mean"),
            avg_salary=("salary", "mean"),
        ).reset_index()

        agg["tier"] = agg["weight_class"].apply(get_tier)
        agg["ceiling_pct"] = (agg["ceiling_pct"] * 100).round(1)
        agg["dud_pct"] = (agg["dud_pct"] * 100).round(1)
        agg = agg.sort_values("avg_value", ascending=False)

        subsection_header(f"Weight Class — {arch} (n={len(arch_data):,})")
        log.info(f"  {'Weight Class':<22s} {'Tier':>4s} {'N':>6s} {'FPTS':>7s} {'Val':>6s} "
                 f"{'Ceil%':>6s} {'Dud%':>6s} {'AvgSal':>7s}")
        log.info(f"  {'-'*22} {'-'*4} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
        for _, row in agg.iterrows():
            flag = sample_flag(row["n"])
            log.info(f"  {row['weight_class']:<22s} {row['tier']:>4s} {row['n']:>6.0f} "
                     f"{row['avg_fpts']:>7.1f} {row['avg_value']:>6.2f} "
                     f"{row['ceiling_pct']:>6.1f} {row['dud_pct']:>6.1f} "
                     f"{row['avg_salary']:>7.0f}{flag}")

        results[arch] = {
            row["weight_class"]: {
                "tier": row["tier"],
                "n": int(row["n"]),
                "avg_fpts": round(row["avg_fpts"], 2),
                "avg_value": round(row["avg_value"], 3),
                "ceiling_pct": round(row["ceiling_pct"], 1),
                "dud_pct": round(row["dud_pct"], 1),
                "small_sample": row["n"] < 50,
            }
            for _, row in agg.iterrows()
        }

    # Summary: tier rankings by archetype
    subsection_header("Tier Ranking Comparison Across Archetypes")
    log.info("  Ranking weight classes by avg_value within each archetype:\n")
    for arch in archetypes:
        if arch not in results:
            continue
        ranked = sorted(results[arch].items(), key=lambda x: x[1]["avg_value"], reverse=True)
        ranking_str = " > ".join(
            f"{wc}({v['tier']}:{v['avg_value']:.2f})" for wc, v in ranked if v["n"] >= 30
        )
        log.info(f"  {arch:<12s}: {ranking_str}")

    return results


# ---------------------------------------------------------------------------
# Section 2: Gender x Archetype
# ---------------------------------------------------------------------------

def analyze_gender_by_archetype(fighters_df, contests_df):
    section_header("2. GENDER x ARCHETYPE (Fighter-Level)")
    log.info("Do women fighters have higher/lower value in specific archetypes?\n")

    expanded = expand_fighters_by_archetype(fighters_df, contests_df)
    wc_data = expanded[expanded["weight_class"].notna() & (expanded["weight_class"] != "Unknown")]

    results = {}
    archetypes = ["UNLIMITED", "LIMITED", "SE", "20-Max", "150-Max"]

    for arch in archetypes:
        arch_data = wc_data[wc_data["archetype"] == arch]
        if arch_data.empty:
            continue

        agg = arch_data.groupby("gender").agg(
            n=("actual_points", "count"),
            avg_fpts=("actual_points", "mean"),
            avg_value=("salary_value", "mean"),
            ceiling_pct=("is_ceiling", "mean"),
            dud_pct=("is_dud", "mean"),
            avg_salary=("salary", "mean"),
            avg_own=("ownership", "mean"),
        ).reset_index()

        agg["ceiling_pct"] = (agg["ceiling_pct"] * 100).round(1)
        agg["dud_pct"] = (agg["dud_pct"] * 100).round(1)

        subsection_header(f"Gender — {arch} (n={len(arch_data):,})")
        log.info(f"  {'Gender':<8s} {'N':>6s} {'FPTS':>7s} {'Val':>6s} {'Ceil%':>6s} "
                 f"{'Dud%':>6s} {'AvgSal':>7s} {'Own%':>6s}")
        log.info(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6}")
        for _, row in agg.iterrows():
            flag = sample_flag(row["n"])
            own_str = f"{row['avg_own']:.1f}" if pd.notna(row["avg_own"]) else "N/A"
            log.info(f"  {row['gender']:<8s} {row['n']:>6.0f} {row['avg_fpts']:>7.1f} "
                     f"{row['avg_value']:>6.2f} {row['ceiling_pct']:>6.1f} "
                     f"{row['dud_pct']:>6.1f} {row['avg_salary']:>7.0f} "
                     f"{own_str:>6s}{flag}")

        results[arch] = {
            row["gender"]: {
                "n": int(row["n"]),
                "avg_fpts": round(row["avg_fpts"], 2),
                "avg_value": round(row["avg_value"], 3),
                "ceiling_pct": round(row["ceiling_pct"], 1),
                "dud_pct": round(row["dud_pct"], 1),
                "small_sample": row["n"] < 50,
            }
            for _, row in agg.iterrows()
        }

    # Summary comparison
    subsection_header("Gender Value Gap by Archetype")
    log.info(f"  {'Archetype':<12s} {'Men Val':>8s} {'Women Val':>10s} {'Delta':>8s} {'Women N':>8s}")
    log.info(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for arch in archetypes:
        if arch not in results:
            continue
        men_val = results[arch].get("Men", {}).get("avg_value", 0)
        women_val = results[arch].get("Women", {}).get("avg_value", 0)
        women_n = results[arch].get("Women", {}).get("n", 0)
        delta = women_val - men_val
        flag = sample_flag(women_n)
        log.info(f"  {arch:<12s} {men_val:>8.3f} {women_val:>10.3f} {delta:>+8.3f} {women_n:>8d}{flag}")

    return results


# ---------------------------------------------------------------------------
# Section 3: Line Movement x Archetype
# ---------------------------------------------------------------------------

def analyze_line_movement_by_archetype(odds_fighters_df, contests_df):
    section_header("3. LINE MOVEMENT x ARCHETYPE (Fighter-Level)")
    log.info("Does the steamed signal work differently across archetypes?")
    log.info("steamed = >2pp toward fighter, drifted = >2pp away, stable = <2pp\n")

    expanded = expand_fighters_by_archetype(odds_fighters_df, contests_df)
    move_data = expanded[expanded["line_move_dir"] != "unknown"]

    results = {}
    archetypes = ["UNLIMITED", "LIMITED", "SE", "20-Max", "150-Max"]

    for arch in archetypes:
        arch_data = move_data[move_data["archetype"] == arch]
        if arch_data.empty:
            continue

        agg = arch_data.groupby("line_move_dir").agg(
            n=("actual_points", "count"),
            avg_fpts=("actual_points", "mean"),
            avg_value=("salary_value", "mean"),
            ceiling_pct=("is_ceiling", "mean"),
            dud_pct=("is_dud", "mean"),
            avg_own=("actual_ownership", "mean"),
        ).reset_index()

        agg["ceiling_pct"] = (agg["ceiling_pct"] * 100).round(1)
        agg["dud_pct"] = (agg["dud_pct"] * 100).round(1)

        subsection_header(f"Line Movement — {arch} (n={len(arch_data):,})")
        log.info(f"  {'Direction':<10s} {'N':>6s} {'FPTS':>7s} {'Val':>6s} {'Ceil%':>6s} "
                 f"{'Dud%':>6s} {'Own%':>6s}")
        log.info(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
        for _, row in agg.iterrows():
            flag = sample_flag(row["n"])
            own_str = f"{row['avg_own']:.1f}" if pd.notna(row["avg_own"]) else "N/A"
            log.info(f"  {row['line_move_dir']:<10s} {row['n']:>6.0f} {row['avg_fpts']:>7.1f} "
                     f"{row['avg_value']:>6.2f} {row['ceiling_pct']:>6.1f} "
                     f"{row['dud_pct']:>6.1f} {own_str:>6s}{flag}")

        results[arch] = {
            row["line_move_dir"]: {
                "n": int(row["n"]),
                "avg_fpts": round(row["avg_fpts"], 2),
                "avg_value": round(row["avg_value"], 3),
                "ceiling_pct": round(row["ceiling_pct"], 1),
                "dud_pct": round(row["dud_pct"], 1),
                "small_sample": row["n"] < 50,
            }
            for _, row in agg.iterrows()
        }

    # Summary: steamed advantage by archetype
    subsection_header("Steamed vs Drifted Value Gap by Archetype")
    log.info(f"  {'Archetype':<12s} {'Steamed':>8s} {'Stable':>8s} {'Drifted':>8s} {'Steam-Drift':>12s}")
    log.info(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    for arch in archetypes:
        if arch not in results:
            continue
        s_val = results[arch].get("steamed", {}).get("avg_value", 0)
        st_val = results[arch].get("stable", {}).get("avg_value", 0)
        d_val = results[arch].get("drifted", {}).get("avg_value", 0)
        delta = s_val - d_val
        log.info(f"  {arch:<12s} {s_val:>8.3f} {st_val:>8.3f} {d_val:>8.3f} {delta:>+12.3f}")

    return results


# ---------------------------------------------------------------------------
# Section 4: Probability Band x Archetype
# ---------------------------------------------------------------------------

def analyze_prob_band_by_archetype(odds_fighters_df, contests_df):
    section_header("4. PROBABILITY BAND x ARCHETYPE (Fighter-Level)")
    log.info("FPTS/$1K and ceiling rate by prob band x archetype\n")

    expanded = expand_fighters_by_archetype(odds_fighters_df, contests_df)
    prob_data = expanded[expanded["prob_band"].notna()]

    results = {}
    archetypes = ["UNLIMITED", "LIMITED", "SE", "20-Max", "150-Max"]
    band_order = [label for label, _, _ in PROB_BANDS]

    for arch in archetypes:
        arch_data = prob_data[prob_data["archetype"] == arch]
        if arch_data.empty:
            continue

        agg = arch_data.groupby("prob_band").agg(
            n=("actual_points", "count"),
            avg_fpts=("actual_points", "mean"),
            avg_value=("salary_value", "mean"),
            ceiling_pct=("is_ceiling", "mean"),
            dud_pct=("is_dud", "mean"),
            avg_own=("actual_ownership", "mean"),
            avg_salary=("salary", "mean"),
        ).reset_index()

        agg["ceiling_pct"] = (agg["ceiling_pct"] * 100).round(1)
        agg["dud_pct"] = (agg["dud_pct"] * 100).round(1)

        # Sort by prob band order
        agg["_sort"] = agg["prob_band"].apply(lambda x: band_order.index(x) if x in band_order else 99)
        agg = agg.sort_values("_sort").drop(columns=["_sort"])

        subsection_header(f"Prob Band — {arch} (n={len(arch_data):,})")
        log.info(f"  {'Prob Band':<22s} {'N':>6s} {'FPTS':>7s} {'Val':>6s} {'Ceil%':>6s} "
                 f"{'Dud%':>6s} {'Own%':>6s} {'Sal':>7s}")
        log.info(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
        for _, row in agg.iterrows():
            flag = sample_flag(row["n"])
            own_str = f"{row['avg_own']:.1f}" if pd.notna(row["avg_own"]) else "N/A"
            log.info(f"  {row['prob_band']:<22s} {row['n']:>6.0f} {row['avg_fpts']:>7.1f} "
                     f"{row['avg_value']:>6.2f} {row['ceiling_pct']:>6.1f} "
                     f"{row['dud_pct']:>6.1f} {own_str:>6s} {row['avg_salary']:>7.0f}{flag}")

        results[arch] = {
            row["prob_band"]: {
                "n": int(row["n"]),
                "avg_fpts": round(row["avg_fpts"], 2),
                "avg_value": round(row["avg_value"], 3),
                "ceiling_pct": round(row["ceiling_pct"], 1),
                "dud_pct": round(row["dud_pct"], 1),
                "small_sample": row["n"] < 50,
            }
            for _, row in agg.iterrows()
        }

    return results


# ---------------------------------------------------------------------------
# Section 5: Card Position x Archetype
# ---------------------------------------------------------------------------

def analyze_card_position_by_archetype(odds_fighters_df, contests_df):
    section_header("5. CARD POSITION x ARCHETYPE (Fighter-Level)")
    log.info("FPTS/$1K by card_section x archetype\n")

    expanded = expand_fighters_by_archetype(odds_fighters_df, contests_df)
    card_data = expanded[expanded["card_section"].notna()]

    if card_data.empty:
        log.info("No card position data available.")
        return {}

    results = {}
    archetypes = ["UNLIMITED", "LIMITED", "SE", "20-Max", "150-Max"]

    for arch in archetypes:
        arch_data = card_data[card_data["archetype"] == arch]
        if arch_data.empty:
            continue

        agg = arch_data.groupby("card_section").agg(
            n=("actual_points", "count"),
            avg_fpts=("actual_points", "mean"),
            avg_value=("salary_value", "mean"),
            ceiling_pct=("is_ceiling", "mean"),
            dud_pct=("is_dud", "mean"),
            avg_own=("actual_ownership", "mean"),
            avg_salary=("salary", "mean"),
        ).reset_index()

        agg["ceiling_pct"] = (agg["ceiling_pct"] * 100).round(1)
        agg["dud_pct"] = (agg["dud_pct"] * 100).round(1)

        # Sort by card position order
        agg["_sort"] = agg["card_section"].map(CARD_SECTION_ORDER).fillna(99)
        agg = agg.sort_values("_sort").drop(columns=["_sort"])

        subsection_header(f"Card Position — {arch} (n={len(arch_data):,})")
        log.info(f"  {'Position':<15s} {'N':>6s} {'FPTS':>7s} {'Val':>6s} {'Ceil%':>6s} "
                 f"{'Dud%':>6s} {'Own%':>6s} {'Sal':>7s}")
        log.info(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
        for _, row in agg.iterrows():
            flag = sample_flag(row["n"])
            display = CARD_SECTION_DISPLAY.get(row["card_section"], row["card_section"])
            own_str = f"{row['avg_own']:.1f}" if pd.notna(row["avg_own"]) else "N/A"
            log.info(f"  {display:<15s} {row['n']:>6.0f} {row['avg_fpts']:>7.1f} "
                     f"{row['avg_value']:>6.2f} {row['ceiling_pct']:>6.1f} "
                     f"{row['dud_pct']:>6.1f} {own_str:>6s} {row['avg_salary']:>7.0f}{flag}")

        results[arch] = {
            CARD_SECTION_DISPLAY.get(row["card_section"], row["card_section"]): {
                "n": int(row["n"]),
                "avg_fpts": round(row["avg_fpts"], 2),
                "avg_value": round(row["avg_value"], 3),
                "ceiling_pct": round(row["ceiling_pct"], 1),
                "dud_pct": round(row["dud_pct"], 1),
                "small_sample": row["n"] < 50,
            }
            for _, row in agg.iterrows()
        }

    return results


# ---------------------------------------------------------------------------
# Section 6: Weight Class Composition x Archetype (Lineup-Level, Batched)
# ---------------------------------------------------------------------------

def analyze_lineup_composition_by_archetype(conn, contests_df):
    section_header("6. WEIGHT CLASS COMPOSITION x ARCHETYPE (Lineup-Level)")
    log.info("ROI by tier-S count, tier-F count, and women's count per lineup.")
    log.info("Processing batched by date_id to manage memory...\n")

    date_ids = contests_df["date_id"].unique()
    all_chunks = []

    for i, did in enumerate(sorted(date_ids)):
        if (i + 1) % 20 == 0:
            log.info(f"  Processing date {i+1}/{len(date_ids)}...")

        # Get contest info for this date
        date_contests = contests_df[contests_df["date_id"] == did]
        date_cids = date_contests["contest_id"].tolist()
        if not date_cids:
            continue

        cid_list = ",".join(str(c) for c in date_cids)

        # Load lineups + their fighters for this date
        chunk = pd.read_sql_query(f"""
            SELECT
                l.id AS lineup_id,
                l.contest_id,
                l.is_cashing,
                l.payout,
                l.points,
                l.total_ownership,
                pf.weight_class
            FROM lineups l
            JOIN lineup_players lp ON l.id = lp.lineup_id
            JOIN players p ON lp.player_id = p.player_id AND p.contest_id = l.contest_id
            LEFT JOIN player_features pf ON p.player_id = pf.player_id
                AND pf.contest_id = l.contest_id
            WHERE l.contest_id IN ({cid_list})
        """, conn)

        if chunk.empty:
            continue

        # Compute per-lineup composition
        chunk["tier"] = chunk["weight_class"].apply(get_tier)
        chunk["is_tier_s"] = (chunk["tier"] == "S").astype(int)
        chunk["is_tier_f"] = (chunk["tier"] == "F").astype(int)
        chunk["is_women"] = chunk["weight_class"].apply(is_womens).astype(int)

        lineup_comp = chunk.groupby(["lineup_id", "contest_id"]).agg(
            tier_s_count=("is_tier_s", "sum"),
            tier_f_count=("is_tier_f", "sum"),
            women_count=("is_women", "sum"),
            is_cashing=("is_cashing", "first"),
            payout=("payout", "first"),
            points=("points", "first"),
            total_ownership=("total_ownership", "first"),
        ).reset_index()

        all_chunks.append(lineup_comp)

    if not all_chunks:
        log.info("  No lineup composition data available.")
        return {}

    lineup_df = pd.concat(all_chunks, ignore_index=True)
    lineup_df["payout"] = lineup_df["payout"].fillna(0)

    # Merge with contest info for archetype and entry_cost
    lineup_df = lineup_df.merge(
        contests_df[["contest_id", "archetype", "entry_cost"]],
        on="contest_id", how="left",
    )

    log.info(f"  Processed {len(lineup_df):,} lineup-composition rows\n")

    results = {}
    archetypes = ["UNLIMITED", "LIMITED", "SE", "20-Max", "150-Max"]

    for arch in archetypes:
        arch_data = lineup_df[lineup_df["archetype"] == arch]
        if arch_data.empty:
            continue

        arch_results = {}

        # ROI by tier_s_count
        for comp_col, comp_label in [
            ("tier_s_count", "Tier S Count"),
            ("tier_f_count", "Tier F Count"),
            ("women_count", "Women Count"),
        ]:
            agg = arch_data.groupby(comp_col).agg(
                n=("payout", "count"),
                avg_payout=("payout", "mean"),
                avg_entry=("entry_cost", "mean"),
                cash_rate=("is_cashing", "mean"),
                avg_pts=("points", "mean"),
            ).reset_index()

            agg["roi"] = agg["avg_payout"] / agg["avg_entry"]
            agg["cash_rate"] = (agg["cash_rate"] * 100).round(1)

            subsection_header(f"{comp_label} — {arch}")
            log.info(f"  {comp_label:<15s} {'N':>8s} {'ROI':>8s} {'Cash%':>7s} {'AvgPts':>8s}")
            log.info(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")
            for _, row in agg.iterrows():
                flag = sample_flag(row["n"])
                log.info(f"  {int(row[comp_col]):<15d} {row['n']:>8.0f} {row['roi']:>8.4f} "
                         f"{row['cash_rate']:>7.1f} {row['avg_pts']:>8.1f}{flag}")

            arch_results[comp_label] = {
                str(int(row[comp_col])): {
                    "n": int(row["n"]),
                    "roi": round(float(row["roi"]), 4),
                    "cash_rate": round(row["cash_rate"], 1),
                    "avg_pts": round(row["avg_pts"], 2),
                    "small_sample": row["n"] < 50,
                }
                for _, row in agg.iterrows()
            }

        results[arch] = arch_results

    return results


# ---------------------------------------------------------------------------
# Section 7: Field Size x Line Movement (UNLIMITED only)
# ---------------------------------------------------------------------------

def analyze_field_size_x_line_movement(odds_fighters_df, contests_df):
    section_header("7. FIELD SIZE x LINE MOVEMENT (UNLIMITED Only)")
    log.info("Does the steamed signal matter more in bigger fields?\n")

    # Get UNLIMITED contests with field size bucket
    unlimited = contests_df[contests_df["archetype"] == "UNLIMITED"].copy()
    if unlimited.empty:
        log.info("No UNLIMITED contests found.")
        return {}

    # Expand odds fighters to UNLIMITED contests, then add field_bucket
    date_field = unlimited[["date_id", "field_bucket"]].drop_duplicates()

    # For each date, there may be multiple UNLIMITED contests with different sizes.
    # Use the largest contest's field bucket for simplicity.
    date_field = (unlimited.sort_values("contest_size", ascending=False)
                  .drop_duplicates("date_id", keep="first")[["date_id", "field_bucket"]])

    move_data = odds_fighters_df[odds_fighters_df["line_move_dir"] != "unknown"].copy()
    merged = move_data.merge(date_field, on="date_id", how="inner")

    if merged.empty:
        log.info("No merged data for UNLIMITED field size analysis.")
        return {}

    results = {}
    bucket_order = ["Small (<5K)", "Medium (5K-20K)", "Large (20K-50K)", "Mega (50K+)"]

    for bucket in bucket_order:
        bucket_data = merged[merged["field_bucket"] == bucket]
        if bucket_data.empty:
            continue

        agg = bucket_data.groupby("line_move_dir").agg(
            n=("actual_points", "count"),
            avg_fpts=("actual_points", "mean"),
            avg_value=("salary_value", "mean"),
            ceiling_pct=("is_ceiling", "mean"),
        ).reset_index()

        agg["ceiling_pct"] = (agg["ceiling_pct"] * 100).round(1)

        subsection_header(f"Field: {bucket}")
        log.info(f"  {'Direction':<10s} {'N':>6s} {'FPTS':>7s} {'Val':>6s} {'Ceil%':>6s}")
        log.info(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")
        for _, row in agg.iterrows():
            flag = sample_flag(row["n"])
            log.info(f"  {row['line_move_dir']:<10s} {row['n']:>6.0f} {row['avg_fpts']:>7.1f} "
                     f"{row['avg_value']:>6.2f} {row['ceiling_pct']:>6.1f}{flag}")

        results[bucket] = {
            row["line_move_dir"]: {
                "n": int(row["n"]),
                "avg_fpts": round(row["avg_fpts"], 2),
                "avg_value": round(row["avg_value"], 3),
                "ceiling_pct": round(row["ceiling_pct"], 1),
                "small_sample": row["n"] < 50,
            }
            for _, row in agg.iterrows()
        }

    # Summary: steamed advantage by field size
    subsection_header("Steamed-Drifted Delta by Field Size")
    log.info(f"  {'Field Size':<20s} {'Steamed Val':>12s} {'Drifted Val':>12s} {'Delta':>8s}")
    log.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*8}")
    for bucket in bucket_order:
        if bucket not in results:
            continue
        s_val = results[bucket].get("steamed", {}).get("avg_value", 0)
        d_val = results[bucket].get("drifted", {}).get("avg_value", 0)
        delta = s_val - d_val
        log.info(f"  {bucket:<20s} {s_val:>12.3f} {d_val:>12.3f} {delta:>+8.3f}")

    return results


# ---------------------------------------------------------------------------
# Section 8: Effective Entries x Ownership Strategy
# ---------------------------------------------------------------------------

def analyze_effective_entries_x_ownership(conn, contests_df):
    section_header("8. EFFECTIVE ENTRIES x OWNERSHIP STRATEGY")
    log.info("Does optimal ownership level shift with effective entries?")
    log.info("Cross-tab: effective_entries band x total_ownership band -> ROI\n")

    # Compute effective entries per contest
    contests = contests_df.copy()
    max_e = contests["multi_entry_max"].fillna(999)
    contests["effective_entries"] = max_e.clip(upper=contests["contest_size"]) / contests["contest_size"]

    contests["ee_band"] = pd.cut(
        contests["effective_entries"],
        bins=[0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.01],
        labels=["<0.1%", "0.1-1%", "1-5%", "5-10%", "10-50%", "50%+"],
    )

    # Load lineups
    log.info("Loading lineups for effective entries analysis...")
    lineups = pd.read_sql_query(f"""
        SELECT id AS lineup_id, contest_id, total_ownership, is_cashing, payout, points
        FROM lineups
        WHERE contest_id != {FREEROLL_ID}
    """, conn)
    lineups["payout"] = lineups["payout"].fillna(0)

    # Merge
    merged = lineups.merge(
        contests[["contest_id", "entry_cost", "ee_band", "effective_entries"]],
        on="contest_id", how="inner",
    )

    # Ownership bands
    merged["own_band"] = pd.cut(
        merged["total_ownership"],
        bins=[0, 100, 125, 150, 175, 200, 9999],
        labels=["<100%", "100-125%", "125-150%", "150-175%", "175-200%", "200%+"],
        right=True,
    )

    # Cross-tab: ee_band x own_band -> ROI
    results = {}
    ee_bands = ["<0.1%", "0.1-1%", "1-5%", "5-10%", "10-50%", "50%+"]
    own_bands = ["<100%", "100-125%", "125-150%", "150-175%", "175-200%", "200%+"]

    # Print header
    header = f"  {'EE Band':<10s}"
    for ob in own_bands:
        header += f" {ob:>12s}"
    header += f" {'Best':>12s}"
    log.info(header)
    log.info(f"  {'-'*10}" + f" {'-'*12}" * len(own_bands) + f" {'-'*12}")

    for ee in ee_bands:
        ee_data = merged[merged["ee_band"] == ee]
        if ee_data.empty:
            continue

        row_results = {}
        best_roi = -999
        best_band = ""

        for ob in own_bands:
            cell = ee_data[ee_data["own_band"] == ob]
            n = len(cell)
            if n < 10:
                row_results[ob] = {"roi": None, "n": n, "small_sample": True}
                continue

            roi = cell["payout"].mean() / cell["entry_cost"].mean() if cell["entry_cost"].mean() > 0 else 0
            row_results[ob] = {
                "roi": round(float(roi), 4),
                "n": int(n),
                "small_sample": n < 50,
            }
            if roi > best_roi:
                best_roi = roi
                best_band = ob

        # Print row
        row_str = f"  {ee:<10s}"
        for ob in own_bands:
            r = row_results.get(ob, {})
            if r.get("roi") is not None:
                flag = "*" if r.get("small_sample") else ""
                row_str += f" {r['roi']:>11.4f}{flag}"
            else:
                row_str += f" {'N/A':>12s}"
        row_str += f" {best_band:>12s}"
        log.info(row_str)

        results[ee] = row_results

    log.info("\n  * = small sample (n < 50)")

    # Additional: ROI by ee_band alone
    subsection_header("ROI by Effective Entries Band")
    log.info(f"  {'EE Band':<10s} {'N':>10s} {'ROI':>8s} {'Cash%':>7s} {'AvgPts':>8s}")
    log.info(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*8}")
    for ee in ee_bands:
        ee_data = merged[merged["ee_band"] == ee]
        if ee_data.empty:
            continue
        n = len(ee_data)
        roi = ee_data["payout"].mean() / ee_data["entry_cost"].mean() if ee_data["entry_cost"].mean() > 0 else 0
        cash = ee_data["is_cashing"].mean() * 100
        pts = ee_data["points"].mean()
        flag = sample_flag(n)
        log.info(f"  {ee:<10s} {n:>10,d} {roi:>8.4f} {cash:>7.1f} {pts:>8.1f}{flag}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info(SEPARATOR)
    log.info("  UFC DFS Fighter Rules by Archetype Analysis")
    log.info(SEPARATOR)

    conn = sqlite3.connect(DB_PATH)
    contests_df = load_contests(conn)

    # Load fighter data (no lineup_players join needed for sections 1-5)
    fighters_df = load_fighters(conn, contests_df)
    odds_fighters_df = load_odds_fighters(conn, contests_df)

    # Sections 1-5: Fighter-level analyses (fast)
    all_results = {}

    all_results["weight_class_x_archetype"] = analyze_weight_class_by_archetype(
        fighters_df, contests_df
    )

    all_results["gender_x_archetype"] = analyze_gender_by_archetype(
        fighters_df, contests_df
    )

    all_results["line_movement_x_archetype"] = analyze_line_movement_by_archetype(
        odds_fighters_df, contests_df
    )

    all_results["prob_band_x_archetype"] = analyze_prob_band_by_archetype(
        odds_fighters_df, contests_df
    )

    all_results["card_position_x_archetype"] = analyze_card_position_by_archetype(
        odds_fighters_df, contests_df
    )

    # Section 6: Lineup-level composition (batched, slower)
    all_results["lineup_composition_x_archetype"] = analyze_lineup_composition_by_archetype(
        conn, contests_df
    )

    # Section 7: Field size x line movement (UNLIMITED only)
    all_results["field_size_x_line_movement"] = analyze_field_size_x_line_movement(
        odds_fighters_df, contests_df
    )

    # Section 8: Effective entries x ownership
    all_results["effective_entries_x_ownership"] = analyze_effective_entries_x_ownership(
        conn, contests_df
    )

    conn.close()

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "fighter_archetype_results.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert_for_json)

    log.info(f"\n{SEPARATOR}")
    log.info(f"  Results saved to {output_path}")
    log.info(SEPARATOR)


if __name__ == "__main__":
    main()
