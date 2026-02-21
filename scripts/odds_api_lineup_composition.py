"""
UFC DFS Odds API Lineup Composition Analysis

Uses the Odds API dataset (fighter_odds_enriched) for two analyses:
  Part A — Fighter-level: performance by prob band, line movement, cross-tabs
  Part B — Lineup-level: odds composition (fav count, heavy fav count, avg prob)
           linked to ROI by archetype and entry fee tier

Output: formatted tables to console + JSON to data/analysis/odds_composition_results.json
"""

import sqlite3
import numpy as np
import pandas as pd
import logging
import json
import os
from pathlib import Path

DB_PATH = "/Users/ryancullen/Desktop/resultsdb_ufc.db"
OUTPUT_DIR = Path("/Users/ryancullen/ufc-dfs-analytics/data/analysis")
FREEROLL_ID = 142690805

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SEPARATOR = "=" * 70
SUBSEP = "-" * 50

WOMENS_WEIGHT_CLASSES = {
    "W-Strawweight", "W-Flyweight", "W-Bantamweight", "W-Featherweight",
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
}

PROB_BANDS = [
    ("Heavy Fav (70%+)", 0.70, 1.01),
    ("Mod Fav (55-70%)", 0.55, 0.70),
    ("Toss-up (45-55%)", 0.45, 0.55),
    ("Mod Dog (30-45%)", 0.30, 0.45),
    ("Heavy Dog (<30%)", 0.00, 0.30),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_contest(multi_entry_max):
    """Classify a contest into one of 5 archetypes."""
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


def entry_fee_tier(entry_cost):
    if entry_cost <= 3:
        return "MICRO ($1-3)"
    elif entry_cost <= 8:
        return "LOW ($5-8)"
    elif entry_cost <= 20:
        return "MID ($15-20)"
    else:
        return "HIGH ($25-30)"


def is_womens(weight_class):
    return weight_class in WOMENS_WEIGHT_CLASSES if weight_class else False


def assign_prob_band(prob):
    """Return the label for a close_prob value."""
    if prob is None or np.isnan(prob):
        return None
    for label, lo, hi in PROB_BANDS:
        if lo <= prob < hi:
            return label
    # edge: prob == 1.0
    if prob >= 0.70:
        return "Heavy Fav (70%+)"
    return None


def assign_line_move_dir(open_prob, close_prob):
    """Classify line movement direction based on open vs close prob."""
    if open_prob is None or close_prob is None:
        return "unknown"
    if np.isnan(open_prob) or np.isnan(close_prob):
        return "unknown"
    diff = close_prob - open_prob  # positive = steamed toward fighter
    if diff > 0.02:
        return "steamed"
    elif diff < -0.02:
        return "drifted"
    else:
        return "stable"


def assign_steam_magnitude(open_prob, close_prob):
    """For steamed fighters, classify magnitude."""
    if open_prob is None or close_prob is None:
        return None
    if np.isnan(open_prob) or np.isnan(close_prob):
        return None
    diff = close_prob - open_prob
    if diff <= 0.02:
        return None  # not steamed
    mag = abs(diff) * 100  # convert to percentage points
    if mag < 5:
        return "steamed_small (2-5pp)"
    elif mag < 10:
        return "steamed_medium (5-10pp)"
    elif mag < 20:
        return "steamed_large (10-20pp)"
    else:
        return "steamed_huge (20pp+)"


def fmt_pct(val, digits=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{digits}%}"


def fmt_f(val, digits=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{digits}f}"


def json_safe(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Part A: Fighter-Level Analysis
# ---------------------------------------------------------------------------

def load_fighter_odds(conn):
    """Load fighter_odds_enriched with weight class and card position."""
    log.info("Loading fighter_odds_enriched...")

    df = pd.read_sql_query("""
        SELECT foe.*,
               pf.weight_class,
               fcp.card_section
        FROM fighter_odds_enriched foe
        LEFT JOIN player_features pf
            ON foe.player_id = pf.player_id
            AND pf.contest_id IN (
                SELECT c.contest_id FROM contests c
                WHERE c.date_id = foe.date_id AND c.contest_id != ?
            )
        LEFT JOIN fighter_card_position fcp
            ON foe.date_id = fcp.date_id AND foe.player_id = fcp.player_id
        GROUP BY foe.date_id, foe.canonical_id
    """, conn, params=(FREEROLL_ID,))

    log.info(f"  Loaded {len(df):,} fighter-events across {df['date_id'].nunique()} dates")

    # Derived columns
    df["prob_band"] = df["close_prob"].apply(assign_prob_band)
    df["salary_value"] = df.apply(
        lambda r: r["actual_points"] / (r["salary"] / 1000) if r["salary"] and r["salary"] > 0 else np.nan,
        axis=1,
    )
    df["line_move_dir"] = df.apply(
        lambda r: assign_line_move_dir(r["open_prob"], r["close_prob"]), axis=1
    )
    df["steam_magnitude"] = df.apply(
        lambda r: assign_steam_magnitude(r["open_prob"], r["close_prob"]), axis=1
    )
    df["gender"] = df["weight_class"].apply(lambda wc: "Women" if is_womens(wc) else "Men")
    df["ceiling_hit"] = df["salary_value"] >= 8.0

    # Compute per-event median points for dud classification
    event_medians = df.groupby("date_id")["actual_points"].median().rename("event_median")
    df = df.merge(event_medians, on="date_id", how="left")
    df["is_dud"] = (df["actual_points"] == 0) | (df["actual_points"] < df["event_median"] * 0.3)

    return df


def analyze_performance_by_prob_band(df):
    """Part A.1 — Performance by close_prob band."""
    log.info(f"\n{SEPARATOR}")
    log.info("  PART A.1: Performance by Close Prob Band")
    log.info(SEPARATOR)

    results = {}
    header = (
        f"  {'Band':<22s} {'N':>6s} {'AvgFPTS':>8s} {'FPTS/$1K':>9s} {'AvgSal':>8s} "
        f"{'AvgOwn%':>8s} {'Ceil%':>7s} {'100+%':>7s} {'Dud%':>7s}"
    )
    log.info(header)
    log.info(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    for label, lo, hi in PROB_BANDS:
        band = df[(df["close_prob"] >= lo) & (df["close_prob"] < hi)]
        if band.empty:
            continue
        n = len(band)
        avg_fpts = band["actual_points"].mean()
        avg_sv = band["salary_value"].mean()
        avg_sal = band["salary"].mean()
        avg_own = band["actual_ownership"].mean()
        ceil_rate = band["ceiling_hit"].mean()
        rate_100 = (band["actual_points"] >= 100).mean()
        dud_rate = band["is_dud"].mean()

        results[label] = {
            "n": int(n),
            "avg_fpts": round(float(avg_fpts), 2),
            "fpts_per_1k": round(float(avg_sv), 3),
            "avg_salary": round(float(avg_sal), 0),
            "avg_ownership": round(float(avg_own), 2),
            "ceiling_rate": round(float(ceil_rate), 4),
            "rate_100_plus": round(float(rate_100), 4),
            "dud_rate": round(float(dud_rate), 4),
        }

        log.info(
            f"  {label:<22s} {n:>6,d} {avg_fpts:>8.1f} {avg_sv:>9.3f} {avg_sal:>8.0f} "
            f"{avg_own:>7.1f}% {ceil_rate:>6.1%} {rate_100:>6.1%} {dud_rate:>6.1%}"
        )

    return results


def analyze_outscore_rate(df):
    """Part A.2 — DFS outscore rate vs implied probability."""
    log.info(f"\n{SEPARATOR}")
    log.info("  PART A.2: Outscore Rate vs Implied Probability")
    log.info(SEPARATOR)

    # Per-event median
    event_medians = df.groupby("date_id")["actual_points"].median().rename("slate_median")
    merged = df.merge(event_medians, on="date_id", how="left")
    merged["above_median"] = merged["actual_points"] > merged["slate_median"]

    results = {}
    header = f"  {'Band':<22s} {'N':>6s} {'Outscore%':>10s} {'AvgProb':>9s} {'Delta':>8s}"
    log.info(header)
    log.info(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*9} {'-'*8}")

    for label, lo, hi in PROB_BANDS:
        band = merged[(merged["close_prob"] >= lo) & (merged["close_prob"] < hi)]
        if band.empty:
            continue
        n = len(band)
        outscore = band["above_median"].mean()
        avg_prob = band["close_prob"].mean()
        delta = outscore - avg_prob

        results[label] = {
            "n": int(n),
            "outscore_rate": round(float(outscore), 4),
            "avg_implied_prob": round(float(avg_prob), 4),
            "delta": round(float(delta), 4),
        }

        log.info(
            f"  {label:<22s} {n:>6,d} {outscore:>9.1%} {avg_prob:>8.1%} {delta:>+7.1%}"
        )

    return results


def analyze_line_movement(df):
    """Part A.3 — Line movement analysis."""
    log.info(f"\n{SEPARATOR}")
    log.info("  PART A.3: Line Movement Analysis")
    log.info(SEPARATOR)

    results = {"by_direction": {}, "by_magnitude": {}}

    # By direction
    log.info("\n  By Direction:")
    header = f"  {'Direction':<15s} {'N':>6s} {'AvgFPTS':>8s} {'FPTS/$1K':>9s} {'Ceil%':>7s} {'AvgOwn%':>8s}"
    log.info(header)
    log.info(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*9} {'-'*7} {'-'*8}")

    for direction in ["steamed", "stable", "drifted"]:
        band = df[df["line_move_dir"] == direction]
        if band.empty:
            continue
        n = len(band)
        avg_fpts = band["actual_points"].mean()
        avg_sv = band["salary_value"].mean()
        ceil_rate = band["ceiling_hit"].mean()
        avg_own = band["actual_ownership"].mean()

        results["by_direction"][direction] = {
            "n": int(n),
            "avg_fpts": round(float(avg_fpts), 2),
            "fpts_per_1k": round(float(avg_sv), 3),
            "ceiling_rate": round(float(ceil_rate), 4),
            "avg_ownership": round(float(avg_own), 2),
        }

        log.info(
            f"  {direction:<15s} {n:>6,d} {avg_fpts:>8.1f} {avg_sv:>9.3f} {ceil_rate:>6.1%} {avg_own:>7.1f}%"
        )

    # By magnitude (steamed only)
    steamed = df[df["line_move_dir"] == "steamed"].copy()
    if not steamed.empty:
        log.info("\n  Steam Magnitude (steamed fighters only):")
        log.info(header)
        log.info(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*9} {'-'*7} {'-'*8}")

        mag_order = [
            "steamed_small (2-5pp)",
            "steamed_medium (5-10pp)",
            "steamed_large (10-20pp)",
            "steamed_huge (20pp+)",
        ]
        for mag in mag_order:
            band = steamed[steamed["steam_magnitude"] == mag]
            if band.empty:
                continue
            n = len(band)
            avg_fpts = band["actual_points"].mean()
            avg_sv = band["salary_value"].mean()
            ceil_rate = band["ceiling_hit"].mean()
            avg_own = band["actual_ownership"].mean()

            results["by_magnitude"][mag] = {
                "n": int(n),
                "avg_fpts": round(float(avg_fpts), 2),
                "fpts_per_1k": round(float(avg_sv), 3),
                "ceiling_rate": round(float(ceil_rate), 4),
                "avg_ownership": round(float(avg_own), 2),
            }

            short = mag.split("(")[0].strip()
            log.info(
                f"  {short:<15s} {n:>6,d} {avg_fpts:>8.1f} {avg_sv:>9.3f} {ceil_rate:>6.1%} {avg_own:>7.1f}%"
            )

    return results


def analyze_cross_tabs(df):
    """Part A.4 — Cross-tabulations."""
    log.info(f"\n{SEPARATOR}")
    log.info("  PART A.4: Cross-Tabulations")
    log.info(SEPARATOR)

    results = {}

    # --- Prob band x weight class ---
    log.info("\n  Prob Band x Weight Class (avg FPTS/$1K):")
    wc_vals = sorted(df["weight_class"].dropna().unique())
    prob_labels = [l for l, _, _ in PROB_BANDS]

    xtab_wc = {}
    # Header
    wc_short = {wc: wc[:12] for wc in wc_vals}
    hdr = f"  {'Band':<22s}" + "".join(f" {wc_short[wc]:>12s}" for wc in wc_vals)
    log.info(hdr)
    log.info(f"  {'-'*22}" + "".join(f" {'-'*12}" for _ in wc_vals))

    for label, lo, hi in PROB_BANDS:
        row = {}
        row_str = f"  {label:<22s}"
        band = df[(df["close_prob"] >= lo) & (df["close_prob"] < hi)]
        for wc in wc_vals:
            cell = band[band["weight_class"] == wc]
            if len(cell) >= 5:
                val = cell["salary_value"].mean()
                row[wc] = {"n": int(len(cell)), "fpts_per_1k": round(float(val), 3)}
                row_str += f" {val:>11.3f}*" if len(cell) < 20 else f" {val:>12.3f}"
            else:
                row[wc] = None
                row_str += f" {'---':>12s}"
        xtab_wc[label] = row
        log.info(row_str)
    results["prob_x_weight_class"] = xtab_wc

    # --- Prob band x line movement direction ---
    log.info("\n  Prob Band x Line Movement Direction (avg FPTS/$1K):")
    dirs = ["steamed", "stable", "drifted"]
    hdr = f"  {'Band':<22s}" + "".join(f" {d:>12s}" for d in dirs)
    log.info(hdr)
    log.info(f"  {'-'*22}" + "".join(f" {'-'*12}" for _ in dirs))

    xtab_lm = {}
    for label, lo, hi in PROB_BANDS:
        row = {}
        row_str = f"  {label:<22s}"
        band = df[(df["close_prob"] >= lo) & (df["close_prob"] < hi)]
        for d in dirs:
            cell = band[band["line_move_dir"] == d]
            if len(cell) >= 5:
                val = cell["salary_value"].mean()
                n = len(cell)
                row[d] = {"n": int(n), "fpts_per_1k": round(float(val), 3)}
                row_str += f" {val:>12.3f}"
            else:
                row[d] = None
                row_str += f" {'---':>12s}"
        xtab_lm[label] = row
        log.info(row_str)
    results["prob_x_line_move"] = xtab_lm

    # --- Prob band x card position ---
    log.info("\n  Prob Band x Card Section (avg FPTS/$1K):")
    sections = sorted(df["card_section"].dropna().unique())
    hdr = f"  {'Band':<22s}" + "".join(f" {s:>12s}" for s in sections)
    log.info(hdr)
    log.info(f"  {'-'*22}" + "".join(f" {'-'*12}" for _ in sections))

    xtab_cp = {}
    for label, lo, hi in PROB_BANDS:
        row = {}
        row_str = f"  {label:<22s}"
        band = df[(df["close_prob"] >= lo) & (df["close_prob"] < hi)]
        for s in sections:
            cell = band[band["card_section"] == s]
            if len(cell) >= 5:
                val = cell["salary_value"].mean()
                n = len(cell)
                row[s] = {"n": int(n), "fpts_per_1k": round(float(val), 3)}
                row_str += f" {val:>12.3f}"
            else:
                row[s] = None
                row_str += f" {'---':>12s}"
        xtab_cp[label] = row
        log.info(row_str)
    results["prob_x_card_section"] = xtab_cp

    # --- Prob band x gender ---
    log.info("\n  Prob Band x Gender (avg FPTS/$1K):")
    genders = ["Men", "Women"]
    hdr = f"  {'Band':<22s}" + "".join(f" {g:>12s}" for g in genders)
    log.info(hdr)
    log.info(f"  {'-'*22}" + "".join(f" {'-'*12}" for _ in genders))

    xtab_g = {}
    for label, lo, hi in PROB_BANDS:
        row = {}
        row_str = f"  {label:<22s}"
        band = df[(df["close_prob"] >= lo) & (df["close_prob"] < hi)]
        for g in genders:
            cell = band[band["gender"] == g]
            if len(cell) >= 5:
                val = cell["salary_value"].mean()
                n = len(cell)
                row[g] = {"n": int(n), "fpts_per_1k": round(float(val), 3)}
                row_str += f" {val:>12.3f}"
            else:
                row[g] = None
                row_str += f" {'---':>12s}"
        xtab_g[label] = row
        log.info(row_str)
    results["prob_x_gender"] = xtab_g

    return results


# ---------------------------------------------------------------------------
# Part B: Lineup Composition
# ---------------------------------------------------------------------------

def compute_lineup_odds_batched(conn):
    """
    For each date with odds data, compute per-lineup odds composition.
    Processes date by date to manage memory on the 12M row lineup_players table.
    """
    log.info(f"\n{SEPARATOR}")
    log.info("  PART B: Computing Lineup Odds Composition (batched by date)")
    log.info(SEPARATOR)

    # Get dates that have odds data
    odds_dates = pd.read_sql_query(
        "SELECT DISTINCT date_id FROM fighter_odds_enriched ORDER BY date_id",
        conn,
    )["date_id"].tolist()
    log.info(f"  {len(odds_dates)} dates with odds data")

    # Pre-load canonical_fighter lookup for name matching
    canonical_df = pd.read_sql_query(
        "SELECT canonical_id, canonical_name FROM canonical_fighter", conn
    )
    name_to_canonical = dict(zip(canonical_df["canonical_name"], canonical_df["canonical_id"]))

    # Pre-load all odds keyed by (date_id, canonical_id)
    all_odds = pd.read_sql_query("""
        SELECT date_id, canonical_id, close_prob, is_favorite
        FROM fighter_odds_enriched
    """, conn)
    odds_lookup = {}
    for _, row in all_odds.iterrows():
        key = (int(row["date_id"]), int(row["canonical_id"]))
        odds_lookup[key] = (row["close_prob"], row["is_favorite"])

    all_lineup_chunks = []
    total_lineups = 0

    for i, date_id in enumerate(odds_dates):
        if (i + 1) % 20 == 0 or i == 0:
            log.info(f"  Processing date {i+1}/{len(odds_dates)} (date_id={date_id})...")

        # Get all non-freeroll contests for this date
        contests = pd.read_sql_query(f"""
            SELECT contest_id, entry_cost, multi_entry_max, contest_size, cash_line, total_prizes
            FROM contests
            WHERE date_id = {date_id} AND contest_id != {FREEROLL_ID}
        """, conn)

        if contests.empty:
            continue

        contest_ids = contests["contest_id"].tolist()
        cid_str = ",".join(str(c) for c in contest_ids)

        # Get all lineups and their fighters for these contests
        lineup_data = pd.read_sql_query(f"""
            SELECT l.id as lineup_id, l.contest_id, l.points, l.payout, l.is_cashing,
                   l.lineup_rank, l.total_ownership,
                   p.full_name, p.player_id
            FROM lineups l
            JOIN lineup_players lp ON l.id = lp.lineup_id
            JOIN players p ON lp.player_id = p.player_id AND p.contest_id = l.contest_id
            WHERE l.contest_id IN ({cid_str})
        """, conn)

        if lineup_data.empty:
            continue

        # Map fighter names to canonical_id
        lineup_data["canonical_id"] = lineup_data["full_name"].map(name_to_canonical)

        # Look up odds for each fighter
        lineup_data["close_prob"] = lineup_data.apply(
            lambda r: odds_lookup.get((date_id, r["canonical_id"]), (None, None))[0]
            if pd.notna(r["canonical_id"]) else None,
            axis=1,
        )
        lineup_data["is_fav_odds"] = lineup_data.apply(
            lambda r: odds_lookup.get((date_id, r["canonical_id"]), (None, None))[1]
            if pd.notna(r["canonical_id"]) else None,
            axis=1,
        )

        # Aggregate per lineup
        per_lineup = lineup_data.groupby(["lineup_id", "contest_id"]).agg(
            points=("points", "first"),
            payout=("payout", "first"),
            is_cashing=("is_cashing", "first"),
            lineup_rank=("lineup_rank", "first"),
            total_ownership=("total_ownership", "first"),
            odds_coverage=("close_prob", "count"),  # non-null count below
            fav_count=("is_fav_odds", lambda x: (x == 1).sum()),
            heavy_fav_count=("close_prob", lambda x: (x >= 0.7).sum()),
            avg_close_prob=("close_prob", "mean"),
        ).reset_index()

        # Recount odds_coverage as count of non-null close_prob
        odds_cov = lineup_data.groupby("lineup_id")["close_prob"].apply(
            lambda x: x.notna().sum()
        ).rename("odds_coverage_actual")
        per_lineup = per_lineup.merge(odds_cov, on="lineup_id", how="left")
        per_lineup["odds_coverage"] = per_lineup["odds_coverage_actual"]
        per_lineup.drop(columns=["odds_coverage_actual"], inplace=True)

        per_lineup["date_id"] = date_id

        # Merge contest info
        per_lineup = per_lineup.merge(
            contests[["contest_id", "entry_cost", "multi_entry_max", "contest_size"]],
            on="contest_id", how="left",
        )

        all_lineup_chunks.append(per_lineup)
        total_lineups += len(per_lineup)

    if not all_lineup_chunks:
        log.info("  No lineup data found for odds-covered dates!")
        return pd.DataFrame()

    lineup_odds_df = pd.concat(all_lineup_chunks, ignore_index=True)
    lineup_odds_df["payout"] = lineup_odds_df["payout"].fillna(0)
    lineup_odds_df["archetype"] = lineup_odds_df["multi_entry_max"].apply(classify_contest)
    lineup_odds_df["fee_tier"] = lineup_odds_df["entry_cost"].apply(entry_fee_tier)

    log.info(f"  Total lineups with odds stats: {len(lineup_odds_df):,}")
    log.info(f"  Lineups with full odds coverage (6/6): {(lineup_odds_df['odds_coverage'] == 6).sum():,}")

    return lineup_odds_df


def compute_roi_from_df(df):
    """
    Compute ROI = avg_payout / entry_cost, per contest then weighted-average.
    df must have contest_id, payout, entry_cost columns.
    """
    if df.empty:
        return np.nan
    per_contest = df.groupby("contest_id").agg(
        avg_payout=("payout", "mean"),
        entry_cost=("entry_cost", "first"),
        n_lineups=("payout", "count"),
    )
    per_contest["roi"] = per_contest["avg_payout"] / per_contest["entry_cost"]
    total = per_contest["n_lineups"].sum()
    if total == 0:
        return np.nan
    return (per_contest["roi"] * per_contest["n_lineups"]).sum() / total


def analyze_lineup_composition(lineup_odds_df):
    """Part B analyses — ROI by favorite count, heavy fav count, cross-tabs."""
    log.info(f"\n{SEPARATOR}")
    log.info("  PART B: Lineup Composition — Odds-Based Metrics")
    log.info(SEPARATOR)

    # Filter to full coverage lineups
    full = lineup_odds_df[lineup_odds_df["odds_coverage"] == 6].copy()
    log.info(f"  Analyzing {len(full):,} lineups with full odds coverage (6/6)")
    log.info(f"  Dates covered: {full['date_id'].nunique()}")
    log.info(f"  Contests covered: {full['contest_id'].nunique()}")

    results = {}

    # --- B.1: ROI by favorite count (0-6) ---
    log.info(f"\n{SUBSEP}")
    log.info("  ROI by Favorite Count (all archetypes)")
    log.info(SUBSEP)
    header = f"  {'FavCount':>8s} {'N':>10s} {'ROI':>8s} {'Cash%':>8s} {'AvgPts':>8s} {'AvgProb':>8s}"
    log.info(header)
    log.info(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    roi_by_fav = {}
    for fc in range(7):
        grp = full[full["fav_count"] == fc]
        if grp.empty:
            continue
        roi = compute_roi_from_df(grp)
        cash_rate = grp["is_cashing"].mean()
        avg_pts = grp["points"].mean()
        avg_prob = grp["avg_close_prob"].mean()

        roi_by_fav[str(fc)] = {
            "n": int(len(grp)),
            "roi": round(float(roi), 4) if not np.isnan(roi) else None,
            "cash_rate": round(float(cash_rate), 4),
            "avg_points": round(float(avg_pts), 2),
            "avg_close_prob": round(float(avg_prob), 4),
        }

        roi_str = fmt_f(roi, 4)
        log.info(
            f"  {fc:>8d} {len(grp):>10,d} {roi_str:>8s} {cash_rate:>7.1%} {avg_pts:>8.1f} {avg_prob:>7.1%}"
        )
    results["roi_by_fav_count"] = roi_by_fav

    # --- B.2: ROI by heavy favorite count (0-6) ---
    log.info(f"\n{SUBSEP}")
    log.info("  ROI by Heavy Favorite Count (close_prob >= 70%)")
    log.info(SUBSEP)
    log.info(header)
    log.info(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    roi_by_hfav = {}
    for hfc in range(7):
        grp = full[full["heavy_fav_count"] == hfc]
        if grp.empty:
            continue
        roi = compute_roi_from_df(grp)
        cash_rate = grp["is_cashing"].mean()
        avg_pts = grp["points"].mean()
        avg_prob = grp["avg_close_prob"].mean()

        roi_by_hfav[str(hfc)] = {
            "n": int(len(grp)),
            "roi": round(float(roi), 4) if not np.isnan(roi) else None,
            "cash_rate": round(float(cash_rate), 4),
            "avg_points": round(float(avg_pts), 2),
            "avg_close_prob": round(float(avg_prob), 4),
        }

        roi_str = fmt_f(roi, 4)
        log.info(
            f"  {hfc:>8d} {len(grp):>10,d} {roi_str:>8s} {cash_rate:>7.1%} {avg_pts:>8.1f} {avg_prob:>7.1%}"
        )
    results["roi_by_heavy_fav_count"] = roi_by_hfav

    # --- B.3: Cross-tab: Favorite count x archetype ---
    log.info(f"\n{SUBSEP}")
    log.info("  ROI: Favorite Count x Archetype")
    log.info(SUBSEP)

    archetypes = ["SE", "LIMITED", "20-Max", "150-Max", "UNLIMITED"]
    hdr = f"  {'FavCnt':>6s}" + "".join(f" {a:>12s}" for a in archetypes)
    log.info(hdr)
    log.info(f"  {'-'*6}" + "".join(f" {'-'*12}" for _ in archetypes))

    xtab_fav_arch = {}
    for fc in range(7):
        row = {}
        row_str = f"  {fc:>6d}"
        for arch in archetypes:
            grp = full[(full["fav_count"] == fc) & (full["archetype"] == arch)]
            if len(grp) >= 30:
                roi = compute_roi_from_df(grp)
                row[arch] = {
                    "n": int(len(grp)),
                    "roi": round(float(roi), 4) if not np.isnan(roi) else None,
                }
                roi_str = fmt_f(roi, 4)
                row_str += f" {roi_str:>12s}"
            else:
                row[arch] = None
                row_str += f" {'---':>12s}"
        xtab_fav_arch[str(fc)] = row
        log.info(row_str)
    results["fav_count_x_archetype"] = xtab_fav_arch

    # --- B.4: Cross-tab: Favorite count x entry fee tier ---
    log.info(f"\n{SUBSEP}")
    log.info("  ROI: Favorite Count x Entry Fee Tier")
    log.info(SUBSEP)

    tiers = ["MICRO ($1-3)", "LOW ($5-8)", "MID ($15-20)", "HIGH ($25-30)"]
    hdr = f"  {'FavCnt':>6s}" + "".join(f" {t:>15s}" for t in tiers)
    log.info(hdr)
    log.info(f"  {'-'*6}" + "".join(f" {'-'*15}" for _ in tiers))

    xtab_fav_fee = {}
    for fc in range(7):
        row = {}
        row_str = f"  {fc:>6d}"
        for tier in tiers:
            grp = full[(full["fav_count"] == fc) & (full["fee_tier"] == tier)]
            if len(grp) >= 30:
                roi = compute_roi_from_df(grp)
                row[tier] = {
                    "n": int(len(grp)),
                    "roi": round(float(roi), 4) if not np.isnan(roi) else None,
                }
                roi_str = fmt_f(roi, 4)
                row_str += f" {roi_str:>15s}"
            else:
                row[tier] = None
                row_str += f" {'---':>15s}"
        xtab_fav_fee[str(fc)] = row
        log.info(row_str)
    results["fav_count_x_fee_tier"] = xtab_fav_fee

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info(SEPARATOR)
    log.info("  UFC DFS Odds API Lineup Composition Analysis")
    log.info(SEPARATOR)

    conn = sqlite3.connect(DB_PATH)

    all_results = {}

    # ---- Part A: Fighter-Level ----
    fighter_df = load_fighter_odds(conn)
    all_results["part_a"] = {}
    all_results["part_a"]["performance_by_prob_band"] = analyze_performance_by_prob_band(fighter_df)
    all_results["part_a"]["outscore_rate"] = analyze_outscore_rate(fighter_df)
    all_results["part_a"]["line_movement"] = analyze_line_movement(fighter_df)
    all_results["part_a"]["cross_tabs"] = analyze_cross_tabs(fighter_df)

    # Summary stats for Part A
    all_results["part_a"]["summary"] = {
        "total_fighter_events": int(len(fighter_df)),
        "unique_dates": int(fighter_df["date_id"].nunique()),
        "unique_fighters": int(fighter_df["canonical_id"].nunique()),
        "pct_with_open_prob": round(float(fighter_df["open_prob"].notna().mean()), 4),
        "avg_close_prob": round(float(fighter_df["close_prob"].mean()), 4),
    }

    # ---- Part B: Lineup Composition ----
    lineup_odds_df = compute_lineup_odds_batched(conn)

    conn.close()

    if not lineup_odds_df.empty:
        all_results["part_b"] = analyze_lineup_composition(lineup_odds_df)

        # Coverage summary
        all_results["part_b"]["coverage_summary"] = {
            "total_lineups_processed": int(len(lineup_odds_df)),
            "lineups_full_coverage": int((lineup_odds_df["odds_coverage"] == 6).sum()),
            "pct_full_coverage": round(
                float((lineup_odds_df["odds_coverage"] == 6).mean()), 4
            ),
            "dates_with_odds": int(lineup_odds_df["date_id"].nunique()),
            "contests_with_odds": int(lineup_odds_df["contest_id"].nunique()),
            "coverage_distribution": {
                str(k): int(v)
                for k, v in lineup_odds_df["odds_coverage"].value_counts().sort_index().items()
            },
        }
    else:
        all_results["part_b"] = {"error": "No lineup data for odds-covered dates"}

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "odds_composition_results.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=json_safe)

    log.info(f"\n{SEPARATOR}")
    log.info(f"  Results saved to {output_path}")
    log.info(SEPARATOR)


if __name__ == "__main__":
    main()
