"""
UFC DFS Contest Archetype Analysis

Classifies contests into 5 archetypes (SE, LIMITED, 20-Max, 150-Max, UNLIMITED)
and mines ROI across all dimensions: ownership, salary, duplication, scoring,
fighter composition, and more.

Output: formatted tables to console + JSON to data/analysis/archetype_results.json
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


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classify_contest(multi_entry_max, entry_cost, contest_size):
    """Classify a contest into one of 5 archetypes."""
    if multi_entry_max == 1:
        return "SE"
    elif multi_entry_max is not None and 3 <= multi_entry_max <= 18:
        return "LIMITED"
    elif multi_entry_max == 20:
        return "20-Max"
    elif multi_entry_max == 150:
        return "150-Max"
    else:  # NULL = unlimited
        return "UNLIMITED"


def effective_entries(multi_entry_max, contest_size):
    """Compute effective entries ratio."""
    max_e = multi_entry_max if multi_entry_max is not None else 999
    return min(max_e, contest_size) / contest_size


def entry_fee_tier(entry_cost):
    """Classify UNLIMITED contests by entry fee."""
    if entry_cost <= 3:
        return "MICRO ($1-3)"
    elif entry_cost <= 8:
        return "LOW ($5-8)"
    elif entry_cost <= 20:
        return "MID ($15-20)"
    else:
        return "HIGH ($25-30)"


def field_size_bucket(contest_size):
    """Classify by field size."""
    if contest_size < 5000:
        return "Small (<5K)"
    elif contest_size < 20000:
        return "Medium (5K-20K)"
    elif contest_size < 50000:
        return "Large (20K-50K)"
    else:
        return "Mega (50K+)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_contests(conn) -> pd.DataFrame:
    """Load all contests excluding freeroll."""
    df = pd.read_sql_query(f"""
        SELECT contest_id, date_id, entry_cost, contest_size,
               multi_entry_max, cash_line, total_prizes
        FROM contests
        WHERE contest_id != {FREEROLL_ID}
    """, conn)

    df["archetype"] = df.apply(
        lambda r: classify_contest(r["multi_entry_max"], r["entry_cost"], r["contest_size"]),
        axis=1,
    )
    df["effective_entries"] = df.apply(
        lambda r: effective_entries(r["multi_entry_max"], r["contest_size"]),
        axis=1,
    )
    df["fee_tier"] = df["entry_cost"].apply(entry_fee_tier)
    df["field_bucket"] = df["contest_size"].apply(field_size_bucket)

    log.info(f"Loaded {len(df)} contests (excl freeroll)")
    log.info(f"Archetype distribution:\n{df['archetype'].value_counts().to_string()}\n")
    return df


def load_lineups(conn) -> pd.DataFrame:
    """Load all lineups for non-freeroll contests."""
    log.info("Loading lineups (this may take a moment)...")
    df = pd.read_sql_query(f"""
        SELECT l.id as lineup_id, l.contest_id, l.lineup_rank, l.points,
               l.total_salary, l.total_ownership, l.min_ownership, l.max_ownership,
               l.is_cashing, l.payout, l.lineup_percentile, l.lineup_count,
               l.favorite_count, l.underdog_count
        FROM lineups l
        WHERE l.contest_id != {FREEROLL_ID}
    """, conn)
    log.info(f"Loaded {len(df):,} lineups")
    return df


def compute_salary_stats(conn) -> pd.DataFrame:
    """
    Compute per-lineup salary spread and $8.5K+ fighter count
    via SQL to avoid loading all 12M lineup_players rows at once.
    """
    log.info("Computing per-lineup salary stats from lineup_players...")

    df = pd.read_sql_query(f"""
        SELECT
            lp.lineup_id,
            MAX(p.salary) - MIN(p.salary) AS salary_spread,
            SUM(CASE WHEN p.salary >= 8500 THEN 1 ELSE 0 END) AS high_salary_count
        FROM lineup_players lp
        JOIN players p ON lp.player_id = p.player_id
            AND p.contest_id = (
                SELECT contest_id FROM lineups WHERE id = lp.lineup_id
            )
        JOIN lineups l ON l.id = lp.lineup_id
        WHERE l.contest_id != {FREEROLL_ID}
        GROUP BY lp.lineup_id
    """, conn)

    log.info(f"Computed salary stats for {len(df):,} lineups")
    return df


def compute_salary_stats_batched(conn) -> pd.DataFrame:
    """
    Compute per-lineup salary spread and $8.5K+ fighter count
    by processing one contest_id at a time to avoid massive joins.
    """
    log.info("Computing per-lineup salary stats (batched by contest)...")

    contest_ids = pd.read_sql_query(f"""
        SELECT DISTINCT contest_id FROM contests WHERE contest_id != {FREEROLL_ID}
    """, conn)["contest_id"].tolist()

    chunks = []
    for i, cid in enumerate(contest_ids):
        if (i + 1) % 50 == 0:
            log.info(f"  Processing contest {i+1}/{len(contest_ids)}...")

        chunk = pd.read_sql_query(f"""
            SELECT
                lp.lineup_id,
                MAX(p.salary) - MIN(p.salary) AS salary_spread,
                SUM(CASE WHEN p.salary >= 8500 THEN 1 ELSE 0 END) AS high_salary_count
            FROM lineup_players lp
            JOIN lineups l ON l.id = lp.lineup_id AND l.contest_id = {cid}
            JOIN players p ON lp.player_id = p.player_id AND p.contest_id = {cid}
            GROUP BY lp.lineup_id
        """, conn)
        if not chunk.empty:
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["lineup_id", "salary_spread", "high_salary_count"])
    log.info(f"Computed salary stats for {len(df):,} lineups")
    return df


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def compute_roi(group_df, contests_df):
    """
    Compute ROI for a group of lineups.
    ROI = avg_payout / entry_cost, computed per contest then averaged.
    Returns weighted-average ROI across contests.
    """
    if "entry_cost" in group_df.columns:
        merged = group_df
    else:
        merged = group_df.merge(contests_df[["contest_id", "entry_cost"]], on="contest_id", how="left")

    if merged.empty or merged["entry_cost"].isna().all():
        return np.nan

    # Per-contest ROI, then average
    per_contest = merged.groupby("contest_id").agg(
        avg_payout=("payout", "mean"),
        entry_cost=("entry_cost", "first"),
        n_lineups=("payout", "count"),
    )
    per_contest["roi"] = per_contest["avg_payout"] / per_contest["entry_cost"]
    # Weight by number of lineups in that contest
    total = per_contest["n_lineups"].sum()
    if total == 0:
        return np.nan
    return (per_contest["roi"] * per_contest["n_lineups"]).sum() / total


def roi_by_band(lineups_df, contests_df, band_col, band_name):
    """Compute ROI for each value of band_col, return dict."""
    results = {}
    for band_val, grp in lineups_df.groupby(band_col, observed=False):
        roi = compute_roi(grp, contests_df)
        n = len(grp)
        cash_rate = grp["is_cashing"].mean() if "is_cashing" in grp.columns else np.nan
        avg_pts = grp["points"].mean()
        results[str(band_val)] = {
            "roi": round(roi, 4) if not np.isnan(roi) else None,
            "n_lineups": int(n),
            "cash_rate": round(cash_rate, 4) if not np.isnan(cash_rate) else None,
            "avg_points": round(avg_pts, 2) if not np.isnan(avg_pts) else None,
        }
    return results


def print_roi_table(results, title, col_label="Band"):
    """Print a formatted ROI table."""
    log.info(f"\n{SUBSEP}")
    log.info(f"  {title}")
    log.info(SUBSEP)
    log.info(f"  {col_label:<25s} {'ROI':>8s} {'Cash%':>8s} {'AvgPts':>8s} {'N':>10s}")
    log.info(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for band, vals in results.items():
        roi_str = f"{vals['roi']:.4f}" if vals["roi"] is not None else "N/A"
        cash_str = f"{vals['cash_rate']:.1%}" if vals["cash_rate"] is not None else "N/A"
        pts_str = f"{vals['avg_points']:.1f}" if vals["avg_points"] is not None else "N/A"
        log.info(f"  {band:<25s} {roi_str:>8s} {cash_str:>8s} {pts_str:>8s} {vals['n_lineups']:>10,d}")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_ownership_roi(lineups_df, contests_df, archetype):
    """ROI by total ownership band."""
    df = lineups_df.copy()
    df["own_band"] = pd.cut(
        df["total_ownership"],
        bins=[0, 100, 125, 150, 175, 200, 9999],
        labels=["<100%", "100-125%", "125-150%", "150-175%", "175-200%", "200%+"],
        right=True,
    )
    results = roi_by_band(df, contests_df, "own_band", "Ownership Band")
    print_roi_table(results, f"ROI by Total Ownership — {archetype}", "Ownership Band")
    return results


def analyze_salary_remaining_roi(lineups_df, contests_df, archetype):
    """ROI by salary remaining."""
    df = lineups_df.copy()
    df["salary_remaining"] = 50000 - df["total_salary"]
    df["sal_rem_band"] = pd.cut(
        df["salary_remaining"],
        bins=[-1, 500, 1000, 2000, 99999],
        labels=["<$500", "$500-999", "$1K-2K", "$2K+"],
    )
    results = roi_by_band(df, contests_df, "sal_rem_band", "Salary Remaining")
    print_roi_table(results, f"ROI by Salary Remaining — {archetype}", "Salary Remaining")
    return results


def analyze_salary_spread_roi(lineups_df, contests_df, archetype):
    """ROI by salary spread (max - min salary of 6 fighters)."""
    df = lineups_df.copy()
    if "salary_spread" not in df.columns:
        log.info(f"  [SKIP] salary_spread not available for {archetype}")
        return {}
    df["spread_band"] = pd.cut(
        df["salary_spread"],
        bins=[-1, 1000, 2000, 3000, 4000, 5000, 99999],
        labels=["<$1K", "$1K-2K", "$2K-3K", "$3K-4K", "$4K-5K", "$5K+"],
    )
    results = roi_by_band(df, contests_df, "spread_band", "Salary Spread")
    print_roi_table(results, f"ROI by Salary Spread — {archetype}", "Salary Spread")
    return results


def analyze_min_ownership_roi(lineups_df, contests_df, archetype):
    """ROI by min ownership fighter."""
    df = lineups_df.copy()
    df["min_own_band"] = pd.cut(
        df["min_ownership"],
        bins=[-0.1, 2, 5, 8, 12, 18, 100],
        labels=["0-2%", "2-5%", "5-8%", "8-12%", "12-18%", "18%+"],
    )
    results = roi_by_band(df, contests_df, "min_own_band", "Min Ownership")
    print_roi_table(results, f"ROI by Min Ownership Fighter — {archetype}", "Min Ownership")
    return results


def analyze_max_ownership_roi(lineups_df, contests_df, archetype):
    """ROI by max ownership fighter."""
    df = lineups_df.copy()
    df["max_own_band"] = pd.cut(
        df["max_ownership"],
        bins=[0, 20, 30, 40, 50, 60, 100],
        labels=["<20%", "20-30%", "30-40%", "40-50%", "50-60%", "60%+"],
    )
    results = roi_by_band(df, contests_df, "max_own_band", "Max Ownership")
    print_roi_table(results, f"ROI by Max Ownership Fighter — {archetype}", "Max Ownership")
    return results


def analyze_lineup_count_roi(lineups_df, contests_df, archetype):
    """ROI by lineup duplication count."""
    df = lineups_df.copy()
    df["dup_band"] = pd.cut(
        df["lineup_count"].fillna(1),
        bins=[0, 1, 5, 20, 100, 999999],
        labels=["1 (unique)", "2-5", "6-20", "21-100", "101+"],
        right=True,
    )
    results = roi_by_band(df, contests_df, "dup_band", "Lineup Count")
    print_roi_table(results, f"ROI by Lineup Duplication — {archetype}", "Duplication")
    return results


def analyze_high_salary_count_roi(lineups_df, contests_df, archetype):
    """ROI by count of $8.5K+ fighters."""
    df = lineups_df.copy()
    if "high_salary_count" not in df.columns:
        log.info(f"  [SKIP] high_salary_count not available for {archetype}")
        return {}
    results = roi_by_band(df, contests_df, "high_salary_count", "$8.5K+ Count")
    print_roi_table(results, f"ROI by $8.5K+ Fighter Count — {archetype}", "$8.5K+ Fighters")
    return results


def analyze_favorite_count_roi(lineups_df, contests_df, archetype):
    """ROI by favorite count."""
    df = lineups_df.copy()
    df["fav_count"] = df["favorite_count"].fillna(0).astype(int)
    results = roi_by_band(df, contests_df, "fav_count", "Favorite Count")
    print_roi_table(results, f"ROI by Favorite Count — {archetype}", "Favorites")
    return results


def analyze_scoring_thresholds(lineups_df, contests_df, archetype):
    """Compute scoring thresholds by archetype."""
    merged = lineups_df.merge(
        contests_df[["contest_id", "entry_cost", "contest_size", "cash_line"]],
        on="contest_id", how="left",
    )

    avg_pts = merged["points"].mean()

    # Cash line points: average points at the cash line rank
    cash_pts_list = []
    for cid, grp in merged.groupby("contest_id"):
        cl = grp["cash_line"].iloc[0]
        if pd.notna(cl) and cl > 0:
            cash_lineup = grp[grp["lineup_rank"] <= cl].tail(1)
            if not cash_lineup.empty:
                cash_pts_list.append(cash_lineup["points"].iloc[0])
    cash_line_pts = np.mean(cash_pts_list) if cash_pts_list else np.nan

    # Top 5% and top 1% points
    top5_list = []
    top1_list = []
    winner_list = []
    for cid, grp in merged.groupby("contest_id"):
        n = len(grp)
        top5_rank = max(1, int(n * 0.05))
        top1_rank = max(1, int(n * 0.01))
        sorted_grp = grp.sort_values("lineup_rank")
        top5_list.append(sorted_grp.iloc[min(top5_rank - 1, n - 1)]["points"])
        top1_list.append(sorted_grp.iloc[min(top1_rank - 1, n - 1)]["points"])
        winner_list.append(sorted_grp.iloc[0]["points"])

    results = {
        "avg_points": round(avg_pts, 2) if not np.isnan(avg_pts) else None,
        "cash_line_points": round(np.mean(cash_pts_list), 2) if cash_pts_list else None,
        "top_5pct_points": round(np.mean(top5_list), 2) if top5_list else None,
        "top_1pct_points": round(np.mean(top1_list), 2) if top1_list else None,
        "winner_points": round(np.mean(winner_list), 2) if winner_list else None,
        "n_contests": len(merged["contest_id"].unique()),
    }

    log.info(f"\n{SUBSEP}")
    log.info(f"  Scoring Thresholds — {archetype}")
    log.info(SUBSEP)
    for k, v in results.items():
        label = k.replace("_", " ").title()
        val_str = f"{v:.2f}" if v is not None else "N/A"
        log.info(f"  {label:<25s} {val_str:>10s}")

    return results


# ---------------------------------------------------------------------------
# UNLIMITED sub-splits
# ---------------------------------------------------------------------------

def analyze_unlimited_splits(lineups_df, contests_df):
    """Within UNLIMITED, split by fee tier and field size bucket."""
    results = {"by_fee_tier": {}, "by_field_bucket": {}}

    log.info(f"\n{SEPARATOR}")
    log.info("  UNLIMITED — Sub-Splits")
    log.info(SEPARATOR)

    # By fee tier
    for tier, cgrp in contests_df.groupby("fee_tier"):
        tier_cids = cgrp["contest_id"].tolist()
        tier_lineups = lineups_df[lineups_df["contest_id"].isin(tier_cids)]
        if tier_lineups.empty:
            continue
        roi = compute_roi(tier_lineups, cgrp)
        cash_rate = tier_lineups["is_cashing"].mean()
        avg_pts = tier_lineups["points"].mean()
        results["by_fee_tier"][tier] = {
            "roi": round(roi, 4) if not np.isnan(roi) else None,
            "n_lineups": int(len(tier_lineups)),
            "n_contests": int(len(cgrp)),
            "cash_rate": round(cash_rate, 4),
            "avg_points": round(avg_pts, 2),
        }

    log.info(f"\n  By Entry Fee Tier:")
    log.info(f"  {'Tier':<20s} {'ROI':>8s} {'Cash%':>8s} {'AvgPts':>8s} {'Contests':>10s} {'Lineups':>12s}")
    log.info(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")
    for tier, vals in sorted(results["by_fee_tier"].items()):
        roi_str = f"{vals['roi']:.4f}" if vals["roi"] is not None else "N/A"
        log.info(f"  {tier:<20s} {roi_str:>8s} {vals['cash_rate']:.1%}  {vals['avg_points']:>7.1f} {vals['n_contests']:>10d} {vals['n_lineups']:>12,d}")

    # By field size bucket
    for bucket, cgrp in contests_df.groupby("field_bucket"):
        bucket_cids = cgrp["contest_id"].tolist()
        bucket_lineups = lineups_df[lineups_df["contest_id"].isin(bucket_cids)]
        if bucket_lineups.empty:
            continue
        roi = compute_roi(bucket_lineups, cgrp)
        cash_rate = bucket_lineups["is_cashing"].mean()
        avg_pts = bucket_lineups["points"].mean()
        results["by_field_bucket"][bucket] = {
            "roi": round(roi, 4) if not np.isnan(roi) else None,
            "n_lineups": int(len(bucket_lineups)),
            "n_contests": int(len(cgrp)),
            "cash_rate": round(cash_rate, 4),
            "avg_points": round(avg_pts, 2),
        }

    log.info(f"\n  By Field Size Bucket:")
    log.info(f"  {'Bucket':<20s} {'ROI':>8s} {'Cash%':>8s} {'AvgPts':>8s} {'Contests':>10s} {'Lineups':>12s}")
    log.info(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")
    for bucket, vals in sorted(results["by_field_bucket"].items()):
        roi_str = f"{vals['roi']:.4f}" if vals["roi"] is not None else "N/A"
        log.info(f"  {bucket:<20s} {roi_str:>8s} {vals['cash_rate']:.1%}  {vals['avg_points']:>7.1f} {vals['n_contests']:>10d} {vals['n_lineups']:>12,d}")

    return results


# ---------------------------------------------------------------------------
# Effective entries analysis
# ---------------------------------------------------------------------------

def analyze_effective_entries(lineups_df, contests_df):
    """ROI by effective entries bands."""
    log.info(f"\n{SEPARATOR}")
    log.info("  Effective Entries Analysis")
    log.info(SEPARATOR)

    merged = lineups_df.merge(
        contests_df[["contest_id", "effective_entries", "entry_cost"]],
        on="contest_id", how="left",
    )

    merged["ee_band"] = pd.cut(
        merged["effective_entries"],
        bins=[0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.01],
        labels=["<0.1%", "0.1-1%", "1-5%", "5-10%", "10-50%", "50%+"],
    )

    results = roi_by_band(merged, contests_df, "ee_band", "Effective Entries")
    print_roi_table(results, "ROI by Effective Entries", "Eff. Entries")
    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_archetype_analysis(lineups_df, contests_df, archetype, salary_stats_df=None):
    """Run all analysis dimensions for a single archetype."""
    arch_contests = contests_df[contests_df["archetype"] == archetype]
    arch_cids = arch_contests["contest_id"].tolist()
    arch_lineups = lineups_df[lineups_df["contest_id"].isin(arch_cids)].copy()

    if arch_lineups.empty:
        log.info(f"\n  No lineups for archetype {archetype}, skipping.")
        return {}

    # Merge salary stats if available
    if salary_stats_df is not None and not salary_stats_df.empty:
        arch_lineups = arch_lineups.merge(salary_stats_df, on="lineup_id", how="left")

    log.info(f"\n{SEPARATOR}")
    log.info(f"  ARCHETYPE: {archetype}")
    log.info(f"  Contests: {len(arch_contests):,}  |  Lineups: {len(arch_lineups):,}")
    log.info(SEPARATOR)

    results = {
        "n_contests": int(len(arch_contests)),
        "n_lineups": int(len(arch_lineups)),
        "ownership_roi": analyze_ownership_roi(arch_lineups, arch_contests, archetype),
        "salary_remaining_roi": analyze_salary_remaining_roi(arch_lineups, arch_contests, archetype),
        "salary_spread_roi": analyze_salary_spread_roi(arch_lineups, arch_contests, archetype),
        "min_ownership_roi": analyze_min_ownership_roi(arch_lineups, arch_contests, archetype),
        "max_ownership_roi": analyze_max_ownership_roi(arch_lineups, arch_contests, archetype),
        "lineup_count_roi": analyze_lineup_count_roi(arch_lineups, arch_contests, archetype),
        "high_salary_count_roi": analyze_high_salary_count_roi(arch_lineups, arch_contests, archetype),
        "favorite_count_roi": analyze_favorite_count_roi(arch_lineups, arch_contests, archetype),
        "scoring_thresholds": analyze_scoring_thresholds(arch_lineups, arch_contests, archetype),
    }

    return results


def main():
    log.info(SEPARATOR)
    log.info("  UFC DFS Contest Archetype Analysis")
    log.info(SEPARATOR)

    conn = sqlite3.connect(DB_PATH)

    # Load data
    contests_df = load_contests(conn)
    lineups_df = load_lineups(conn)

    # Compute salary stats (batched for performance)
    salary_stats_df = compute_salary_stats_batched(conn)

    conn.close()

    # Fill payout NaN with 0 (non-cashing lineups)
    lineups_df["payout"] = lineups_df["payout"].fillna(0)

    # Run per-archetype analysis
    all_results = {}
    archetypes = ["UNLIMITED", "LIMITED", "SE", "20-Max", "150-Max"]

    for arch in archetypes:
        all_results[arch] = run_archetype_analysis(lineups_df, contests_df, arch, salary_stats_df)

    # UNLIMITED sub-splits
    log.info("\n")
    unlimited_contests = contests_df[contests_df["archetype"] == "UNLIMITED"]
    unlimited_cids = unlimited_contests["contest_id"].tolist()
    unlimited_lineups = lineups_df[lineups_df["contest_id"].isin(unlimited_cids)]
    all_results["UNLIMITED"]["sub_splits"] = analyze_unlimited_splits(
        unlimited_lineups, unlimited_contests
    )

    # Effective entries (across all archetypes)
    all_results["effective_entries"] = analyze_effective_entries(lineups_df, contests_df)

    # Summary table across archetypes
    log.info(f"\n{SEPARATOR}")
    log.info("  SUMMARY — All Archetypes")
    log.info(SEPARATOR)
    log.info(f"  {'Archetype':<15s} {'Contests':>10s} {'Lineups':>12s} {'AvgPts':>8s} {'CashLine':>10s} {'Winner':>8s}")
    log.info(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
    for arch in archetypes:
        r = all_results.get(arch, {})
        st = r.get("scoring_thresholds", {})
        n_c = r.get("n_contests", 0)
        n_l = r.get("n_lineups", 0)
        avg = st.get("avg_points", None)
        cl = st.get("cash_line_points", None)
        win = st.get("winner_points", None)
        avg_s = f"{avg:.1f}" if avg is not None else "N/A"
        cl_s = f"{cl:.1f}" if cl is not None else "N/A"
        win_s = f"{win:.1f}" if win is not None else "N/A"
        log.info(f"  {arch:<15s} {n_c:>10,d} {n_l:>12,d} {avg_s:>8s} {cl_s:>10s} {win_s:>8s}")

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "archetype_results.json"

    # Convert any numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    log.info(f"\nResults saved to {output_path}")
    log.info(SEPARATOR)


if __name__ == "__main__":
    main()
