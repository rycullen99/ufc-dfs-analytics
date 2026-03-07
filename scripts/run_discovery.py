"""
Run the full Phase 6 ROI discovery and write reports.

Usage:
    uv run scripts/run_discovery.py
    uv run scripts/run_discovery.py --fee-only
    uv run scripts/run_discovery.py --slate STANDARD
"""

import argparse
import sys
from datetime import date
from pathlib import Path

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.phase6_backtesting.loader import load_lineups, weighted_roi
from src.phase6_backtesting.discover import run_all


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _roi_bar(roi: float, scale: float = 3.0) -> str:
    """Visual bar proportional to ROI — makes tables scannable at a glance."""
    filled = min(int((roi / scale) * 20), 20)
    empty = 20 - filled
    marker = "●" * filled + "○" * empty
    return f"{roi:.2f}x  {marker}"


def _section(title: str, df: pd.DataFrame, dim_col: str) -> list[str]:
    lines = [f"\n## {title}\n"]
    if df.empty:
        lines.append("_No data_\n")
        return lines

    header = f"| {dim_col:<20} | n_lineups | n_entries | ROI | ROI bar | top-1% |"
    sep    = f"|{'-'*22}|-----------|-----------|-----|---------|--------|"
    lines += [header, sep]

    for _, row in df.iterrows():
        roi = row["weighted_roi"]
        bar = _roi_bar(roi)
        lines.append(
            f"| {str(row[dim_col]):<20} "
            f"| {row['n_lineups']:>9,} "
            f"| {row['n_entries']:>9,} "
            f"| {roi:.2f}x "
            f"| {bar} "
            f"| {row['top1pct_rate']*100:.2f}% |"
        )
    return lines


def _cross_section(title: str, df: pd.DataFrame, regime_col: str, dim_col: str) -> list[str]:
    """Two-level grouping: regime rows → ownership bands as columns."""
    lines = [f"\n## {title}\n"]
    if df.empty:
        lines.append("_No data_\n")
        return lines

    # Pivot: regime as rows, ownership band as columns, ROI as values
    pivot = df.pivot_table(
        index=regime_col, columns=dim_col, values="weighted_roi", aggfunc="first"
    )
    lines.append(pivot.to_markdown(floatfmt=".2f"))
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame, results: dict) -> str:
    today = date.today().isoformat()
    n_lineups = len(df)
    n_entries = int(df["user_count"].sum())
    overall_roi = weighted_roi(df)

    lines = [
        f"# UFC DFS ROI Validation Report — {today}",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Lineups analyzed | {n_lineups:,} |",
        f"| Total entries (weighted) | {n_entries:,} |",
        f"| Overall weighted ROI | {overall_roi:.4f}x |",
        f"| Contests | {df['contest_id'].nunique()} |",
        f"| Date range | {df['contest_date'].min()} → {df['contest_date'].max()} |",
        "",
        "> ROI is weighted by `user_count`. True ROI < 1.0 due to DK ~13% rake.",
        "> Well-scraped contests only (payout totals within 5% of prize pool).",
        "",
    ]

    # Regime distribution
    lines += ["\n## Regime Distribution\n"]
    for col in ["slate_size", "fee_tier", "field_size"]:
        if col in df.columns:
            dist = df.groupby(col, observed=True)["user_count"].sum().reset_index()
            dist["pct"] = (dist["user_count"] / dist["user_count"].sum() * 100).round(1)
            lines.append(f"**{col}**")
            lines.append(dist.to_markdown(index=False))
            lines.append("")

    # Core dimensions
    lines += _section("Ownership Band", results["ownership"], "total_own_band")
    lines += _section("Salary Remaining", results["salary_remaining"], "salary_remaining_band")
    lines += _section("Favorite Count", results["favorite_count"], "fav_count_band")
    lines += _section("Implied Probability Sum", results["implied_prob_sum"], "prob_sum_band")
    lines += _section("Ownership / Probability Ratio", results["own_prob_ratio"], "own_prob_band")
    lines += _section("Toss-Up Fighter Count", results["tossup_count"], "tossup_label")
    lines += _section("Lineup Duplication", results["duplication"], "dupe_band")

    # Ownership composition
    comp = results["ownership_composition"]
    lines += _section("Min Ownership (Contrarian Floor)", comp["min_own"], "min_own_band")
    lines += _section("Max Ownership (Chalk Anchor)", comp["max_own"], "max_own_band")

    # Cross-tabs
    lines += _cross_section(
        "Ownership by Slate Size", results["ownership_by_slate"], "slate_size", "total_own_band"
    )
    lines += _cross_section(
        "Ownership by Entry Fee", results["ownership_by_fee"], "fee_tier", "total_own_band"
    )
    lines += _cross_section(
        "Favorite Count by Slate Size", results["fav_count_by_slate"], "slate_size", "fav_count_band"
    )
    lines += _cross_section(
        "Favorite Count by Fee", results["fav_count_by_fee"], "fee_tier", "fav_count_band"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run UFC DFS ROI discovery")
    parser.add_argument("--slate", choices=["SHORT", "STANDARD", "FULL"],
                        help="Filter to a single slate size")
    parser.add_argument("--fee", choices=["MICRO", "LOW", "MID", "HIGH"],
                        help="Filter to a single fee tier")
    args = parser.parse_args()

    print("Loading lineups...")
    df = load_lineups(well_scraped_only=True)
    print(f"  {len(df):,} lineups across {df['contest_id'].nunique()} contests")

    if args.slate:
        df = df[df["slate_size"] == args.slate]
        print(f"  Filtered to {args.slate}: {len(df):,} lineups")

    if args.fee:
        df = df[df["fee_tier"] == args.fee]
        print(f"  Filtered to {args.fee}: {len(df):,} lineups")

    print("Running discovery...")
    results = run_all(df)

    print("Building report...")
    report = build_report(df, results)

    suffix = ""
    if args.slate:
        suffix += f"_{args.slate}"
    if args.fee:
        suffix += f"_{args.fee}"
    out_path = REPORTS_DIR / f"roi_validation_{date.today().isoformat()}{suffix}.md"
    out_path.write_text(report)
    print(f"\nReport written to: {out_path}")

    # Console summary
    print("\n--- TOP RULES BY ROI ---")
    for key in ["ownership", "favorite_count", "own_prob_ratio", "tossup_count"]:
        tbl = results[key]
        if isinstance(tbl, pd.DataFrame) and not tbl.empty and "weighted_roi" in tbl.columns:
            best = tbl.loc[tbl["weighted_roi"].idxmax()]
            print(f"  {key:<25} best: {best.iloc[0]!s:<20} → {best['weighted_roi']:.2f}x ROI")


if __name__ == "__main__":
    main()
