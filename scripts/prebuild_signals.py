"""
Pre-build intelligence report for UFC DFS.

Combines sharp signals, leverage features, cash features, and fighter form
from resultsdb_ufc.db into a single fighter-level report for a given slate.

Usage:
    python -m scripts.prebuild_signals --date 2026-03-07
    python -m scripts.prebuild_signals --date 2026-03-07 --format csv
"""

import argparse
import sqlite3
from pathlib import Path

from src.config import DB_PATH


def get_report(date_str: str) -> list[dict]:
    """Pull all pre-build signals for fighters on a given date."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row

    # Get date_id for this date
    c = conn.cursor()
    c.execute(
        "SELECT DISTINCT date_id FROM contests WHERE contest_date LIKE ?",
        (f"{date_str}%",),
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return []
    date_id = row["date_id"]

    # Pull fighter roster with all available signals
    query = """
        SELECT
            p.player_id,
            p.full_name,
            p.salary,
            p.ownership,
            p.actual_points,
            fo.implied_prob,
            fo.is_favorite,
            ss.sharp_vs_field_delta,
            ss.sharp_confidence,
            ss.n_sharp_users,
            lf.leverage_score,
            lf.ceiling_per_ownership,
            lf.ownership_delta,
            lf.was_optimal_play,
            cf.cash_probability,
            cf.bust_probability,
            cf.floor_fpts,
            cf.consistency_score,
            ff.last_3_avg_fpts,
            ff.fpts_trend,
            ff.coming_off_win,
            ff.coming_off_finish
        FROM players p
        JOIN contests c ON c.contest_id = p.contest_id
        LEFT JOIN fighter_odds fo
            ON fo.player_id = p.player_id AND fo.date_id = c.date_id
        LEFT JOIN sharp_signals ss
            ON ss.player_id = p.player_id AND ss.date_id = c.date_id
        LEFT JOIN leverage_features lf
            ON lf.player_id = p.player_id AND lf.contest_id = p.contest_id
        LEFT JOIN cash_features cf
            ON cf.player_id = p.player_id AND cf.contest_id = p.contest_id
        LEFT JOIN fighter_form ff
            ON ff.dfs_player_id = p.player_id AND ff.contest_id = p.contest_id
        WHERE c.date_id = ?
        GROUP BY p.player_id
        ORDER BY p.salary DESC
    """
    rows = conn.execute(query, (date_id,)).fetchall()
    conn.close()

    fighters = []
    for r in rows:
        tag = _auto_tag(r)
        fighters.append({
            "name": r["full_name"],
            "salary": r["salary"],
            "own%": r["ownership"],
            "pts": r["actual_points"],
            "prob": r["implied_prob"],
            "fav": "Y" if r["is_favorite"] else "N" if r["is_favorite"] is not None else "",
            "sharp_delta": r["sharp_vs_field_delta"],
            "sharp_conf": r["sharp_confidence"],
            "lev_score": r["leverage_score"],
            "ceil/own": r["ceiling_per_ownership"],
            "own_delta": r["ownership_delta"],
            "cash_prob": r["cash_probability"],
            "bust_prob": r["bust_probability"],
            "floor": r["floor_fpts"],
            "l3_avg": r["last_3_avg_fpts"],
            "trend": r["fpts_trend"],
            "off_win": r["coming_off_win"],
            "off_fin": r["coming_off_finish"],
            "TAG": tag,
        })
    return fighters


def _auto_tag(r) -> str:
    """Auto-tag a fighter based on feature thresholds."""
    tags = []

    # CASH_LOCK: high cash prob, low bust, favorite
    if (r["cash_probability"] and r["cash_probability"] > 0.55
            and r["bust_probability"] and r["bust_probability"] < 0.35
            and r["is_favorite"]):
        tags.append("CASH_LOCK")

    # GPP_LEVERAGE: high leverage, low ownership, high ceiling per ownership
    if (r["leverage_score"] and r["leverage_score"] > 50
            and r["ownership"] and r["ownership"] < 15
            and r["ceiling_per_ownership"] and r["ceiling_per_ownership"] > 4.0):
        tags.append("GPP_LEV")

    # TRAP_PLAY: significantly over-owned vs expected
    if r["ownership_delta"] and r["ownership_delta"] > 10:
        tags.append("TRAP")

    # DUD_RISK: high bust probability
    if r["bust_probability"] and r["bust_probability"] > 0.45:
        tags.append("DUD_RISK")

    return " | ".join(tags) if tags else ""


def format_markdown(fighters: list[dict], date_str: str) -> str:
    """Format as markdown table."""
    lines = [f"# Pre-Build Intelligence — {date_str}", ""]
    lines.append(f"Fighters: {len(fighters)}")
    lines.append("")

    cols = ["name", "salary", "own%", "fav", "sharp_delta", "lev_score",
            "ceil/own", "cash_prob", "bust_prob", "floor", "l3_avg", "trend", "TAG"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")

    for f in fighters:
        vals = []
        for c in cols:
            v = f.get(c)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.1f}" if abs(v) < 1000 else f"{v:,.0f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pre-build intelligence report")
    parser.add_argument("--date", required=True, help="Slate date (YYYY-MM-DD)")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    args = parser.parse_args()

    fighters = get_report(args.date)
    if not fighters:
        print(f"No data found for {args.date}")
        return

    if args.format == "markdown":
        print(format_markdown(fighters, args.date))
    else:
        import csv
        import sys
        writer = csv.DictWriter(sys.stdout, fieldnames=fighters[0].keys())
        writer.writeheader()
        writer.writerows(fighters)


if __name__ == "__main__":
    main()
