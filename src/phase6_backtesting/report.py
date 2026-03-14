"""
Enhanced ROI report generator (Phase 6).

Produces a markdown report with:
- Point estimates + 95% bootstrap CIs + p-values
- FDR correction summary
- Time-split validation on priority rules
- Contest family breakdown
"""

import pandas as pd
from datetime import datetime

from .loader import load_lineups, weighted_roi
from .discover import run_all, run_all_with_ci
from .validation import validate_rule
from ..config import REPORTS_DIR


# Priority rules to time-split validate
PRIORITY_RULES = [
    ("salary_remaining_band", "$0 (maxed)"),
    ("total_own_band", "100-125%"),
    ("fav_count_band", "4"),
    ("fav_count_band", "5"),
    ("own_prob_band", "<0.50"),
    ("dupe_band", "1 (unique)"),
]


def generate_report(fdr_q: float = 0.10) -> str:
    """
    Generate the full enhanced ROI report.

    Returns markdown string and saves to reports/ directory.
    """
    print("Loading lineups...")
    df = load_lineups(well_scraped_only=True)
    print(f"  {len(df):,} lineups loaded across {df['contest_date'].nunique()} dates")

    lines = []
    lines.append(f"# UFC DFS ROI Validation Report (Enhanced) — {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")

    # ── Overview ──────────────────────────────────────────────────────────
    overall_roi = weighted_roi(df)
    total_entries = int(df["user_count"].sum())
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Lineups analyzed | {len(df):,} |")
    lines.append(f"| Total entries (weighted) | {total_entries:,} |")
    lines.append(f"| Overall weighted ROI | {overall_roi:.4f}x |")
    lines.append(f"| Contests | {df['contest_id'].nunique()} |")
    lines.append(f"| Date range | {df['contest_date'].min()} → {df['contest_date'].max()} |")
    lines.append(f"| FDR q-value | {fdr_q} |")
    lines.append("")

    # ── Run all analyses with CIs ─────────────────────────────────────────
    print("Running bootstrap CIs + FDR correction...")
    ci_results = run_all_with_ci(df, fdr_q=fdr_q)

    # FDR summary
    fdr = ci_results.get("_fdr_summary", {})
    lines.append("## FDR Correction Summary")
    lines.append("")
    lines.append(f"- **Total hypothesis tests:** {fdr.get('total_tests', 0)}")
    lines.append(f"- **Surviving FDR (q={fdr_q}):** {fdr.get('surviving', 0)}")
    lines.append(f"- **Rejected:** {fdr.get('rejected', 0)}")
    if fdr.get("max_surviving_p") is not None:
        lines.append(f"- **Max surviving p-value:** {fdr['max_surviving_p']:.4f}")
    lines.append("")

    # ── Per-analysis tables ───────────────────────────────────────────────
    for name, groups in ci_results.items():
        if name.startswith("_") or not isinstance(groups, dict):
            continue
        # Skip if no bootstrap results (just DataFrames)
        first_val = next(iter(groups.values()), None)
        if not isinstance(first_val, dict) or "point_estimate" not in first_val:
            continue

        lines.append(f"## {_format_name(name)}")
        lines.append("")
        lines.append("| Bucket | n_lineups | n_entries | ROI | 95% CI | p-value | FDR |")
        lines.append("|--------|-----------|-----------|-----|--------|---------|-----|")

        for bucket, res in groups.items():
            roi = f"{res['point_estimate']:.4f}x"
            ci = f"[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]"
            p = f"{res['p_value']:.4f}"
            fdr_flag = "**YES**" if res.get("survives_fdr") else "no"
            n_lin = f"{res.get('n_lineups', res['n_samples']):,}"
            n_ent = f"{res.get('n_entries', ''):,}" if res.get("n_entries") else ""
            lines.append(f"| {bucket} | {n_lin} | {n_ent} | {roi} | {ci} | {p} | {fdr_flag} |")

        lines.append("")

    # ── Point-estimate tables (cross-tabs) ────────────────────────────────
    print("Running point-estimate cross-tabs...")
    point_results = run_all(df)

    for name in ["ownership_by_slate", "ownership_by_fee", "ownership_by_family",
                  "fav_count_by_slate", "fav_count_by_fee", "fav_count_by_family"]:
        if name in point_results:
            tbl = point_results[name]
            lines.append(f"## {_format_name(name)} (Point Estimates)")
            lines.append("")
            lines.append(_df_to_markdown(tbl))
            lines.append("")

    # ── Time-split validation ─────────────────────────────────────────────
    print("Running time-split validation on priority rules...")
    lines.append("## Time-Split Validation (60/40 Chronological)")
    lines.append("")
    lines.append("| Rule | Train ROI | Train CI | Test ROI | Test CI | Holds? |")
    lines.append("|------|-----------|----------|----------|---------|--------|")

    for group_col, target in PRIORITY_RULES:
        result = validate_rule(df, "contest_date", group_col, target)
        train = result.get("train", {})
        test = result.get("test", {})

        t_roi = f"{train['roi']:.3f}x" if train.get("roi") else "n/a"
        t_ci = f"[{train.get('ci_lower', 0):.3f}, {train.get('ci_upper', 0):.3f}]" if train.get("roi") else ""
        s_roi = f"{test['roi']:.3f}x" if test.get("roi") else "n/a"
        s_ci = f"[{test.get('ci_lower', 0):.3f}, {test.get('ci_upper', 0):.3f}]" if test.get("roi") else ""
        holds = "**YES**" if result.get("holds") else "no"

        lines.append(f"| {group_col} = {target} | {t_roi} | {t_ci} | {s_roi} | {s_ci} | {holds} |")

    lines.append("")

    # ── Save ──────────────────────────────────────────────────────────────
    report = "\n".join(lines)
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = REPORTS_DIR / f"roi_validation_enhanced_{today}.md"
    out_path.write_text(report)
    print(f"\nReport saved to {out_path}")

    return report


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without tabulate dependency."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def _format_name(name: str) -> str:
    """Convert snake_case analysis name to Title Case."""
    return name.replace("_", " ").title()


if __name__ == "__main__":
    generate_report()
