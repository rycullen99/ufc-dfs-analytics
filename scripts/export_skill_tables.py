"""
Export ROI tables for skill updates.

Runs the full discovery pipeline and formats results as markdown tables
matching the lineup-review skill's structure. Outputs two files:
- fight_count_rules_YYYY-MM-DD.md: Fight-count-specific rules
- skill_tables_YYYY-MM-DD.md: Standard cross-tabs for all sections

Usage:
    python export_skill_tables.py
"""

from datetime import datetime

from src.phase6_backtesting.loader import load_lineups
from src.phase6_backtesting.discover import run_all


def _df_to_md(df, max_rows=100):
    """Convert a DataFrame to a markdown table string."""
    if df is None or df.empty:
        return "(no data)\n"
    df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if abs(v) < 100 else f"{v:,.0f}")
            elif isinstance(v, int):
                vals.append(f"{v:,}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    print("Loading lineups...")
    df = load_lineups(well_scraped_only=True)
    print(f"  {len(df):,} lineups loaded")

    print("Running all analyses...")
    results = run_all(df)

    today = datetime.now().strftime("%Y-%m-%d")

    # ── Fight-count rules ────────────────────────────────────────────────
    fc_lines = [f"# Fight-Count-Specific Rules — {today}\n"]
    fc_lines.append(f"Lineups: {len(df):,} | Contests: {df['contest_id'].nunique()}\n")

    fight_count_analyses = [
        ("Favorite Count by Fight Count", "fav_count_by_fights"),
        ("Ownership by Fight Count", "ownership_by_fights"),
        ("Salary by Fight Count", "salary_by_fights"),
        ("Duplication by Fight Count", "duplication_by_fights"),
        ("Favorite Count by Fight Count × Archetype", "fav_count_by_fights_arch"),
        ("Ownership by Fight Count × Archetype", "ownership_by_fights_arch"),
    ]

    for title, key in fight_count_analyses:
        fc_lines.append(f"\n## {title}\n")
        fc_lines.append(_df_to_md(results.get(key)))

    fc_path = f"reports/fight_count_rules_{today}.md"
    with open(fc_path, "w") as f:
        f.write("\n".join(fc_lines))
    print(f"  Saved: {fc_path}")

    # ── Standard skill tables ────────────────────────────────────────────
    st_lines = [f"# Skill Tables (Standard Cross-Tabs) — {today}\n"]

    standard_analyses = [
        ("Ownership (Overall)", "ownership"),
        ("Ownership by Slate Size", "ownership_by_slate"),
        ("Ownership by Entry Fee", "ownership_by_fee"),
        ("Ownership by Contest Family", "ownership_by_family"),
        ("Salary Remaining", "salary_remaining"),
        ("Favorite Count", "favorite_count"),
        ("Favorite Count by Slate", "fav_count_by_slate"),
        ("Favorite Count by Fee", "fav_count_by_fee"),
        ("Favorite Count by Family", "fav_count_by_family"),
        ("Implied Prob Sum", "implied_prob_sum"),
        ("Own/Prob Ratio", "own_prob_ratio"),
        ("Tossup Count", "tossup_count"),
        ("Duplication", "duplication"),
        ("Contest Family", "contest_family"),
    ]

    for title, key in standard_analyses:
        st_lines.append(f"\n## {title}\n")
        data = results.get(key)
        if isinstance(data, dict):
            for sub_key, sub_df in data.items():
                st_lines.append(f"\n### {sub_key}\n")
                st_lines.append(_df_to_md(sub_df))
        else:
            st_lines.append(_df_to_md(data))

    st_path = f"reports/skill_tables_{today}.md"
    with open(st_path, "w") as f:
        f.write("\n".join(st_lines))
    print(f"  Saved: {st_path}")


if __name__ == "__main__":
    main()
