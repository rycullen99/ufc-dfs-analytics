"""
UFC DFS Construction Rules — Detailed Analysis

Comprehensive breakdowns by:
  - Weight class (all 11+)
  - Men's vs Women's
  - Favorite vs Underdog
  - Salary tier
  - Card position
  - Line movement
  - Odds band
  - Ownership band
  - Finish type / scoring patterns
  - Cross-cuts of the above

Data: 10,485 fighter-events with closing odds, 8,523 with weight class.
"""

import sqlite3
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

DB_PATH = "/Users/ryancullen/Desktop/resultsdb_ufc.db"

SEP = "=" * 80
SUBSEP = "-" * 80


def pr(msg=""):
    print(msg)


def section(title):
    pr(f"\n{SEP}")
    pr(f"  {title}")
    pr(SEP)


def subsection(title):
    pr(f"\n{SUBSEP}")
    pr(f"  {title}")
    pr(SUBSEP)


def ptable(df, float_fmt=".1f"):
    """Print DataFrame as formatted table."""
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:{float_fmt}}")
    print(df.to_string(index=False))


def pct(series):
    """Convert boolean/0-1 series to percentage."""
    return (series.mean() * 100)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT
            foe.date_id,
            foe.canonical_id,
            foe.dfs_name,
            foe.salary,
            foe.actual_ownership,
            foe.actual_points,
            foe.open_prob,
            foe.close_prob,
            foe.line_move,
            foe.is_favorite,
            foe.close_n_books,
            foe.opponent_odds_name,
            foe.player_id,
            fcp.card_section,
            pf.weight_class
        FROM fighter_odds_enriched foe
        LEFT JOIN fighter_card_position fcp
            ON foe.date_id = fcp.date_id AND foe.player_id = fcp.player_id
        LEFT JOIN player_features pf
            ON foe.player_id = pf.player_id
            AND pf.contest_id IN (SELECT contest_id FROM contests WHERE date_id = foe.date_id)
        WHERE foe.player_id IS NOT NULL
          AND foe.close_prob IS NOT NULL
          AND foe.actual_points IS NOT NULL
    """, conn)

    conn.close()

    # Derived columns
    df["salary_tier"] = pd.cut(
        df["salary"],
        bins=[0, 6000, 7000, 8000, 9000, 10000, 20000],
        labels=["Under $6K", "$6K-$7K", "$7K-$8K", "$8K-$9K", "$9K-$10K", "$10K+"],
    )

    df["ownership_band"] = pd.cut(
        df["actual_ownership"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 100],
        labels=["0-5%", "5-10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-40%", "40%+"],
    )

    df["close_prob_band"] = pd.cut(
        df["close_prob"],
        bins=[0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0],
        labels=["<25%", "25-35%", "35-45%", "45-55%", "55-65%", "65-75%", "75%+"],
    )

    df["line_move_dir"] = np.where(
        df["line_move"].isna(), "no_data",
        np.where(df["line_move"] > 0.02, "steamed",
                 np.where(df["line_move"] < -0.02, "drifted", "stable"))
    )

    df["salary_value"] = df["actual_points"] / (df["salary"] / 1000)
    df["is_ceiling"] = df["salary_value"] >= 8.0
    df["is_smash"] = df["salary_value"] >= 10.0
    df["is_dud"] = df["actual_points"] < 10.0
    df["fav_dog"] = df["is_favorite"].map({1: "Favorite", 0: "Underdog"})

    # Gender
    womens = {"Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight"}
    df["gender"] = df["weight_class"].apply(lambda x: "Women's" if x in womens else ("Men's" if pd.notna(x) and x not in ("Unknown", "Catch Weight") else "Unknown"))

    # Weight class grouping for cleaner analysis
    df["weight_group"] = df["weight_class"].replace({
        "Women's Strawweight": "W-Strawweight",
        "Women's Flyweight": "W-Flyweight",
        "Women's Bantamweight": "W-Bantamweight",
        "Women's Featherweight": "W-Featherweight",
        "Catch Weight": "Other",
        "Unknown": "Other",
    })

    # Prob tier for simpler cross-cuts
    df["prob_tier"] = pd.cut(
        df["close_prob"],
        bins=[0, 0.35, 0.50, 0.65, 1.0],
        labels=["Heavy Dog (<35%)", "Slight Dog (35-50%)", "Slight Fav (50-65%)", "Heavy Fav (65%+)"],
    )

    return df


def agg_stats(group):
    """Standard aggregation for any groupby."""
    return pd.Series({
        "n": len(group),
        "avg_fpts": group["actual_points"].mean(),
        "med_fpts": group["actual_points"].median(),
        "p75_fpts": group["actual_points"].quantile(0.75),
        "p90_fpts": group["actual_points"].quantile(0.90),
        "avg_own": group["actual_ownership"].mean(),
        "avg_salary": group["salary"].mean(),
        "pts_per_1k": group["salary_value"].mean(),
        "ceil_%": pct(group["is_ceiling"]),
        "smash_%": pct(group["is_smash"]),
        "dud_%": pct(group["is_dud"]),
        "avg_prob": group["close_prob"].mean(),
    })


# ---------------------------------------------------------------------------
# Analysis Sections
# ---------------------------------------------------------------------------


def section_1_overview(df):
    section("1. OVERALL DATASET OVERVIEW")
    pr(f"Total fighter-events with odds: {len(df)}")
    pr(f"With weight class: {df['weight_class'].notna().sum()}")
    pr(f"Events: {df['date_id'].nunique()}")
    pr(f"Date range: {df['date_id'].min()} – {df['date_id'].max()}")
    pr()

    subsection("1a. Men's vs Women's Overview")
    result = df.groupby("gender").apply(agg_stats).reset_index()
    ptable(result)

    subsection("1b. Favorite vs Underdog Overview")
    result = df.groupby("fav_dog").apply(agg_stats).reset_index()
    ptable(result)

    subsection("1c. Gender × Fav/Dog")
    result = df.groupby(["gender", "fav_dog"]).apply(agg_stats).reset_index()
    ptable(result)


def section_2_weight_class(df):
    section("2. WEIGHT CLASS BREAKDOWN")

    subsection("2a. All Weight Classes — Key Metrics")
    wc_df = df[~df["weight_group"].isin(["Other"])]
    result = wc_df.groupby("weight_group").apply(agg_stats).reset_index()
    result = result.sort_values("avg_fpts", ascending=False)
    ptable(result)

    subsection("2b. Weight Class × Favorite/Underdog")
    result = wc_df.groupby(["weight_group", "fav_dog"]).apply(agg_stats).reset_index()
    result = result.sort_values(["weight_group", "fav_dog"])
    ptable(result)

    subsection("2c. Which Weight Classes Produce the Most Ceiling Games?")
    pr("Ceiling = 8+ pts/$1K, Smash = 10+ pts/$1K")
    pr()
    result = wc_df.groupby("weight_group").agg(
        n=("actual_points", "count"),
        ceil_pct=("is_ceiling", lambda x: pct(x)),
        smash_pct=("is_smash", lambda x: pct(x)),
        dud_pct=("is_dud", lambda x: pct(x)),
        avg_fpts=("actual_points", "mean"),
        pts_per_1k=("salary_value", "mean"),
    ).reset_index().sort_values("ceil_pct", ascending=False)
    ptable(result)

    subsection("2d. Weight Class Underdog Upset Rates")
    pr("How often do underdogs beat the event median FPTS?")
    pr()
    dogs = wc_df[wc_df["is_favorite"] == 0].copy()
    event_med = df.groupby("date_id")["actual_points"].median().reset_index()
    event_med.columns = ["date_id", "event_med"]
    dogs = dogs.merge(event_med, on="date_id")
    dogs["beat_med"] = dogs["actual_points"] > dogs["event_med"]
    dogs["top_q"] = dogs["actual_points"] > dogs.merge(
        df.groupby("date_id")["actual_points"].quantile(0.75).reset_index().rename(columns={"actual_points": "p75"}),
        on="date_id"
    )["p75"]

    result = dogs.groupby("weight_group").agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        beat_med_pct=("beat_med", lambda x: pct(x)),
        ceil_pct=("is_ceiling", lambda x: pct(x)),
        avg_own=("actual_ownership", "mean"),
    ).reset_index().sort_values("ceil_pct", ascending=False)
    ptable(result)


def section_3_salary(df):
    section("3. SALARY ANALYSIS")

    subsection("3a. Salary Tier × Fav/Dog")
    result = df.groupby(["salary_tier", "fav_dog"]).apply(agg_stats).reset_index()
    ptable(result)

    subsection("3b. Salary Tier × Gender")
    result = df[df["gender"] != "Unknown"].groupby(["salary_tier", "gender"]).apply(agg_stats).reset_index()
    ptable(result)

    subsection("3c. Salary Tier × Weight Class (Men's only, top 7)")
    mens = df[df["gender"] == "Men's"]
    result = mens.groupby(["salary_tier", "weight_group"]).apply(agg_stats).reset_index()
    # Filter to meaningful sample sizes
    result = result[result["n"] >= 20]
    result = result.sort_values(["salary_tier", "pts_per_1k"], ascending=[True, False])
    ptable(result)


def section_4_line_movement(df):
    section("4. LINE MOVEMENT DEEP DIVE")

    move_df = df[df["line_move_dir"] != "no_data"]

    subsection("4a. Line Movement × Fav/Dog × Gender")
    result = move_df.groupby(["line_move_dir", "fav_dog", "gender"]).apply(agg_stats).reset_index()
    result = result[result["n"] >= 20]
    result = result.sort_values(["line_move_dir", "fav_dog", "gender"])
    ptable(result)

    subsection("4b. Line Movement × Weight Class")
    result = move_df[~move_df["weight_group"].isin(["Other"])].groupby(
        ["weight_group", "line_move_dir"]
    ).apply(agg_stats).reset_index()
    result = result[result["n"] >= 15]
    result = result.sort_values(["weight_group", "line_move_dir"])
    ptable(result)

    subsection("4c. Line Movement Magnitude — Does Bigger Movement = Bigger Edge?")
    pr("Bucketed by absolute line movement size")
    pr()
    move_df2 = move_df.copy()
    move_df2["abs_move"] = move_df2["line_move"].abs()
    move_df2["move_size"] = pd.cut(
        move_df2["abs_move"],
        bins=[0, 0.02, 0.05, 0.10, 0.20, 1.0],
        labels=["<2pp", "2-5pp", "5-10pp", "10-20pp", "20pp+"],
    )

    # Split by direction
    for direction in ["steamed", "drifted"]:
        pr(f"\n  {direction.upper()} fighters by movement magnitude:")
        sub = move_df2[move_df2["line_move_dir"] == direction]
        result = sub.groupby("move_size").apply(agg_stats).reset_index()
        ptable(result)


def section_5_odds_bands(df):
    section("5. CLOSING PROBABILITY DEEP DIVE")

    subsection("5a. Prob Band × Gender")
    result = df[df["gender"] != "Unknown"].groupby(["prob_tier", "gender"]).apply(agg_stats).reset_index()
    ptable(result)

    subsection("5b. Prob Band × Weight Class")
    wc_df = df[~df["weight_group"].isin(["Other"])]
    result = wc_df.groupby(["weight_group", "prob_tier"]).apply(agg_stats).reset_index()
    result = result[result["n"] >= 15]
    result = result.sort_values(["weight_group", "prob_tier"])
    ptable(result)

    subsection("5c. Prob Band × Salary Tier")
    result = df.groupby(["prob_tier", "salary_tier"]).apply(agg_stats).reset_index()
    result = result[result["n"] >= 15]
    ptable(result)

    subsection("5d. Chalk Plays: Do Heavy Favorites Deliver?")
    pr("Heavy favorites (65%+) by salary tier — are expensive chalk worth it?")
    pr()
    chalk = df[df["close_prob"] >= 0.65]
    result = chalk.groupby("salary_tier").apply(agg_stats).reset_index()
    ptable(result)


def section_6_ownership(df):
    section("6. OWNERSHIP DEEP DIVE")

    subsection("6a. Ownership Band × Fav/Dog")
    result = df.groupby(["ownership_band", "fav_dog"]).apply(agg_stats).reset_index()
    ptable(result)

    subsection("6b. Ownership Band × Gender")
    result = df[df["gender"] != "Unknown"].groupby(["ownership_band", "gender"]).apply(agg_stats).reset_index()
    ptable(result)

    subsection("6c. Low Ownership Leverage Plays (5-15% owned)")
    pr("What makes a low-owned fighter more likely to hit a ceiling?")
    pr()
    low_own = df[(df["actual_ownership"] >= 5) & (df["actual_ownership"] <= 15)]
    result = low_own.groupby(["fav_dog", "prob_tier"]).apply(agg_stats).reset_index()
    result = result[result["n"] >= 10]
    ptable(result)

    subsection("6d. Ownership vs Implied Probability Mismatch")
    pr("Fighters who are under-owned relative to their win probability (potential leverage)")
    pr()
    df2 = df.copy()
    # Expected ownership roughly scales with probability
    df2["own_vs_prob"] = df2["actual_ownership"] - (df2["close_prob"] * 35)  # rough scale
    df2["own_mismatch"] = pd.cut(
        df2["own_vs_prob"],
        bins=[-100, -10, -5, 0, 5, 10, 100],
        labels=["Very Underowned", "Underowned", "Slightly Under", "Slightly Over", "Overowned", "Very Overowned"],
    )
    result = df2.groupby("own_mismatch").apply(agg_stats).reset_index()
    ptable(result)


def section_7_card_position(df):
    section("7. CARD POSITION DEEP DIVE")

    card_df = df[df["card_section"].notna()]

    subsection("7a. Card Position × Fav/Dog")
    result = card_df.groupby(["card_section", "fav_dog"]).apply(agg_stats).reset_index()
    order = {"main_event": 0, "co_main": 1, "main_card": 2, "prelim": 3}
    result["_sort"] = result["card_section"].map(order)
    result = result.sort_values(["_sort", "fav_dog"]).drop(columns=["_sort"])
    ptable(result)

    subsection("7b. Card Position × Gender")
    result = card_df[card_df["gender"] != "Unknown"].groupby(
        ["card_section", "gender"]
    ).apply(agg_stats).reset_index()
    order = {"main_event": 0, "co_main": 1, "main_card": 2, "prelim": 3}
    result["_sort"] = result["card_section"].map(order)
    result = result.sort_values(["_sort", "gender"]).drop(columns=["_sort"])
    ptable(result)

    subsection("7c. Card Position × Line Movement")
    move_card = card_df[card_df["line_move_dir"] != "no_data"]
    result = move_card.groupby(["card_section", "line_move_dir"]).apply(agg_stats).reset_index()
    result = result[result["n"] >= 10]
    order = {"main_event": 0, "co_main": 1, "main_card": 2, "prelim": 3}
    result["_sort"] = result["card_section"].map(order)
    result = result.sort_values(["_sort", "line_move_dir"]).drop(columns=["_sort"])
    ptable(result)


def section_8_scoring_patterns(df):
    section("8. SCORING PATTERNS & DUD ANALYSIS")

    subsection("8a. Dud Rate (< 10 FPTS) by Weight Class × Fav/Dog")
    pr("What % of fighters score under 10 FPTS (essentially zero for DFS)?")
    pr()
    wc_df = df[~df["weight_group"].isin(["Other"])]
    result = wc_df.groupby(["weight_group", "fav_dog"]).agg(
        n=("actual_points", "count"),
        dud_pct=("is_dud", lambda x: pct(x)),
        avg_fpts=("actual_points", "mean"),
        med_fpts=("actual_points", "median"),
    ).reset_index().sort_values(["weight_group", "fav_dog"])
    ptable(result)

    subsection("8b. Floor Analysis — P10 (worst 10% outcome) by Prob Tier")
    pr("What's the floor for favorites vs underdogs?")
    pr()
    result = df.groupby(["prob_tier", "fav_dog"]).agg(
        n=("actual_points", "count"),
        p10_fpts=("actual_points", lambda x: x.quantile(0.10)),
        p25_fpts=("actual_points", lambda x: x.quantile(0.25)),
        med_fpts=("actual_points", "median"),
        p75_fpts=("actual_points", lambda x: x.quantile(0.75)),
        p90_fpts=("actual_points", lambda x: x.quantile(0.90)),
    ).reset_index()
    ptable(result)

    subsection("8c. Blowout Scoring — When Favorites Win Big")
    pr("FPTS distribution for heavy favorites (65%+) vs toss-ups (45-55%)")
    pr()
    for label, filt in [("Heavy Fav 65%+", df["close_prob"] >= 0.65),
                         ("Toss-up 45-55%", (df["close_prob"] >= 0.45) & (df["close_prob"] <= 0.55))]:
        sub = df[filt]
        pr(f"\n  {label} (n={len(sub)}):")
        pr(f"    Mean: {sub['actual_points'].mean():.1f}")
        pr(f"    Median: {sub['actual_points'].median():.1f}")
        pr(f"    P10/P25/P75/P90: {sub['actual_points'].quantile(0.10):.1f} / "
           f"{sub['actual_points'].quantile(0.25):.1f} / "
           f"{sub['actual_points'].quantile(0.75):.1f} / "
           f"{sub['actual_points'].quantile(0.90):.1f}")
        pr(f"    Ceiling rate: {pct(sub['is_ceiling']):.1f}%")
        pr(f"    Dud rate: {pct(sub['is_dud']):.1f}%")


def section_9_cross_cuts(df):
    section("9. HIGH-VALUE CROSS-CUTS FOR LINEUP CONSTRUCTION")

    subsection("9a. GPP Goldmine: Low-Owned + Steamed + Underdog")
    pr("Fighters 5-20% owned, line steamed toward them, underdog")
    pr()
    gpp = df[
        (df["actual_ownership"] >= 5) & (df["actual_ownership"] <= 20) &
        (df["line_move_dir"] == "steamed") & (df["is_favorite"] == 0)
    ]
    pr(f"  n = {len(gpp)}")
    if len(gpp) > 0:
        pr(f"  Avg FPTS: {gpp['actual_points'].mean():.1f}")
        pr(f"  Ceiling rate: {pct(gpp['is_ceiling']):.1f}%")
        pr(f"  Dud rate: {pct(gpp['is_dud']):.1f}%")
        pr(f"  Avg ownership: {gpp['actual_ownership'].mean():.1f}%")
        pr(f"  Avg close prob: {gpp['close_prob'].mean():.3f}")

    subsection("9b. Cash Lock: High-Owned + Stable/Steamed + Favorite")
    pr("Fighters 25%+ owned, line stable or steamed, favorite")
    pr()
    cash = df[
        (df["actual_ownership"] >= 25) &
        (df["line_move_dir"].isin(["stable", "steamed"])) &
        (df["is_favorite"] == 1)
    ]
    pr(f"  n = {len(cash)}")
    if len(cash) > 0:
        pr(f"  Avg FPTS: {cash['actual_points'].mean():.1f}")
        pr(f"  Median FPTS: {cash['actual_points'].median():.1f}")
        pr(f"  Dud rate: {pct(cash['is_dud']):.1f}%")
        pr(f"  Floor (P10): {cash['actual_points'].quantile(0.10):.1f}")

    subsection("9c. Trap Plays: High-Owned + Drifted")
    pr("Fighters 20%+ owned whose line drifted away (sharp money against)")
    pr()
    trap = df[
        (df["actual_ownership"] >= 20) & (df["line_move_dir"] == "drifted")
    ]
    pr(f"  n = {len(trap)}")
    if len(trap) > 0:
        pr(f"  Avg FPTS: {trap['actual_points'].mean():.1f}")
        pr(f"  Ceiling rate: {pct(trap['is_ceiling']):.1f}%")
        pr(f"  Dud rate: {pct(trap['is_dud']):.1f}%")
        pr(f"  vs all fighters avg FPTS: {df['actual_points'].mean():.1f}")

    subsection("9d. Weight Class Value Leaders")
    pr("Best pts/$1K by weight class for favorites and underdogs separately")
    pr()
    wc_df = df[~df["weight_group"].isin(["Other"])]
    for role in ["Favorite", "Underdog"]:
        pr(f"\n  {role}s:")
        sub = wc_df[wc_df["fav_dog"] == role]
        result = sub.groupby("weight_group").agg(
            n=("actual_points", "count"),
            pts_per_1k=("salary_value", "mean"),
            ceil_pct=("is_ceiling", lambda x: pct(x)),
            avg_own=("actual_ownership", "mean"),
        ).reset_index().sort_values("pts_per_1k", ascending=False)
        result = result[result["n"] >= 20]
        ptable(result)

    subsection("9e. Highest Ceiling Weight Classes by Gender")
    pr()
    for g in ["Men's", "Women's"]:
        pr(f"\n  {g}:")
        sub = df[df["gender"] == g]
        result = sub.groupby("weight_group").agg(
            n=("actual_points", "count"),
            ceil_pct=("is_ceiling", lambda x: pct(x)),
            smash_pct=("is_smash", lambda x: pct(x)),
            avg_fpts=("actual_points", "mean"),
            dud_pct=("is_dud", lambda x: pct(x)),
        ).reset_index().sort_values("ceil_pct", ascending=False)
        ptable(result)


def section_10_womens_deep_dive(df):
    section("10. WOMEN'S DIVISIONS DEEP DIVE")

    womens = df[df["gender"] == "Women's"]
    pr(f"Total women's fighter-events: {len(womens)}")

    subsection("10a. Women's Weight Class × Fav/Dog")
    result = womens.groupby(["weight_group", "fav_dog"]).apply(agg_stats).reset_index()
    ptable(result)

    subsection("10b. Women's Salary Tier Performance")
    result = womens.groupby(["salary_tier", "fav_dog"]).apply(agg_stats).reset_index()
    result = result[result["n"] >= 5]
    ptable(result)

    subsection("10c. Women's Line Movement")
    move_w = womens[womens["line_move_dir"] != "no_data"]
    result = move_w.groupby(["line_move_dir", "fav_dog"]).apply(agg_stats).reset_index()
    result = result[result["n"] >= 5]
    ptable(result)


def section_11_heavyweight(df):
    section("11. HEAVYWEIGHT DEEP DIVE")
    pr("HW is unique: fewer rounds, more KO finishes, more volatile scoring")

    hw = df[df["weight_group"] == "Heavyweight"]
    pr(f"\nTotal HW fighter-events: {len(hw)}")

    subsection("11a. HW Fav/Dog Performance")
    result = hw.groupby("fav_dog").apply(agg_stats).reset_index()
    ptable(result)

    subsection("11b. HW Prob Band Performance")
    result = hw.groupby("prob_tier").apply(agg_stats).reset_index()
    ptable(result)

    subsection("11c. HW Scoring Volatility vs Other Weight Classes")
    pr()
    for wc in ["Heavyweight", "Middleweight", "Lightweight", "Bantamweight"]:
        sub = df[df["weight_group"] == wc]
        if len(sub) > 0:
            pr(f"  {wc}: mean={sub['actual_points'].mean():.1f}, "
               f"std={sub['actual_points'].std():.1f}, "
               f"CV={sub['actual_points'].std()/sub['actual_points'].mean():.2f}, "
               f"dud%={pct(sub['is_dud']):.1f}%, ceil%={pct(sub['is_ceiling']):.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()

    section("UFC DFS CONSTRUCTION RULES — COMPREHENSIVE ANALYSIS")
    pr(f"Generated from {len(df):,} fighter-events with closing odds")
    pr(f"Date range: {df['date_id'].min()} – {df['date_id'].max()} ({df['date_id'].nunique()} events)")

    section_1_overview(df)
    section_2_weight_class(df)
    section_3_salary(df)
    section_4_line_movement(df)
    section_5_odds_bands(df)
    section_6_ownership(df)
    section_7_card_position(df)
    section_8_scoring_patterns(df)
    section_9_cross_cuts(df)
    section_10_womens_deep_dive(df)
    section_11_heavyweight(df)

    pr(f"\n{'=' * 80}")
    pr("  END OF ANALYSIS")
    pr(f"{'=' * 80}")


if __name__ == "__main__":
    main()
