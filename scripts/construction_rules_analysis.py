"""
UFC DFS Construction Rules Analysis

Mines historical data (10,485 fighter-events with odds) to extract
data-backed lineup construction rules.

Sections:
  1. Favorite vs Underdog by Salary Tier
  2. Line Movement as a Signal
  3. Optimal Ownership Bands (GPP vs Cash)
  4. Sharp Exposure + Odds Interaction
  5. Salary Tier Hit Rates (Ceiling Games)
  6. Card Position Value
  7. Closing Probability Bands
  8. Upset Rate by Odds Band
"""

import sqlite3
import numpy as np
import pandas as pd
import logging

DB_PATH = "/Users/ryancullen/Desktop/resultsdb_ufc.db"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SEPARATOR = "=" * 70


def load_data() -> pd.DataFrame:
    """Load the enriched fighter-event data."""
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
            fcp.card_section
        FROM fighter_odds_enriched foe
        LEFT JOIN fighter_card_position fcp
            ON foe.date_id = fcp.date_id AND foe.player_id = fcp.player_id
        WHERE foe.player_id IS NOT NULL
          AND foe.close_prob IS NOT NULL
          AND foe.actual_points IS NOT NULL
    """, conn)

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

    # Ceiling game = 3x salary value (e.g., $8K salary → 24+ FPTS is a ceiling)
    df["salary_value"] = df["actual_points"] / (df["salary"] / 1000)
    df["is_ceiling"] = df["salary_value"] >= 8.0  # ~8 pts per $1K is a smash
    df["is_optimal_range"] = df["salary_value"] >= 6.0  # solid value

    conn.close()
    return df


def section(title):
    log.info("")
    log.info(SEPARATOR)
    log.info(f"  {title}")
    log.info(SEPARATOR)


def print_table(df, float_cols=None):
    """Pretty print a dataframe."""
    if float_cols:
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].round(1)
    log.info(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Analysis Sections
# ---------------------------------------------------------------------------


def analysis_1_fav_vs_dog_by_salary(df):
    section("1. FAVORITE vs UNDERDOG BY SALARY TIER")
    log.info("Which salary tiers produce the best value for favorites vs underdogs?")
    log.info("")

    result = df.groupby(["salary_tier", "is_favorite"]).agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        avg_own=("actual_ownership", "mean"),
        avg_salary=("salary", "mean"),
        ceiling_rate=("is_ceiling", "mean"),
        value=("salary_value", "mean"),
    ).reset_index()

    result["is_favorite"] = result["is_favorite"].map({1: "Fav", 0: "Dog"})
    result["ceiling_rate"] = (result["ceiling_rate"] * 100).round(1)

    print_table(result, ["avg_fpts", "avg_own", "avg_salary", "value"])

    log.info("")
    log.info("KEY INSIGHT: Compare ceiling_rate and value columns across tiers.")
    log.info("Higher value = more FPTS per $1K salary. Higher ceiling_rate = more smash spots.")


def analysis_2_line_movement(df):
    section("2. LINE MOVEMENT AS A SIGNAL")
    log.info("Do fighters whose lines steam (move toward them) outperform?")
    log.info("steamed = line moved >2pp toward fighter, drifted = >2pp away, stable = <2pp")
    log.info("")

    move_df = df[df["line_move_dir"] != "no_data"]

    result = move_df.groupby("line_move_dir").agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        avg_own=("actual_ownership", "mean"),
        avg_close_prob=("close_prob", "mean"),
        ceiling_rate=("is_ceiling", "mean"),
        avg_value=("salary_value", "mean"),
    ).reset_index()

    result["ceiling_rate"] = (result["ceiling_rate"] * 100).round(1)

    print_table(result, ["avg_fpts", "avg_own", "avg_close_prob", "avg_value"])

    log.info("")

    # Line movement × favorite/underdog
    log.info("LINE MOVEMENT × FAVORITE/UNDERDOG:")
    result2 = move_df.groupby(["line_move_dir", "is_favorite"]).agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        avg_own=("actual_ownership", "mean"),
        ceiling_rate=("is_ceiling", "mean"),
    ).reset_index()
    result2["is_favorite"] = result2["is_favorite"].map({1: "Fav", 0: "Dog"})
    result2["ceiling_rate"] = (result2["ceiling_rate"] * 100).round(1)
    print_table(result2, ["avg_fpts", "avg_own"])


def analysis_3_ownership_bands(df):
    section("3. OPTIMAL OWNERSHIP BANDS")
    log.info("What ownership levels produce the best FPTS and ceiling games?")
    log.info("")

    result = df.groupby("ownership_band").agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        median_fpts=("actual_points", "median"),
        avg_value=("salary_value", "mean"),
        ceiling_rate=("is_ceiling", "mean"),
        avg_close_prob=("close_prob", "mean"),
        avg_salary=("salary", "mean"),
    ).reset_index()

    result["ceiling_rate"] = (result["ceiling_rate"] * 100).round(1)
    print_table(result, ["avg_fpts", "median_fpts", "avg_value", "avg_close_prob", "avg_salary"])

    log.info("")
    log.info("GPP INSIGHT: Low-owned fighters with high ceiling_rate = leverage gold.")
    log.info("CASH INSIGHT: High-owned fighters with high floor (median_fpts) = safe plays.")


def analysis_4_sharp_interaction(df):
    section("4. SHARP EXPOSURE + ODDS INTERACTION")
    log.info("Do sharp-backed fighters outperform, especially among underdogs?")
    log.info("")

    conn = sqlite3.connect(DB_PATH)
    sharp = pd.read_sql_query("""
        SELECT date_id, player_id, sharp_vs_field_delta, sharp_confidence
        FROM sharp_signals
    """, conn)
    conn.close()

    # Merge sharp signals with main data
    merged = df.merge(
        sharp,
        left_on=["date_id", "canonical_id"],
        right_on=["date_id", "player_id"],
        how="left",
        suffixes=("", "_sharp"),
    )

    merged["sharp_play"] = merged["sharp_vs_field_delta"] > 0
    merged["has_sharp"] = merged["sharp_vs_field_delta"].notna()

    sharp_data = merged[merged["has_sharp"]]

    if len(sharp_data) == 0:
        log.info("No sharp signal data available for this analysis.")
        return

    result = sharp_data.groupby(["sharp_play", "is_favorite"]).agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        avg_own=("actual_ownership", "mean"),
        ceiling_rate=("is_ceiling", "mean"),
        avg_delta=("sharp_vs_field_delta", "mean"),
    ).reset_index()

    result["sharp_play"] = result["sharp_play"].map({True: "Sharp+", False: "Sharp-"})
    result["is_favorite"] = result["is_favorite"].map({1: "Fav", 0: "Dog"})
    result["ceiling_rate"] = (result["ceiling_rate"] * 100).round(1)

    print_table(result, ["avg_fpts", "avg_own", "avg_delta"])

    log.info("")
    log.info("SHARP+ UNDERDOGS are the holy grail for GPPs — low owned, sharp-backed.")


def analysis_5_salary_tier_ceilings(df):
    section("5. SALARY TIER HIT RATES")
    log.info("Which salary tiers produce the most ceiling games (8+ pts/$1K)?")
    log.info("")

    result = df.groupby("salary_tier").agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        median_fpts=("actual_points", "median"),
        p75_fpts=("actual_points", lambda x: x.quantile(0.75)),
        p90_fpts=("actual_points", lambda x: x.quantile(0.90)),
        ceiling_rate=("is_ceiling", "mean"),
        optimal_rate=("is_optimal_range", "mean"),
        avg_value=("salary_value", "mean"),
    ).reset_index()

    result["ceiling_rate"] = (result["ceiling_rate"] * 100).round(1)
    result["optimal_rate"] = (result["optimal_rate"] * 100).round(1)

    print_table(result, ["avg_fpts", "median_fpts", "p75_fpts", "p90_fpts", "avg_value"])

    log.info("")
    log.info("ceiling_rate = % of fighters hitting 8+ pts/$1K")
    log.info("optimal_rate = % of fighters hitting 6+ pts/$1K (solid value)")


def analysis_6_card_position(df):
    section("6. CARD POSITION VALUE")
    log.info("Main Card vs Prelims — scoring and ownership patterns.")
    log.info("")

    card_df = df[df["card_section"].notna()]

    if len(card_df) == 0:
        log.info("No card position data available.")
        return

    result = card_df.groupby("card_section").agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        avg_own=("actual_ownership", "mean"),
        avg_salary=("salary", "mean"),
        ceiling_rate=("is_ceiling", "mean"),
        avg_value=("salary_value", "mean"),
    ).reset_index()

    result["ceiling_rate"] = (result["ceiling_rate"] * 100).round(1)

    # Sort by card order
    order = {"Main Event": 0, "Co-Main": 1, "Main Card": 2, "Prelim": 3, "Early Prelim": 4}
    result["_sort"] = result["card_section"].map(order)
    result = result.sort_values("_sort").drop(columns=["_sort"])

    print_table(result, ["avg_fpts", "avg_own", "avg_salary", "avg_value"])


def analysis_7_close_prob_bands(df):
    section("7. CLOSING PROBABILITY BANDS")
    log.info("How do fighters perform relative to their implied win probability?")
    log.info("")

    result = df.groupby("close_prob_band").agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        avg_own=("actual_ownership", "mean"),
        avg_salary=("salary", "mean"),
        ceiling_rate=("is_ceiling", "mean"),
        avg_value=("salary_value", "mean"),
    ).reset_index()

    result["ceiling_rate"] = (result["ceiling_rate"] * 100).round(1)

    print_table(result, ["avg_fpts", "avg_own", "avg_salary", "avg_value"])

    log.info("")
    log.info("Lower close_prob = underdog = lower salary + lower ownership.")
    log.info("Compare value and ceiling_rate to find the sweet spot.")


def analysis_8_upset_rates(df):
    section("8. UPSET RATES BY ODDS BAND")
    log.info("How often do underdogs win outright, by implied probability?")
    log.info("")

    dogs = df[df["is_favorite"] == 0].copy()

    # An "upset" in DFS: underdog scores more than the median for the event
    event_medians = df.groupby("date_id")["actual_points"].median().reset_index()
    event_medians.columns = ["date_id", "event_median_fpts"]
    dogs = dogs.merge(event_medians, on="date_id")
    dogs["beat_median"] = dogs["actual_points"] > dogs["event_median_fpts"]

    # High FPTS threshold — top quartile for the event
    event_p75 = df.groupby("date_id")["actual_points"].quantile(0.75).reset_index()
    event_p75.columns = ["date_id", "event_p75_fpts"]
    dogs = dogs.merge(event_p75, on="date_id")
    dogs["top_quartile"] = dogs["actual_points"] > dogs["event_p75_fpts"]

    result = dogs.groupby("close_prob_band").agg(
        n=("actual_points", "count"),
        avg_fpts=("actual_points", "mean"),
        avg_own=("actual_ownership", "mean"),
        beat_median_rate=("beat_median", "mean"),
        top_quartile_rate=("top_quartile", "mean"),
        ceiling_rate=("is_ceiling", "mean"),
    ).reset_index()

    result["beat_median_rate"] = (result["beat_median_rate"] * 100).round(1)
    result["top_quartile_rate"] = (result["top_quartile_rate"] * 100).round(1)
    result["ceiling_rate"] = (result["ceiling_rate"] * 100).round(1)

    print_table(result, ["avg_fpts", "avg_own"])

    log.info("")
    log.info("beat_median_rate = % of underdogs scoring above event median")
    log.info("top_quartile_rate = % of underdogs in top 25% of event scorers")


def analysis_summary(df):
    section("SUMMARY: KEY CONSTRUCTION RULES")
    log.info("")
    log.info("Based on %d fighter-events with closing odds:", len(df))
    log.info("")

    # Top value segments
    fav_value = df[df["is_favorite"] == 1]["salary_value"].mean()
    dog_value = df[df["is_favorite"] == 0]["salary_value"].mean()
    log.info("  Avg pts/$1K — Favorites: %.1f, Underdogs: %.1f", fav_value, dog_value)

    fav_ceil = df[df["is_favorite"] == 1]["is_ceiling"].mean() * 100
    dog_ceil = df[df["is_favorite"] == 0]["is_ceiling"].mean() * 100
    log.info("  Ceiling rate — Favorites: %.1f%%, Underdogs: %.1f%%", fav_ceil, dog_ceil)

    # Best salary tier by value
    tier_value = df.groupby("salary_tier")["salary_value"].mean()
    best_tier = tier_value.idxmax()
    log.info("  Best salary tier by pts/$1K: %s (%.1f)", best_tier, tier_value.max())

    # Line movement
    move_df = df[df["line_move_dir"] != "no_data"]
    if len(move_df) > 0:
        steamed_fpts = move_df[move_df["line_move_dir"] == "steamed"]["actual_points"].mean()
        drifted_fpts = move_df[move_df["line_move_dir"] == "drifted"]["actual_points"].mean()
        log.info("  Steamed fighters avg FPTS: %.1f vs Drifted: %.1f", steamed_fpts, drifted_fpts)

    log.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()
    log.info("Loaded %d fighter-events with closing odds", len(df))
    log.info("Date range: %d to %d", df["date_id"].min(), df["date_id"].max())
    log.info("Unique events: %d", df["date_id"].nunique())

    analysis_1_fav_vs_dog_by_salary(df)
    analysis_2_line_movement(df)
    analysis_3_ownership_bands(df)
    analysis_4_sharp_interaction(df)
    analysis_5_salary_tier_ceilings(df)
    analysis_6_card_position(df)
    analysis_7_close_prob_bands(df)
    analysis_8_upset_rates(df)
    analysis_summary(df)


if __name__ == "__main__":
    main()
