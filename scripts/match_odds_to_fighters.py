"""
Match Odds API fighter names to canonical fighters and DFS player records.

Creates:
  - odds_api_fighter_map: maps each Odds API name → canonical_id + match method
  - odds_api_matched: fully joined table with date_id, player_id, canonical_id,
    opening/closing odds, line movement, and DFS fields

Usage:
    python scripts/match_odds_to_fighters.py
"""

import sqlite3
import logging
import sys

sys.path.insert(0, ".")
from src.phase0_data.identity import (
    build_canonical_fighters,
    match_name,
    normalize_name,
)

DB_PATH = "/Users/ryancullen/Desktop/resultsdb_ufc.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_unique_odds_names(conn: sqlite3.Connection) -> set[str]:
    """Get all unique fighter names from odds_api_consensus."""
    names = set()
    for row in conn.execute("SELECT DISTINCT fighter_a FROM odds_api_consensus"):
        names.add(row[0])
    for row in conn.execute("SELECT DISTINCT fighter_b FROM odds_api_consensus"):
        names.add(row[0])
    return names


def build_odds_fighter_map(conn: sqlite3.Connection, lookup: dict[str, int]):
    """
    Match every Odds API fighter name to a canonical_id.
    Save results to odds_api_fighter_map table.
    """
    odds_names = get_unique_odds_names(conn)
    log.info("Unique Odds API fighter names: %d", len(odds_names))

    conn.execute("DROP TABLE IF EXISTS odds_api_fighter_map")
    conn.execute("""
        CREATE TABLE odds_api_fighter_map (
            odds_name TEXT PRIMARY KEY,
            canonical_id INTEGER,
            canonical_name TEXT,
            match_method TEXT
        )
    """)

    matched = 0
    unmatched_names = []

    for name in sorted(odds_names):
        cid, method = match_name(name, lookup)

        if cid is not None:
            # Look up canonical display name
            row = conn.execute(
                "SELECT canonical_name FROM canonical_fighter WHERE canonical_id = ?",
                (cid,),
            ).fetchone()
            canon_name = row[0] if row else None
            matched += 1
        else:
            canon_name = None
            unmatched_names.append(name)

        conn.execute(
            "INSERT INTO odds_api_fighter_map VALUES (?, ?, ?, ?)",
            (name, cid, canon_name, method),
        )

    conn.commit()

    log.info("Matched: %d/%d (%.1f%%)", matched, len(odds_names),
             100 * matched / len(odds_names))
    log.info("Unmatched: %d", len(unmatched_names))

    # Log match method distribution
    for row in conn.execute("""
        SELECT match_method, COUNT(*) as n
        FROM odds_api_fighter_map
        GROUP BY match_method
        ORDER BY n DESC
    """):
        log.info("  %s: %d", row[0], row[1])

    if unmatched_names:
        log.info("Unmatched names (first 30):")
        for name in unmatched_names[:30]:
            log.info("  - %s", name)

    # Register odds names as aliases for matched fighters
    for row in conn.execute("""
        SELECT odds_name, canonical_id FROM odds_api_fighter_map
        WHERE canonical_id IS NOT NULL
    """):
        norm = normalize_name(row[0])
        conn.execute(
            "INSERT OR IGNORE INTO fighter_alias (canonical_id, alias_name, alias_source) "
            "VALUES (?, ?, 'odds_api')",
            (row[1], norm),
        )
    conn.commit()
    log.info("Registered odds names as aliases in fighter_alias table")

    return unmatched_names


def build_matched_odds_table(conn: sqlite3.Connection):
    """
    Create odds_api_matched: join consensus odds with DFS player records.

    Links odds to players by matching canonical_id + date_id.
    Pivots opening/closing into a single row per fighter per event.
    """
    conn.execute("DROP TABLE IF EXISTS odds_api_matched")

    conn.execute("""
        CREATE TABLE odds_api_matched AS
        WITH closing AS (
            SELECT
                oc.event_id,
                oc.date_id,
                oc.fighter_a,
                oc.fighter_b,
                ma.canonical_id AS canonical_id_a,
                mb.canonical_id AS canonical_id_b,
                oc.implied_prob_a_fair AS close_prob_a,
                oc.implied_prob_b_fair AS close_prob_b,
                oc.n_books AS close_n_books,
                oc.line_move_a,
                oc.line_move_b
            FROM odds_api_consensus oc
            LEFT JOIN odds_api_fighter_map ma ON oc.fighter_a = ma.odds_name
            LEFT JOIN odds_api_fighter_map mb ON oc.fighter_b = mb.odds_name
            WHERE oc.line_type = 'closing'
        ),
        opening AS (
            SELECT
                oc.event_id,
                oc.implied_prob_a_fair AS open_prob_a,
                oc.implied_prob_b_fair AS open_prob_b,
                oc.n_books AS open_n_books
            FROM odds_api_consensus oc
            WHERE oc.line_type = 'opening'
        )
        SELECT
            c.event_id,
            c.date_id,
            c.fighter_a,
            c.fighter_b,
            c.canonical_id_a,
            c.canonical_id_b,
            o.open_prob_a,
            o.open_prob_b,
            c.close_prob_a,
            c.close_prob_b,
            c.line_move_a,
            c.line_move_b,
            o.open_n_books,
            c.close_n_books
        FROM closing c
        LEFT JOIN opening o ON c.event_id = o.event_id
    """)

    n = conn.execute("SELECT COUNT(*) FROM odds_api_matched").fetchone()[0]
    log.info("Created odds_api_matched with %d rows", n)

    # Now create per-fighter view joined to DFS players
    conn.execute("DROP TABLE IF EXISTS fighter_odds_enriched")
    conn.execute("""
        CREATE TABLE fighter_odds_enriched AS

        -- Fighter A rows
        SELECT
            m.date_id,
            m.canonical_id_a AS canonical_id,
            m.fighter_a AS odds_name,
            p.player_id,
            p.full_name AS dfs_name,
            p.salary,
            p.ownership AS actual_ownership,
            p.actual_points,
            m.open_prob_a AS open_prob,
            m.close_prob_a AS close_prob,
            m.line_move_a AS line_move,
            CASE WHEN m.close_prob_a > 0.5 THEN 1 ELSE 0 END AS is_favorite,
            m.open_n_books,
            m.close_n_books,
            m.fighter_b AS opponent_odds_name,
            m.canonical_id_b AS opponent_canonical_id
        FROM odds_api_matched m
        LEFT JOIN players p
            ON p.full_name IN (
                SELECT canonical_name FROM canonical_fighter WHERE canonical_id = m.canonical_id_a
            )
            AND p.contest_id IN (SELECT contest_id FROM contests WHERE date_id = m.date_id)
        WHERE m.canonical_id_a IS NOT NULL

        UNION ALL

        -- Fighter B rows
        SELECT
            m.date_id,
            m.canonical_id_b AS canonical_id,
            m.fighter_b AS odds_name,
            p.player_id,
            p.full_name AS dfs_name,
            p.salary,
            p.ownership AS actual_ownership,
            p.actual_points,
            m.open_prob_b AS open_prob,
            m.close_prob_b AS close_prob,
            m.line_move_b AS line_move,
            CASE WHEN m.close_prob_b > 0.5 THEN 1 ELSE 0 END AS is_favorite,
            m.open_n_books,
            m.close_n_books,
            m.fighter_a AS opponent_odds_name,
            m.canonical_id_a AS opponent_canonical_id
        FROM odds_api_matched m
        LEFT JOIN players p
            ON p.full_name IN (
                SELECT canonical_name FROM canonical_fighter WHERE canonical_id = m.canonical_id_b
            )
            AND p.contest_id IN (SELECT contest_id FROM contests WHERE date_id = m.date_id)
        WHERE m.canonical_id_b IS NOT NULL
    """)

    n2 = conn.execute("SELECT COUNT(*) FROM fighter_odds_enriched").fetchone()[0]
    matched_dfs = conn.execute(
        "SELECT COUNT(*) FROM fighter_odds_enriched WHERE player_id IS NOT NULL"
    ).fetchone()[0]
    log.info("Created fighter_odds_enriched: %d rows (%d with DFS player_id)", n2, matched_dfs)

    conn.commit()


def print_summary(conn: sqlite3.Connection):
    """Print coverage summary."""
    log.info("=" * 60)
    log.info("COVERAGE SUMMARY")
    log.info("=" * 60)

    # Total DFS events and fighter-events
    total_dates = conn.execute("SELECT COUNT(DISTINCT date_id) FROM contests").fetchone()[0]
    total_players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]

    # Odds coverage
    odds_dates = conn.execute(
        "SELECT COUNT(DISTINCT date_id) FROM fighter_odds_enriched WHERE close_prob IS NOT NULL"
    ).fetchone()[0]
    odds_fighters = conn.execute(
        "SELECT COUNT(*) FROM fighter_odds_enriched WHERE player_id IS NOT NULL AND close_prob IS NOT NULL"
    ).fetchone()[0]

    # Line movement coverage
    move_fighters = conn.execute(
        "SELECT COUNT(*) FROM fighter_odds_enriched WHERE player_id IS NOT NULL AND line_move IS NOT NULL"
    ).fetchone()[0]

    log.info("DFS events: %d", total_dates)
    log.info("DFS fighter-event rows: %d", total_players)
    log.info("Odds API date coverage: %d/%d (%.0f%%)", odds_dates, total_dates,
             100 * odds_dates / max(total_dates, 1))
    log.info("Fighter-events with closing odds: %d (%.0f%% of DFS rows)",
             odds_fighters, 100 * odds_fighters / max(total_players, 1))
    log.info("Fighter-events with line movement: %d (%.0f%% of DFS rows)",
             move_fighters, 100 * move_fighters / max(total_players, 1))

    # Favorite/underdog breakdown
    fav_stats = conn.execute("""
        SELECT
            is_favorite,
            COUNT(*) as n,
            ROUND(AVG(actual_points), 1) as avg_fpts,
            ROUND(AVG(actual_ownership), 1) as avg_own
        FROM fighter_odds_enriched
        WHERE player_id IS NOT NULL AND close_prob IS NOT NULL
        GROUP BY is_favorite
    """).fetchall()

    log.info("Favorite/Underdog breakdown:")
    for row in fav_stats:
        label = "Favorite" if row[0] == 1 else "Underdog"
        log.info("  %s: n=%d, avg_fpts=%.1f, avg_own=%.1f%%", label, row[1], row[2], row[3])


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Step 1: Build/refresh canonical lookup
    log.info("Building canonical fighter lookup...")
    lookup = build_canonical_fighters(conn)
    log.info("Canonical lookup: %d entries", len(lookup))

    # Step 2: Match odds names
    log.info("=" * 60)
    log.info("Matching Odds API names to canonical fighters")
    log.info("=" * 60)
    unmatched = build_odds_fighter_map(conn, lookup)

    # Step 3: Build joined tables
    log.info("=" * 60)
    log.info("Building matched odds tables")
    log.info("=" * 60)
    build_matched_odds_table(conn)

    # Step 4: Summary
    print_summary(conn)

    conn.close()


if __name__ == "__main__":
    main()
