"""
Backfill card_position from salary rank within each event.

Salary rank is a strong proxy for card position in MMA DFS:
  - Highest-salary fighters → Main Event (position 1-2)
  - Next highest → Co-Main (position 3-4)
  - Remaining high salaries → Main Card (positions 5-10)
  - Lower salaries → Prelims (positions 11+)

Since we don't have explicit fight pairing data, we rank all fighters
by salary within each event date and assign positions accordingly.
"""

import sqlite3

import pandas as pd

from ..config import CARD_SECTIONS, FREEROLL_CONTEST_ID


def infer_card_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Infer card position for every fighter on every event date.

    Returns DataFrame with columns:
        date_id, player_id, salary, card_position, card_section
    """
    df = pd.read_sql(
        """
        SELECT DISTINCT
            c.date_id,
            p.player_id,
            p.full_name,
            p.salary
        FROM players p
        JOIN contests c ON p.contest_id = c.contest_id
        WHERE p.salary IS NOT NULL
          AND c.contest_id != ?
        ORDER BY c.date_id, p.salary DESC
        """,
        conn,
        params=(FREEROLL_CONTEST_ID,),
    )

    if df.empty:
        return pd.DataFrame()

    # Deduplicate: keep one row per fighter per date (highest salary if multiple contests)
    df = df.sort_values(["date_id", "salary"], ascending=[True, False])
    df = df.drop_duplicates(subset=["date_id", "player_id"], keep="first")

    # Rank by salary within each event (1 = highest salary)
    df["card_position"] = (
        df.groupby("date_id")["salary"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Map position to card section
    df["card_section"] = df["card_position"].map(CARD_SECTIONS).fillna("prelim")

    return df[["date_id", "player_id", "card_position", "card_section"]]


def backfill_card_position(conn: sqlite3.Connection) -> int:
    """
    Write card_position and card_section into a fighter_card_position table.

    Returns the number of rows written.
    """
    positions_df = infer_card_positions(conn)
    if positions_df.empty:
        return 0

    conn.execute("DROP TABLE IF EXISTS fighter_card_position")
    conn.execute("""
        CREATE TABLE fighter_card_position (
            date_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            card_position INTEGER NOT NULL,
            card_section TEXT NOT NULL,
            PRIMARY KEY (date_id, player_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fcp_date ON fighter_card_position(date_id)"
    )

    rows = positions_df.to_dict("records")
    conn.executemany(
        """
        INSERT OR REPLACE INTO fighter_card_position
        (date_id, player_id, card_position, card_section)
        VALUES (:date_id, :player_id, :card_position, :card_section)
        """,
        rows,
    )
    conn.commit()

    return len(rows)
