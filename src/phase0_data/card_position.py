"""
Backfill card_position from salary rank within each event.

Salary rank is a strong proxy for card position in MMA DFS:
  - Highest-salary pair → Main Event (position 1)
  - 2nd-highest pair → Co-Main (position 2)
  - Remaining main card → positions 3-5
  - Prelims → positions 6+

We assign card_position per fight pair (opponent pairs share position),
then map to card_section labels.
"""

import sqlite3

import pandas as pd

from ..config import CARD_SECTIONS, FREEROLL_CONTEST_ID


def infer_card_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Infer card position for every fighter on every event date.

    Returns DataFrame with columns:
        date_id, player_id, salary, salary_rank, card_position, card_section
    """
    df = pd.read_sql(
        """
        SELECT DISTINCT
            c.date_id,
            p.player_id,
            p.full_name,
            p.salary,
            p.opponent
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

    results = []

    for date_id, group in df.groupby("date_id"):
        # Deduplicate: keep one row per fighter per date (highest salary if multiple contests)
        fighters = group.drop_duplicates(subset=["player_id"]).copy()
        fighters = fighters.sort_values("salary", ascending=False).reset_index(drop=True)

        # Pair fighters into fights by matching opponents
        assigned: set[int] = set()
        fight_pairs: list[tuple[int, int, float]] = []  # (pid1, pid2, max_salary)

        name_to_pid = {}
        for _, row in fighters.iterrows():
            name_to_pid[row["full_name"]] = row["player_id"]

        for _, row in fighters.iterrows():
            pid = row["player_id"]
            if pid in assigned:
                continue

            opp_name = row["opponent"]
            opp_pid = name_to_pid.get(opp_name)

            if opp_pid is not None and opp_pid not in assigned:
                max_sal = max(row["salary"], fighters.loc[fighters["player_id"] == opp_pid, "salary"].iloc[0])
                fight_pairs.append((pid, opp_pid, max_sal))
                assigned.add(pid)
                assigned.add(opp_pid)
            else:
                # No matched opponent — assign as solo
                fight_pairs.append((pid, None, row["salary"]))
                assigned.add(pid)

        # Sort fights by max salary descending → card position
        fight_pairs.sort(key=lambda x: x[2], reverse=True)

        for pos_idx, (pid1, pid2, _) in enumerate(fight_pairs, start=1):
            section = CARD_SECTIONS.get(pos_idx, "prelim")
            for pid in [pid1, pid2]:
                if pid is not None:
                    results.append({
                        "date_id": date_id,
                        "player_id": pid,
                        "card_position": pos_idx,
                        "card_section": section,
                    })

    return pd.DataFrame(results)


def backfill_card_position(conn: sqlite3.Connection) -> int:
    """
    Write card_position and card_section into player_features or a new table.

    Creates a `fighter_card_position` table with the inferred positions.
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
