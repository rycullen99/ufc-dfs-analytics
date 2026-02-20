#!/usr/bin/env python3
"""Quick DB exploration tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import pandas as pd
from src.config import DB_PATH


def explore():
    conn = sqlite3.connect(DB_PATH)

    # List all tables
    tables = pd.read_sql(
        "SELECT name, type FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )
    print("=" * 60)
    print(f"DATABASE: {DB_PATH}")
    print(f"Tables: {len(tables)}")
    print("=" * 60)

    for _, row in tables.iterrows():
        count = conn.execute(f"SELECT COUNT(*) FROM [{row['name']}]").fetchone()[0]
        print(f"  {row['name']:<30} {count:>12,} rows")

    # Date range
    try:
        dates = pd.read_sql("SELECT MIN(date_id), MAX(date_id), COUNT(DISTINCT date_id) FROM contests", conn)
        print(f"\nDate range: {dates.iloc[0, 0]} to {dates.iloc[0, 1]} ({dates.iloc[0, 2]} events)")
    except Exception:
        pass

    conn.close()


if __name__ == "__main__":
    explore()
