#!/usr/bin/env python3
"""
Show statistics for semantic constraint cache files.
Usage: python show_cache_stats.py <cache_file.db>
"""

import sqlite3
import sys
from pathlib import Path


def show_cache_stats(db_path):
    if not Path(db_path).exists():
        print(f"Error: Cache file not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cursor = conn.execute("SELECT result FROM cache")
        results = [row[0] for row in cursor.fetchall()]

        if not results:
            print(f"Cache file is empty: {db_path}")
            return

        positive = sum(1 for r in results if r == 1)
        negative = sum(1 for r in results if r == 0)
        total = len(results)

        print(f"Cache file: {db_path}")
        print(f"Total entries: {total}")
        print(f"Positive (True/Safe): {positive} ({100*positive/total:.1f}%)")
        print(f"Negative (False/Unsafe): {negative} ({100*negative/total:.1f}%)")
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_cache_stats.py <cache_file.db>")
        sys.exit(1)

    show_cache_stats(sys.argv[1])
