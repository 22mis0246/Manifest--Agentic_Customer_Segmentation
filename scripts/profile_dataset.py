"""Print a quick schema profile for the active banking dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_store import store
from pipeline.preprocessing import clean_raw_data, missing_value_report


def main() -> None:
    if not store.auto_load():
        print("No dataset found. Place CSV at data/customers.csv or set DATA_PATH.")
        raise SystemExit(1)

    raw = store.get_raw()
    df = clean_raw_data(raw)

    print("=" * 60)
    print("SegmentIQ Dataset Profile")
    print("=" * 60)
    print(f"Source: {store.source_name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    if "client_id" in df.columns:
        print(f"Unique customers: {df['client_id'].nunique():,}")

    print("\nColumn dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    missing = missing_value_report(df)
    print("\nMissing values:")
    if missing:
        for row in missing[:15]:
            print(f"  {row['column']}: {row['missing_count']} ({row['missing_pct']}%)")
    else:
        print("  None")

    print("\nSample rows:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
