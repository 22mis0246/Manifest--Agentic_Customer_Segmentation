"""Load a CSV from disk into the shared data store (CLI fallback)."""

import sys
from pathlib import Path

from data_store import store


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python load_data.py path/to/customers.csv")
        raise SystemExit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        raise SystemExit(1)

    df = store.load_csv_path(str(path))
    print(f"Loaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
