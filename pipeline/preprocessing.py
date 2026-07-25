"""Clean and normalize the raw transaction/customer table."""

from __future__ import annotations

import pandas as pd

from pipeline.column_mapper import resolve_column


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [c.strip() for c in cleaned.columns]

    date_col = resolve_column(cleaned, "date")
    if date_col:
        cleaned[date_col] = pd.to_datetime(cleaned[date_col], dayfirst=True, errors="coerce")

    amount_col = resolve_column(cleaned, "amount")
    if amount_col:
        cleaned[amount_col] = pd.to_numeric(cleaned[amount_col], errors="coerce")

    numeric_hints = [
        "avg_monthly_balance", "max_monthly_balance", "txn_frequency_monthly",
        "days_since_last_txn", "credit_score", "yearly_income", "num_products_held",
    ]
    for hint in numeric_hints:
        col = resolve_column(cleaned, hint)
        if col:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned


def missing_value_report(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing:
            rows.append({
                "column": col,
                "missing_count": missing,
                "missing_pct": round(100 * missing / len(df), 2),
            })
    return sorted(rows, key=lambda x: x["missing_count"], reverse=True)