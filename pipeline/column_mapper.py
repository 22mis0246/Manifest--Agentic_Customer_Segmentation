"""Map messy column names to the fields our banking tools expect."""

from __future__ import annotations

import pandas as pd

COLUMN_ALIASES: dict[str, list[str]] = {
    "client_id": ["client_id", "customer_id", "cust_id", "client", "customer"],
    "avg_monthly_balance": ["avg_monthly_balance", "average_monthly_balance", "monthly_balance", "balance"],
    "max_monthly_balance": ["max_monthly_balance", "maximum_monthly_balance", "peak_balance"],
    "txn_frequency_monthly": ["txn_frequency_monthly", "transaction_frequency", "tx_frequency", "monthly_txn_count"],
    "days_since_last_txn": ["days_since_last_txn", "recency_days", "days_since_last_transaction"],
    "amount": ["amount", "transaction_amount", "txn_amount", "value"],
    "date": ["date", "transaction_date", "txn_date"],
}


def resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES.get(canonical, [canonical]):
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def required_columns_present(df: pd.DataFrame, keys: list[str]) -> dict[str, str | None]:
    return {key: resolve_column(df, key) for key in keys}