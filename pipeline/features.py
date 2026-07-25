"""Build one row per customer with the features segmentation needs."""

from __future__ import annotations

import pandas as pd

from pipeline.column_mapper import resolve_column
from pipeline.preprocessing import clean_raw_data


def build_customer_features(df: pd.DataFrame, features_requested: list[str] | None = None) -> pd.DataFrame:
    cleaned = clean_raw_data(df)
    client_col = resolve_column(cleaned, "client_id")
    if not client_col:
        raise ValueError("Could not find a customer id column (expected client_id).")

    amount_col = resolve_column(cleaned, "amount")
    balance_col = resolve_column(cleaned, "avg_monthly_balance")
    max_balance_col = resolve_column(cleaned, "max_monthly_balance")
    freq_col = resolve_column(cleaned, "txn_frequency_monthly")
    recency_col = resolve_column(cleaned, "days_since_last_txn")

    agg: dict[str, tuple[str, str]] = {}
    if balance_col:
        agg["avg_monthly_balance"] = (balance_col, "max")
    if max_balance_col:
        agg["max_monthly_balance"] = (max_balance_col, "max")
    if freq_col:
        agg["txn_frequency_monthly"] = (freq_col, "max")
    if recency_col:
        agg["days_since_last_txn"] = (recency_col, "min")
    if amount_col:
        agg["avg_txn_amount"] = (amount_col, "mean")
        agg["total_txn_amount"] = (amount_col, "sum")
        agg["txn_count"] = (amount_col, "count")

    profile_cols = [
        "credit_score", "yearly_income", "num_products_held",
        "has_savings_account", "has_credit_card", "has_personal_loan",
        "has_investment_product", "has_fixed_deposit", "preferred_channel",
    ]
    for col in profile_cols:
        if col in cleaned.columns:
            agg[col] = (col, "first")

    grouped = cleaned.groupby(client_col, as_index=False).agg(**agg)
    grouped = grouped.rename(columns={client_col: "client_id"})

    if features_requested:
        keep = ["client_id"] + [f for f in features_requested if f in grouped.columns]
        grouped = grouped[keep]

    return grouped