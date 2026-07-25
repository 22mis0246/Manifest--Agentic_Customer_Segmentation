"""Exploratory data analysis tool."""

from __future__ import annotations

import pandas as pd

from data_store import store
from pipeline.preprocessing import clean_raw_data, missing_value_report


def eda_tool(columns: list[str] | None = None, group_by_segment: bool = False) -> dict:
    raw = store.get_raw()
    df = clean_raw_data(raw)
    target_cols = columns or list(df.columns)
    target_cols = [c for c in target_cols if c in df.columns]

    numeric = df[target_cols].select_dtypes(include="number")
    summary = {
        "dataset_rows": len(df),
        "unique_customers": int(df["client_id"].nunique()) if "client_id" in df.columns else None,
        "columns_analyzed": target_cols,
        "missing_values": missing_value_report(df[target_cols]),
        "numeric_summary": numeric.describe().round(2).to_dict() if not numeric.empty else {},
    }

    if len(numeric.columns) >= 2:
        summary["correlations"] = numeric.corr(numeric_only=True).round(3).to_dict()

    if group_by_segment and store.has_segments():
        segmented = store.get_segments()
        if "avg_txn_amount" in segmented.columns and "segment" in segmented.columns:
            by_segment = (
                segmented.groupby("segment")["avg_txn_amount"]
                .agg(["mean", "median", "count"])
                .round(2)
                .reset_index()
                .to_dict(orient="records")
            )
            summary["avg_txn_amount_by_segment"] = by_segment

    return summary
