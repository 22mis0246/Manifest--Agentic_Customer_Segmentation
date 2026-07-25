"""Explainability tool for segment assignments."""

from __future__ import annotations

import pandas as pd

from data_store import store
from tools.segmentation_tool import segmentation_tool


def explainability_tool(customer_id: str | None = None, segment_name: str | None = None) -> dict:
    if not store.has_segments():
        segmentation_tool()

    segmented = store.get_segments()
    meta = store.segment_meta

    if customer_id:
        match = segmented[segmented["client_id"] == customer_id]
        if match.empty:
            return {"status": "not_found", "message": f"No customer found with id {customer_id}"}
        row = match.iloc[0]
        return {
            "status": "success",
            "client_id": customer_id,
            "segment": row["segment"],
            "feature_snapshot": row.drop(labels=["segment"]).to_dict(),
            "rules_applied": meta.get("thresholds", meta),
        }

    target = segment_name or "priority"
    group = segmented[segmented["segment"] == target]
    if group.empty:
        return {"status": "not_found", "message": f"No customers in segment '{target}'"}

    feature_cols = [
        c for c in ["avg_monthly_balance", "max_monthly_balance", "txn_frequency_monthly", "avg_txn_amount", "days_since_last_txn"]
        if c in group.columns
    ]
    distinguishing = {}
    overall = segmented[feature_cols].mean()
    segment_means = group[feature_cols].mean()
    for col in feature_cols:
        distinguishing[col] = {
            "segment_avg": round(float(segment_means[col]), 2),
            "overall_avg": round(float(overall[col]), 2),
            "difference": round(float(segment_means[col] - overall[col]), 2),
        }

    return {
        "status": "success",
        "segment": target,
        "customers_in_segment": len(group),
        "rules_applied": meta.get("thresholds", meta),
        "distinguishing_features": distinguishing,
        "explanation": (
            f"{target.title()} customers were selected using the stored segmentation rules "
            f"and show higher/lower values on the listed features compared to the overall base."
        ),
    }
