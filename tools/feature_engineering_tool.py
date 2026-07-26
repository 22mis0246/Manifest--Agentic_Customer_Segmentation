"""Feature engineering tool."""

from __future__ import annotations

from data_store import store
from pipeline.column_mapper import normalize_feature_names
from pipeline.features import build_customer_features


def feature_engineering_tool(features_requested: list[str] | None = None) -> dict:
    raw = store.get_raw()
    requested = normalize_feature_names(features_requested) or [
        "avg_monthly_balance",
        "max_monthly_balance",
        "txn_frequency_monthly",
        "days_since_last_txn",
        "avg_txn_amount",
    ]
    customer_df = build_customer_features(raw, requested)
    store.set_customer_features(customer_df)

    preview = customer_df.head(5).to_dict(orient="records")
    return {
        "status": "success",
        "customers_engineered": len(customer_df),
        "features_created": [c for c in customer_df.columns if c != "client_id"],
        "preview": preview,
    }
