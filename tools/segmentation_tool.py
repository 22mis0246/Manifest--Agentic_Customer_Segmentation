"""Customer segmentation tool — rule-based and ML modes."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_store import store
from pipeline.evaluation import evaluate_segments
from pipeline.features import build_customer_features
from pipeline.personas import build_personas
from tools.feature_engineering_tool import feature_engineering_tool


def _ensure_features() -> pd.DataFrame:
    if not store.has_customer_features():
        feature_engineering_tool()
    return store.get_customer_features()


def _rule_based_segments(df: pd.DataFrame, criteria: list[str]) -> tuple[pd.DataFrame, dict]:
    working = df.copy()
    balance_col = "max_monthly_balance" if "max_monthly_balance" in working.columns else "avg_monthly_balance"
    freq_col = "txn_frequency_monthly"
    recency_col = "days_since_last_txn"

    balance_p66 = working[balance_col].quantile(0.66)
    balance_p33 = working[balance_col].quantile(0.33)
    freq_p66 = working[freq_col].quantile(0.66)
    freq_p33 = working[freq_col].quantile(0.33)

    def assign(row: pd.Series) -> str:
        balance = row.get(balance_col, 0) or 0
        freq = row.get(freq_col, 0) or 0
        recency = row.get(recency_col, 999) or 999

        if balance >= balance_p66 and freq >= freq_p66 and recency <= 30:
            return "priority"
        if freq <= freq_p33 or recency >= 90:
            return "dormant"
        return "regular"

    working["segment"] = working.apply(assign, axis=1)

    rules = {
        "method": "rule_based",
        "criteria_used": criteria,
        "thresholds": {
            "priority": {
                balance_col: f">= {balance_p66:.2f}",
                freq_col: f">= {freq_p66:.2f}",
                recency_col: "<= 30 days",
            },
            "dormant": {
                freq_col: f"<= {freq_p33:.2f} OR",
                recency_col: ">= 90 days",
            },
            "regular": "Customers not matching priority or dormant rules",
        },
    }
    return working, rules


def _kmeans_segments(df: pd.DataFrame, criteria: list[str], num_segments: int) -> tuple[pd.DataFrame, dict]:
    feature_cols = [c for c in criteria if c in df.columns]
    if len(feature_cols) < 2:
        feature_cols = [c for c in ["avg_monthly_balance", "txn_frequency_monthly", "avg_txn_amount"] if c in df.columns]

    clean = df.dropna(subset=feature_cols).copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean[feature_cols])
    model = KMeans(n_clusters=num_segments, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)

    score_col = feature_cols[0]
    order = sorted(range(num_segments), key=lambda i: model.cluster_centers_[i][feature_cols.index(score_col)])
    label_map = {old: f"segment_{idx + 1}" for idx, old in enumerate(order)}
    clean["segment"] = [label_map[l] for l in labels]

    meta = {
        "method": "kmeans",
        "criteria_used": feature_cols,
        "num_segments": num_segments,
        "cluster_centers": model.cluster_centers_.round(3).tolist(),
    }
    return clean, meta


def segmentation_tool(
    criteria: list[str] | None = None,
    num_segments: int = 3,
    method: str = "rule_based",
) -> dict:
    criteria = criteria or ["avg_monthly_balance", "txn_frequency_monthly", "days_since_last_txn"]
    customer_df = _ensure_features()

    if method == "kmeans":
        segmented, meta = _kmeans_segments(customer_df, criteria, num_segments)
    else:
        segmented, meta = _rule_based_segments(customer_df, criteria)

    if method == "rule_based" and num_segments == 3:
        rename = {"priority": "priority", "regular": "regular", "dormant": "dormant"}
        segmented["segment"] = segmented["segment"].map(rename).fillna(segmented["segment"])

    eval_cols = [c for c in ["avg_monthly_balance", "txn_frequency_monthly", "avg_txn_amount"] if c in segmented.columns]
    evaluation = evaluate_segments(segmented, segmented["segment"], eval_cols) if eval_cols else {}
    personas = build_personas(segmented)

    store.set_segments(segmented, {**meta, "evaluation": evaluation, "personas": personas})
    export_path = store.save_segment_export(segmented)

    distribution = segmented["segment"].value_counts().to_dict()
    return {
        "status": "success",
        "method": meta.get("method", method),
        "customers_segmented": len(segmented),
        "segment_distribution": distribution,
        "personas": personas,
        "evaluation": evaluation,
        "export_csv": export_path,
        "sample": segmented.head(10).to_dict(orient="records"),
    }
