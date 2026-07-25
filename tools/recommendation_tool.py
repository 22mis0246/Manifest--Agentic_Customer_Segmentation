"""Product recommendations and segment conversion advice."""

from __future__ import annotations

import pandas as pd

from data_store import store
from tools.segmentation_tool import segmentation_tool


PRODUCT_PLAYBOOK = {
    "priority": [
        "Offer premium wealth advisory and priority lounge access.",
        "Cross-sell investment products aligned to high balance profile.",
    ],
    "regular": [
        "Promote credit card upgrade and automated savings plans.",
        "Bundle personal loan pre-approval for eligible clients.",
    ],
    "dormant": [
        "Launch win-back campaign with fee waivers.",
        "Send personalised re-engagement nudges via preferred digital channel.",
    ],
}


def recommendation_tool(target_segment: str | None = None, conversion_query: bool = False) -> dict:
    if not store.has_segments():
        segmentation_tool()

    segmented = store.get_segments()
    meta = store.segment_meta
    thresholds = meta.get("thresholds", {})
    priority_rules = thresholds.get("priority", {}) if isinstance(thresholds, dict) else {}

    if conversion_query:
        regulars = segmented[segmented["segment"] == "regular"].copy()
        if regulars.empty:
            return {"status": "empty", "message": "No regular customers found."}

        balance_col = "max_monthly_balance" if "max_monthly_balance" in regulars.columns else "avg_monthly_balance"
        balance_cut = segmented[balance_col].quantile(0.66)
        freq_cut = segmented["txn_frequency_monthly"].quantile(0.66)

        candidates = regulars[
            (regulars[balance_col] >= balance_cut * 0.9)
            & (regulars["txn_frequency_monthly"] >= freq_cut * 0.9)
        ].sort_values(balance_col, ascending=False)

        advice = (
            "Encourage higher monthly balance maintenance, increase transaction frequency "
            "through bill-pay autopilot, and offer fee-free premium account trial for 90 days."
        )

        return {
            "status": "success",
            "conversion_candidates": len(candidates),
            "top_candidates": candidates.head(15)[["client_id", balance_col, "txn_frequency_monthly", "avg_txn_amount"]].to_dict(orient="records"),
            "target_thresholds": priority_rules or {
                balance_col: f"~{balance_cut:.2f}",
                "txn_frequency_monthly": f"~{freq_cut:.2f}",
            },
            "recommended_actions": advice,
        }

    segment = target_segment or "regular"
    group = segmented[segmented["segment"] == segment]
    if group.empty:
        return {"status": "not_found", "message": f"Segment '{segment}' not found."}

    product_gaps = {}
    for product_col in ["has_credit_card", "has_personal_loan", "has_investment_product", "has_fixed_deposit"]:
        if product_col in group.columns:
            product_gaps[product_col] = round(100 * (1 - group[product_col].mean()), 1)

    return {
        "status": "success",
        "segment": segment,
        "customers": len(group),
        "product_gap_pct": product_gaps,
        "recommendations": PRODUCT_PLAYBOOK.get(segment, ["Review segment profile for tailored offers."]),
        "retention_focus": (
            "Keep engagement high with proactive service and relevant product bundles."
            if segment == "priority"
            else "Increase product adoption and transaction frequency."
        ),
    }
