"""Turn segment statistics into readable banking personas."""

from __future__ import annotations

import pandas as pd


PERSONA_TEMPLATES = {
    "priority": {
        "title": "Priority Banking Client",
        "tagline": "High balance, active engagement, strong relationship depth.",
    },
    "regular": {
        "title": "Steady Regular Client",
        "tagline": "Consistent but moderate activity; room for deeper product adoption.",
    },
    "dormant": {
        "title": "Dormant / At-Risk Client",
        "tagline": "Low recent activity; retention outreach recommended.",
    },
}


def _fmt_currency(value: float) -> str:
    return f"INR {value:,.0f}"


def build_personas(segmented: pd.DataFrame) -> list[dict]:
    personas: list[dict] = []
    for segment, group in segmented.groupby("segment"):
        profile = PERSONA_TEMPLATES.get(segment, {"title": segment.title(), "tagline": "Custom segment."})
        stats = {
            "customer_count": int(len(group)),
            "avg_monthly_balance": round(float(group["avg_monthly_balance"].mean()), 2)
            if "avg_monthly_balance" in group.columns else None,
            "avg_txn_amount": round(float(group["avg_txn_amount"].mean()), 2)
            if "avg_txn_amount" in group.columns else None,
            "avg_txn_frequency": round(float(group["txn_frequency_monthly"].mean()), 2)
            if "txn_frequency_monthly" in group.columns else None,
            "avg_days_since_last_txn": round(float(group["days_since_last_txn"].mean()), 1)
            if "days_since_last_txn" in group.columns else None,
        }

        insights = []
        if stats["avg_monthly_balance"] is not None:
            insights.append(f"Typical monthly balance around {_fmt_currency(stats['avg_monthly_balance'])}.")
        if stats["avg_txn_frequency"] is not None:
            insights.append(f"Averages {stats['avg_txn_frequency']:.1f} transactions per month.")
        if stats["avg_days_since_last_txn"] is not None:
            insights.append(f"Last activity about {stats['avg_days_since_last_txn']:.0f} days ago on average.")

        personas.append({
            "segment": segment,
            "title": profile["title"],
            "tagline": profile["tagline"],
            "stats": stats,
            "insights": insights,
        })
    return personas