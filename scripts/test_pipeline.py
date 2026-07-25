"""Offline pipeline test — no Gemini API key required."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_store import store
from tools.eda_tool import eda_tool
from tools.explainability_tool import explainability_tool
from tools.feature_engineering_tool import feature_engineering_tool
from tools.recommendation_tool import recommendation_tool
from tools.segmentation_tool import segmentation_tool


def run_step(name: str, fn):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    result = fn()
    if isinstance(result, dict):
        for key in list(result.keys())[:8]:
            print(f"  {key}: {result[key]}")
    else:
        print(result)
    return result


def main() -> None:
    if not store.auto_load():
        print("Dataset missing at data/customers.csv")
        raise SystemExit(1)

    print("Loaded:", store.summary())

    run_step("Feature engineering", lambda: feature_engineering_tool())
    run_step("Segmentation", lambda: segmentation_tool())
    run_step(
        "Explain priority segment",
        lambda: explainability_tool(segment_name="priority"),
    )
    run_step(
        "Avg txn by segment",
        lambda: eda_tool(group_by_segment=True),
    )
    run_step(
        "Conversion candidates",
        lambda: recommendation_tool(conversion_query=True),
    )

    print("\nAll pipeline steps completed successfully.")


if __name__ == "__main__":
    main()