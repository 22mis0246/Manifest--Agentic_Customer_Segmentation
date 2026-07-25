"""Simple model quality checks for segmentation outputs."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import silhouette_score


def evaluate_segments(features: pd.DataFrame, labels: pd.Series, feature_cols: list[str]) -> dict:
    clean = features[feature_cols].dropna()
    aligned_labels = labels.loc[clean.index]

    counts = aligned_labels.value_counts(normalize=True).round(3).to_dict()
    report: dict = {
        "segment_distribution": {str(k): float(v) for k, v in counts.items()},
        "segment_sizes": aligned_labels.value_counts().to_dict(),
    }

    if len(clean) > 10 and aligned_labels.nunique() > 1:
        try:
            report["silhouette_score"] = round(
                float(silhouette_score(clean, aligned_labels)), 3
            )
        except ValueError:
            report["silhouette_score"] = None
    else:
        report["silhouette_score"] = None

    return report