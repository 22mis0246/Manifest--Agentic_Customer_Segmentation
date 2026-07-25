"""Shared in-memory state for raw data, engineered features, and segment results."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


class DataStore:
    def __init__(self) -> None:
        self._raw: pd.DataFrame | None = None
        self._customer: pd.DataFrame | None = None
        self._segments: pd.DataFrame | None = None
        self._source_name: str | None = None
        self._segment_meta: dict[str, Any] = {}
        self._last_export_path: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._raw is not None

    @property
    def source_name(self) -> str | None:
        return self._source_name

    @property
    def segment_meta(self) -> dict[str, Any]:
        return self._segment_meta

    @property
    def last_export_path(self) -> str | None:
        return self._last_export_path

    def load_csv_bytes(self, data: bytes, filename: str) -> pd.DataFrame:
        df = pd.read_csv(BytesIO(data))
        self._raw = df
        self._customer = None
        self._segments = None
        self._segment_meta = {}
        self._source_name = filename
        return df

    def load_csv_path(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        df = pd.read_csv(path)
        self._raw = df
        self._customer = None
        self._segments = None
        self._segment_meta = {}
        self._source_name = str(path)
        return df

    def auto_load(self) -> bool:
        env_path = os.environ.get("DATA_PATH")
        candidate = Path(env_path) if env_path else DEFAULT_DATA_PATH
        if candidate.exists():
            self.load_csv_path(candidate)
            return True
        return False

    def get_raw(self) -> pd.DataFrame:
        if self._raw is None:
            if not self.auto_load():
                raise RuntimeError(
                    "No dataset loaded. Place customers.csv in data/ or upload via the app."
                )
        assert self._raw is not None
        return self._raw

    def get_dataframe(self) -> pd.DataFrame:
        return self.get_raw()

    def set_customer_features(self, df: pd.DataFrame) -> None:
        self._customer = df.copy()

    def get_customer_features(self) -> pd.DataFrame:
        if self._customer is None:
            raise RuntimeError("Customer features not built yet. Run feature engineering first.")
        return self._customer

    def has_customer_features(self) -> bool:
        return self._customer is not None

    def set_segments(self, df: pd.DataFrame, meta: dict[str, Any]) -> None:
        self._segments = df.copy()
        self._segment_meta = meta

    def get_segments(self) -> pd.DataFrame:
        if self._segments is None:
            raise RuntimeError("No segmentation has been run yet.")
        return self._segments

    def has_segments(self) -> bool:
        return self._segments is not None

    def save_segment_export(self, df: pd.DataFrame, prefix: str = "segments") -> str:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"{prefix}_{stamp}.csv"
        preferred = [
            "client_id", "segment", "avg_monthly_balance", "max_monthly_balance",
            "txn_frequency_monthly", "avg_txn_amount", "days_since_last_txn",
        ]
        export_cols = [c for c in preferred if c in df.columns]
        if not export_cols:
            export_cols = list(df.columns)
        df[export_cols].to_csv(path, index=False)
        self._last_export_path = str(path)
        return str(path)

    def summary(self) -> dict[str, Any]:
        if not self.is_loaded:
            return {"loaded": False}
        raw = self._raw
        assert raw is not None
        return {
            "loaded": True,
            "source": self._source_name,
            "rows": len(raw),
            "columns": list(raw.columns),
            "unique_customers": raw["client_id"].nunique() if "client_id" in raw.columns else None,
            "has_features": self.has_customer_features(),
            "has_segments": self.has_segments(),
        }


store = DataStore()
