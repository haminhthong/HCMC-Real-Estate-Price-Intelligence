"""Mô-đun định nghĩa các mô hình cơ sở thẩm định (Baselines).

Bao gồm:
- Naive Median: Giá trị trung vị toàn cục của giá bất động sản.
- Segment Median: Giá trị trung vị theo từng phân khúc (Loại hình x Quận/huyện).
"""

from typing import Any
import numpy as np
import pandas as pd


class NaiveMedianBaseline:
    """Mô hình cơ sở trung vị toàn cục (Naive Overall Median)."""

    def __init__(self) -> None:
        self.median_price: float = 0.0

    def fit(self, df_train: pd.DataFrame) -> "NaiveMedianBaseline":
        self.median_price = float(df_train["Price"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.median_price, dtype=float)


class SegmentMedianBaseline:
    """Mô hình cơ sở định giá phân khúc theo (Loại hình x Quận/Huyện)."""

    def __init__(self) -> None:
        self.segment_medians: dict[tuple[str, str], float] = {}
        self.overall_median: float = 0.0

    def fit(self, df_train: pd.DataFrame) -> "SegmentMedianBaseline":
        self.overall_median = float(df_train["Price"].median())
        self.segment_medians = (
            df_train.groupby(["Property Type", "location_area"])["Price"]
            .median()
            .to_dict()
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            pt = str(row.get("Property Type", ""))
            loc = str(row.get("location_area", ""))
            val = self.segment_medians.get((pt, loc), self.overall_median)
            preds.append(val)
        return np.array(preds, dtype=float)
