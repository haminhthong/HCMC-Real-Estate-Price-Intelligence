"""Mô-đun tính toán đặc trưng kết cấu và đo lường độ hoàn thiện thông tin."""

import pandas as pd

QUALITY_COLUMNS: list[str] = [
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Width",
    "Length",
    "Alley Width",
    "Direction",
    "Position",
    "Latitude",
    "Longitude",
]


def calculate_input_completeness(df: pd.DataFrame) -> pd.Series:
    """Tính điểm hoàn thiện dữ liệu đo lường đầu vào (0 - 100%).

    Tỷ lệ các trường đo lường không bị khuyết (NaN hoặc 'Không rõ').
    """
    quality_frame = df.reindex(columns=QUALITY_COLUMNS).copy()
    quality_frame = quality_frame.mask(quality_frame.eq("Không rõ"))
    return (quality_frame.notna().mean(axis=1) * 100).round(1)
