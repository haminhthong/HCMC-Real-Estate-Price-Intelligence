"""Mô-đun tính toán đặc trưng thời gian (Temporal Features)."""

from typing import Any
import numpy as np
import pandas as pd


def calculate_days_from_reference(
    listing_dates: pd.Series,
    reference_date: Any,
) -> pd.Series:
    """Tính số ngày chênh lệch giữa ngày đăng và mốc thời gian tham chiếu tập Train/Dev.

    Không sử dụng clip lower=0 để tránh collapse toàn bộ các tin đăng mới hơn về 0
    khi thực hiện inference trong tương lai.

    Args:
        listing_dates: Chuỗi ngày đăng tin.
        reference_date: Mốc thời gian tham chiếu đóng băng từ tập huấn luyện.

    Returns:
        pd.Series chứa số ngày chênh lệch (âm nếu tin cũ hơn, dương nếu tin mới hơn).
    """
    if reference_date is None:
        return pd.Series(0, index=listing_dates.index, dtype=float)

    ref_timestamp = pd.to_datetime(reference_date)
    parsed_dates = pd.to_datetime(listing_dates, errors="coerce")

    # Nếu ngày bị khuyết, gán 0
    days = (parsed_dates - ref_timestamp).dt.days
    return days.fillna(0).astype(float)
