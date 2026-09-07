"""Mô-đun tính toán đặc trưng thời gian (Temporal Features) và mốc định giá tham chiếu (as-of / valuation date)."""

from typing import Any
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
        return pd.Series(0.0, index=listing_dates.index, dtype=float)

    ref_timestamp = pd.to_datetime(reference_date)
    parsed_dates = pd.to_datetime(listing_dates, errors="coerce")

    # Nếu ngày bị khuyết, gán 0
    days = (parsed_dates - ref_timestamp).dt.days
    return days.fillna(0.0).astype(float)


def calculate_market_time_offset(
    dates: pd.Series,
    reference_date: Any,
    as_of_date: Any = None,
) -> pd.Series:
    """Tính toán khoảng cách thời gian thị trường (market_time_offset_days).

    Nếu có `as_of_date` (thời điểm định giá do người dùng chỉ định hoặc ngày hiện tại),
    đặc trưng thời gian phản ánh khoảng cách từ mốc tham chiếu huấn luyện tới thời điểm định giá.
    Nếu không có, fallback về `dates` (ngày đăng của tin).
    """
    if as_of_date is not None:
        ref_ts = pd.to_datetime(reference_date) if reference_date is not None else pd.Timestamp.now()
        if isinstance(as_of_date, pd.Series):
            as_of_ts = pd.to_datetime(as_of_date, errors="coerce")
            return (as_of_ts - ref_ts).dt.days.fillna(0.0).astype(float)
        else:
            as_of_ts = pd.to_datetime(as_of_date)
            offset_days = float((as_of_ts - ref_ts).days)
            return pd.Series(offset_days, index=dates.index, dtype=float)

    return calculate_days_from_reference(dates, reference_date=reference_date)
