"""Mô-đun tính toán đặc trưng thời gian (Temporal Features) và mốc định giá tham chiếu (as-of / valuation date)."""

from typing import Any

import pandas as pd


def _parse_datetime_utc(value: Any) -> Any:
    """Chuẩn hóa ngày giờ về UTC để tránh lỗi trừ tz-naive và tz-aware."""
    return pd.to_datetime(value, errors="coerce", utc=True)


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

    # Chuẩn hóa cả hai phía về UTC. Dữ liệu cũ thường là tz-naive, trong khi
    # ngày tham chiếu serving có thể mang múi giờ HCMC.
    ref_timestamp = _parse_datetime_utc(reference_date)
    parsed_dates = _parse_datetime_utc(listing_dates)

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
        ref_ts = (
            _parse_datetime_utc(reference_date)
            if reference_date is not None
            else pd.Timestamp.now(tz="UTC")
        )
        as_of_ts = _parse_datetime_utc(as_of_date)
        if isinstance(as_of_date, pd.Series):
            return (as_of_ts - ref_ts).dt.days.fillna(0.0).astype(float)
        offset = float((as_of_ts - ref_ts).days) if pd.notna(as_of_ts) else 0.0
        return pd.Series(offset, index=dates.index, dtype=float)

    return calculate_days_from_reference(dates, reference_date=reference_date)
