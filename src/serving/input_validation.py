"""Các kiểm tra đầu vào nhỏ, dễ giải thích cho lớp serving."""

from typing import Any

import numpy as np
import pandas as pd


def validate_supported_input(
    values: dict[str, Any],
    supported_areas: list[str],
    supported_property_types: list[str],
) -> None:
    """Từ chối loại hình hoặc khu vực không có trong dữ liệu train."""
    property_type = values.get("Property Type")
    if supported_property_types and property_type not in supported_property_types:
        raise ValueError(f"Loại bất động sản '{property_type}' chưa được hỗ trợ.")

    location_area = values.get("location_area")
    if supported_areas and location_area not in supported_areas:
        raise ValueError(f"Khu vực '{location_area}' chưa được hỗ trợ.")


def build_prediction_warnings(
    feature_frame: pd.DataFrame,
    values: dict[str, Any],
    model_package: dict[str, Any],
    predicted_price: float,
    lower_bound: float,
    upper_bound: float,
    input_completeness_score: float,
    as_of_date: Any,
) -> list[str]:
    """Tạo các cảnh báo cụ thể, không gán nhãn reliability heuristic."""
    warnings: list[str] = []
    training_ranges = model_package.get("training_ranges", {})
    for feature, bounds in training_ranges.items():
        if feature not in feature_frame.columns or len(bounds) != 2:
            continue
        value = feature_frame.iloc[0].get(feature)
        if (
            pd.notna(value)
            and np.isfinite(float(value))
            and not (bounds[0] <= float(value) <= bounds[1])
        ):
            warnings.append(
                f"INPUT_OUTSIDE_TRAINING_RANGE: {feature}={float(value):g} "
                f"nằm ngoài miền train [{bounds[0]:g}, {bounds[1]:g}]."
            )

    if pd.isna(values.get("Latitude")) or pd.isna(values.get("Longitude")):
        warnings.append(
            "MISSING_GPS: Thiếu GPS nên comparable không dùng được khoảng cách địa lý."
        )
    if input_completeness_score < 60:
        warnings.append(
            f"MISSING_INPUTS: Độ đầy đủ input chỉ đạt {input_completeness_score:.0f}/100."
        )

    reference_date = pd.to_datetime(
        model_package.get("reference_date"), errors="coerce", utc=True
    )
    valuation_date = pd.to_datetime(as_of_date, errors="coerce", utc=True)
    if pd.notna(reference_date) and pd.notna(valuation_date):
        market_age_days = int((valuation_date - reference_date).days)
        if market_age_days > 180:
            warnings.append(
                f"STALE_MARKET_REFERENCE: Dữ liệu train cũ hơn {market_age_days} ngày."
            )

    relative_width = (upper_bound - lower_bound) / max(predicted_price, 1.0)
    if relative_width > 0.8:
        warnings.append(
            f"WIDE_PREDICTION_INTERVAL: Khoảng dự báo rộng {relative_width:.0%}."
        )
    return warnings


__all__ = ["build_prediction_warnings", "validate_supported_input"]
