"""Mô-đun phân tích sâu độ bao phủ và sai số mô hình theo các lát cắt dữ liệu (Slice Analysis)."""

from typing import Any

import numpy as np
import pandas as pd


def summarize_slice(df_group: pd.DataFrame) -> dict[str, Any]:
    """Tổng hợp các chỉ số trung bình, trung vị, WAPE và chất lượng khoảng Conformal theo lát cắt."""
    count = len(df_group)
    if count == 0:
        return {"count": 0, "note": "Không có mẫu"}

    total_actual = float(df_group["Price"].sum())
    total_abs_err = float(df_group["absolute_error_million"].sum())
    wape = round(total_abs_err / max(total_actual, 1.0) * 100.0, 2)

    rel_widths = df_group["interval_width_million"] / df_group["Price"].clip(
        lower=100.0
    )

    return {
        "count": count,
        "mean_mae_million": round(float(df_group["absolute_error_million"].mean()), 2),
        "median_mae_million": round(
            float(df_group["absolute_error_million"].median()), 2
        ),
        "wape_percent": wape,
        "interval_coverage": round(float(df_group["in_interval"].mean()), 4),
        "mean_interval_width_million": round(
            float(df_group["interval_width_million"].mean()), 2
        ),
        "median_interval_width_million": round(
            float(df_group["interval_width_million"].median()), 2
        ),
        "median_relative_interval_width": round(float(rel_widths.median()), 3),
    }


def analyze_slices(
    df_test: pd.DataFrame,
    test_actual: np.ndarray,
    test_pred_price: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    features_test: pd.DataFrame,
) -> dict[str, Any]:
    """Phân tích hiệu năng khoảng dự báo và sai số theo 5 chiều lát cắt:

    1. Theo Loại hình bất động sản (Apartment, House, Villa...)
    2. Theo Quận / huyện (location_area)
    3. Theo Tầm giá chuẩn hóa (< 5 tỷ, 5 - 10 tỷ, 10 - 15 tỷ, > 15 tỷ)
    4. Theo Điểm hoàn thiện dữ liệu (< 60%, 60 - 80%, >= 80%)
    5. Theo Khoảng cách tới trung tâm CBD (< 5 km, 5 - 10 km, > 10 km)
    """
    test_result = df_test[["Property Type", "location_area", "Price"]].copy()
    test_result["predicted_price_million"] = test_pred_price
    test_result["absolute_error_million"] = np.abs(test_pred_price - test_actual)
    test_result["in_interval"] = (test_actual >= lower_bound) & (
        test_actual <= upper_bound
    )
    test_result["interval_width_million"] = upper_bound - lower_bound

    if "input_completeness_score" in features_test:
        test_result["input_completeness_score"] = features_test[
            "input_completeness_score"
        ].to_numpy()
    elif "data_quality_score" in features_test:
        test_result["input_completeness_score"] = features_test[
            "data_quality_score"
        ].to_numpy()
    else:
        test_result["input_completeness_score"] = 100.0

    if "distance_to_cbd_km" in features_test:
        test_result["distance_to_cbd_km"] = features_test[
            "distance_to_cbd_km"
        ].to_numpy()
    else:
        test_result["distance_to_cbd_km"] = 10.0

    # Phân vị tầm giá theo đề xuất chuẩn: < 5B, 5 - 10B, 10 - 15B, > 15B
    price_bins = [0, 5000, 10000, 15000, np.inf]
    price_labels = ["< 5 tỷ", "5 - 10 tỷ", "10 - 15 tỷ", "> 15 tỷ"]
    test_result["price_range"] = pd.cut(
        test_result["Price"], bins=price_bins, labels=price_labels
    )

    completeness_bins = [0, 60, 80, 100.1]
    completeness_labels = ["< 60% (Thiếu nhiều)", "60 - 80% (Khá)", ">= 80% (Đầy đủ)"]
    test_result["completeness_range"] = pd.cut(
        test_result["input_completeness_score"],
        bins=completeness_bins,
        labels=completeness_labels,
    )

    cbd_bins = [0, 5, 10, np.inf]
    cbd_labels = [
        "< 5 km (Trung tâm)",
        "5 - 10 km (Cận trung tâm)",
        "> 10 km (Ngoại vi)",
    ]
    test_result["cbd_distance_range"] = pd.cut(
        test_result["distance_to_cbd_km"],
        bins=cbd_bins,
        labels=cbd_labels,
    )

    return {
        "by_property_type": {
            str(name): summarize_slice(group)
            for name, group in test_result.groupby("Property Type", observed=False)
        },
        "by_location_area": {
            str(name): summarize_slice(group)
            for name, group in test_result.groupby("location_area", observed=False)
        },
        "by_price_range": {
            str(name): summarize_slice(group)
            for name, group in test_result.groupby("price_range", observed=False)
        },
        "by_completeness_range": {
            str(name): summarize_slice(group)
            for name, group in test_result.groupby("completeness_range", observed=False)
        },
        "by_cbd_distance_range": {
            str(name): summarize_slice(group)
            for name, group in test_result.groupby("cbd_distance_range", observed=False)
        },
    }
