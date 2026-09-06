"""Mô-đun phân tích sâu độ bao phủ và sai số mô hình theo các lát cắt dữ liệu (Slice Analysis)."""

from typing import Any
import numpy as np
import pandas as pd


def summarize_slice(df_group: pd.DataFrame) -> dict[str, Any]:
    """Tổng hợp các chỉ số trung bình, trung vị và độ bao phủ cho một nhóm con."""
    count = len(df_group)
    if count == 0:
        return {"count": 0, "note": "Không có mẫu"}
    return {
        "count": count,
        "mean_mae_million": round(float(df_group["absolute_error_million"].mean()), 2),
        "median_mae_million": round(float(df_group["absolute_error_million"].median()), 2),
        "interval_coverage": round(float(df_group["in_interval"].mean()), 4),
        "median_interval_width_million": round(float(df_group["interval_width_million"].median()), 2),
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

    1. Theo Loại hình bất động sản
    2. Theo Quận / huyện (location_area)
    3. Theo Tầm giá (price_range)
    4. Theo Điểm hoàn thiện dữ liệu (completeness_range)
    5. Theo Khoảng cách tới trung tâm CBD (cbd_distance_range)
    """
    test_result = df_test[["Property Type", "location_area", "Price"]].copy()
    test_result["predicted_price_million"] = test_pred_price
    test_result["absolute_error_million"] = np.abs(test_pred_price - test_actual)
    test_result["in_interval"] = (test_actual >= lower_bound) & (test_actual <= upper_bound)
    test_result["interval_width_million"] = upper_bound - lower_bound

    if "input_completeness_score" in features_test:
        test_result["input_completeness_score"] = features_test["input_completeness_score"].to_numpy()
    else:
        test_result["input_completeness_score"] = 100.0

    if "distance_to_cbd_km" in features_test:
        test_result["distance_to_cbd_km"] = features_test["distance_to_cbd_km"].to_numpy()
    else:
        test_result["distance_to_cbd_km"] = 10.0

    price_bins = [0, 3000, 7000, 15000, np.inf]
    price_labels = ["< 3 tỷ", "3 - 7 tỷ", "7 - 15 tỷ", "> 15 tỷ"]
    test_result["price_range"] = pd.cut(test_result["Price"], bins=price_bins, labels=price_labels)

    completeness_bins = [0, 60, 80, 100.1]
    completeness_labels = ["< 60% (Thiếu nhiều)", "60 - 80% (Khá)", ">= 80% (Đầy đủ)"]
    test_result["completeness_range"] = pd.cut(
        test_result["input_completeness_score"],
        bins=completeness_bins,
        labels=completeness_labels,
    )

    cbd_bins = [0, 5, 10, np.inf]
    cbd_labels = ["< 5 km (Trung tâm)", "5 - 10 km (Cận trung tâm)", "> 10 km (Ngoại vi)"]
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
