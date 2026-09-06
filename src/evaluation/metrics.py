"""Mô-đun tính toán các chỉ số đo lường hiệu năng hồi quy và khoảng dự báo."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Tính toán toàn diện các chỉ số đo lường sai số hồi quy bất động sản (đơn vị: triệu VND).

    Bao gồm cả các chỉ số robust cho long-tail distribution: WAPE, sMAPE, Median AE.
    """
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)

    abs_errors = np.abs(pred_arr - actual_arr)
    percentage_error = abs_errors / np.maximum(actual_arr, 1.0)
    wape = float(np.sum(abs_errors) / np.maximum(np.sum(actual_arr), 1.0) * 100)
    denominator = (np.abs(actual_arr) + np.abs(pred_arr)) / 2.0
    smape = float(np.mean(abs_errors / np.maximum(denominator, 1.0)) * 100)

    return {
        "mae_million": float(mean_absolute_error(actual_arr, pred_arr)),
        "median_ae_million": float(median_absolute_error(actual_arr, pred_arr)),
        "rmse_million": float(mean_squared_error(actual_arr, pred_arr) ** 0.5),
        "r2": float(r2_score(actual_arr, pred_arr)),
        "mape_percent": float(percentage_error.mean() * 100),
        "wape_percent": round(wape, 2),
        "smape_percent": round(smape, 2),
    }


def interval_metrics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.8,
) -> dict[str, float]:
    """Tính các chỉ số đo lường chất lượng khoảng dự báo Conformal Prediction Interval."""
    actual_arr = np.asarray(actual, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    in_interval = (actual_arr >= lower_arr) & (actual_arr <= upper_arr)
    coverage = float(np.mean(in_interval))
    widths = upper_arr - lower_arr
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    coverage_gap = float(coverage - target_coverage)
    relative_width = float(mean_width / max(float(np.mean(actual_arr)), 1.0))

    return {
        "target_coverage": target_coverage,
        "actual_coverage": coverage,
        "coverage_gap": coverage_gap,
        "mean_interval_width_million": mean_width,
        "median_interval_width_million": median_width,
        "relative_interval_width": relative_width,
    }
