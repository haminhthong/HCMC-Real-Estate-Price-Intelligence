"""Mô-đun hiệu chuẩn khoảng dự báo (Prediction Interval) bằng Split Conformal Prediction."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def conformal_quantile(residuals: np.ndarray, coverage: float = 0.8) -> float:
    """Tính toán phân vị (Quantile) phần dư tuyệt đối phục vụ Split Conformal Prediction.

    Công thức thứ tự bảo thủ hữu hạn mẫu (finite-sample guarantee):
        p = ceil((n + 1) * coverage) / n
        q = sort(residuals)[rank - 1]

    Args:
        residuals: Mảng 1D chứa các phần dư tuyệt đối không âm trên tập Calibration.
        coverage: Mức độ bao phủ mục tiêu (ví dụ: 0.8 ứng với 80%).

    Returns:
        Giá trị phân vị tuyệt đối dạng float.

    Raises:
        ValueError: Nếu tập phần dư rỗng hoặc coverage không nằm trong khoảng (0, 1).
    """
    if len(residuals) == 0:
        raise ValueError(
            "Tập phần dư hiệu chỉnh (calibration residuals) không được rỗng."
        )
    if not 0 < coverage < 1:
        raise ValueError("Mức bao phủ (coverage) phải nằm trong khoảng (0, 1).")

    res_sorted = np.sort(residuals)
    rank = min(int(np.ceil((len(residuals) + 1) * coverage)), len(residuals))
    return float(res_sorted[rank - 1])


def calibrate_conformal(
    model_pipeline: Pipeline,
    features_calib: pd.DataFrame,
    df_calib: pd.DataFrame,
    target_formulation: str,
    target_coverage: float = 0.8,
) -> dict[str, Any]:
    """Thực hiện hiệu chuẩn Conformal Prediction Interval trên tập Calibration (10%).

    NGUYÊN TẮC:
    1. Chạy trên tập Calibration sau khi champion đã được refit.
    2. Phần dư được tính toán trong không gian log-target: e_i = |y_i - y_hat_i|.
    3. Trả về giá trị `residual_log_quantile` phục vụ thiết lập cận trên/dưới.
    """
    calib_raw_pred = model_pipeline.predict(features_calib)

    if target_formulation == "price_per_m2":
        calib_y_true = np.log1p(df_calib["Price"] / df_calib["Area"]).to_numpy()
    else:
        calib_y_true = np.log1p(df_calib["Price"]).to_numpy()

    calibration_residuals = np.abs(calib_y_true - calib_raw_pred)
    residual_log_quantile = conformal_quantile(
        calibration_residuals,
        coverage=target_coverage,
    )

    return {
        "residual_log_quantile": residual_log_quantile,
        "target_coverage": target_coverage,
        "calibration_samples": len(df_calib),
        "mean_calibration_residual": float(np.mean(calibration_residuals)),
        "median_calibration_residual": float(np.median(calibration_residuals)),
    }
