"""Mô-đun đánh giá độc lập trên tập Test (Independent Model Evaluator)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import logger

from .metrics import interval_metrics, regression_metrics
from .slices import analyze_slices


def evaluate_champion_on_test(
    champion_pipeline: Pipeline,
    df_test: pd.DataFrame,
    features_test: pd.DataFrame,
    selected_target_fmt: str,
    residual_log_quantile: float,
    naive_baseline: Any,
    segment_baseline: Any,
    target_coverage: float = 0.8,
) -> dict[str, Any]:
    """Đánh giá độc lập mô hình Champion và các baselines trên tập Test (Report Only).

    NGUYÊN TẮC P0:
    1. Tập Test CHỈ ĐÁNH GIÁ duy nhất mô hình Champion đã refit và 2 baselines:
       - Naive Median
       - District x Property Type Segment Median
    2. TUYỆT ĐỐI KHÔNG so sánh hoặc đánh giá 5 candidate models trên Test để ngăn ngừa
       nguy cơ data leakage quyết định (nhìn test rồi thay đổi lựa chọn).
    3. Tính toán toàn diện: metrics hồi quy, metrics khoảng dự báo, và phân tích slice.
    """
    logger.info("--- BẮT ĐẦU ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (15%) ---")
    test_actual = df_test["Price"].to_numpy()

    # 1. Đánh giá Naive Median Baseline trên Test
    naive_preds = naive_baseline.predict(df_test)
    naive_test_metrics = regression_metrics(test_actual, naive_preds)

    # 2. Đánh giá Segment Median Baseline trên Test
    segment_preds = segment_baseline.predict(df_test)
    segment_test_metrics = regression_metrics(test_actual, segment_preds)

    # 3. Đánh giá Champion Model trên Test
    test_raw_pred = champion_pipeline.predict(features_test)
    if selected_target_fmt == "price_per_m2":
        test_pred_price = np.maximum(
            np.expm1(test_raw_pred) * df_test["Area"].to_numpy(),
            0.0,
        )
        lower_bound = np.maximum(
            np.expm1(test_raw_pred - residual_log_quantile) * df_test["Area"].to_numpy(),
            0.0,
        )
        upper_bound = np.expm1(test_raw_pred + residual_log_quantile) * df_test["Area"].to_numpy()
    else:
        test_pred_price = np.maximum(np.expm1(test_raw_pred), 0.0)
        lower_bound = np.maximum(
            np.expm1(test_raw_pred - residual_log_quantile),
            0.0,
        )
        upper_bound = np.expm1(test_raw_pred + residual_log_quantile)

    champion_test_metrics = regression_metrics(test_actual, test_pred_price)
    int_metrics = interval_metrics(
        test_actual,
        lower_bound,
        upper_bound,
        target_coverage=target_coverage,
    )

    # 4. Phân tích các lát cắt dữ liệu (Slices)
    slice_analysis = analyze_slices(
        df_test=df_test,
        test_actual=test_actual,
        test_pred_price=test_pred_price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        features_test=features_test,
    )

    logger.info(
        "Hoàn tất đánh giá tập Test: Champion MAE=%.1f triệu (Naive=%.1f tr, Segment=%.1f tr), Coverage=%.1f%% (Mục tiêu %.0f%%).",
        champion_test_metrics["mae_million"],
        naive_test_metrics["mae_million"],
        segment_test_metrics["mae_million"],
        int_metrics["actual_coverage"] * 100,
        target_coverage * 100,
    )

    return {
        "champion_metrics": champion_test_metrics,
        "interval_metrics": int_metrics,
        "baselines": {
            "naive_median": naive_test_metrics,
            "district_property_segment_median": segment_test_metrics,
        },
        "slice_analysis": slice_analysis,
        "test_sample_count": len(df_test),
        "test_predictions": {
            "predicted_price_million": test_pred_price.tolist(),
            "lower_bound_million": lower_bound.tolist(),
            "upper_bound_million": upper_bound.tolist(),
        },
    }
