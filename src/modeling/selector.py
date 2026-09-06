"""Mô-đun Phase A: Tuyển chọn mô hình Champion trên tập Validation (Model Selection)."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import logger
from src.evaluation.metrics import regression_metrics
from src.features.builder import build_features
from src.features.context import FeatureContext
from .baselines import SegmentMedianBaseline
from .candidates import CANDIDATE_MODELS, TARGET_FORMULATIONS
from .pipelines import build_pipeline


def select_champion_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> dict[str, Any]:
    """Huấn luyện các candidate models trên tập Train (60%) và chọn lọc trên tập Validation (15%).

    NGUYÊN TẮC:
    1. Chỉ sử dụng mốc thời gian `reference_date_selection` từ tập Train để chống data leakage.
    2. Đánh giá đa chiều trên tập Validation (MAE, WAPE, Median AE, RMSE, R²).
    3. Chọn mô hình Champion và Target formulation tốt nhất mà KHÔNG đụng tới tập Test.
    """
    logger.info("--- BẮT ĐẦU PHASE A: MODEL SELECTION TRÊN TẬP VALIDATION ---")

    # 1. Khởi tạo FeatureContext chỉ từ tập Train
    feature_ctx = FeatureContext.fit(df_train)
    features_train = build_features(df_train, context=feature_ctx)
    features_val = build_features(df_val, context=feature_ctx)

    val_actual = df_val["Price"].to_numpy()

    # 2. Đánh giá Baseline Segment Median trên Validation
    segment_baseline = SegmentMedianBaseline().fit(df_train)
    segment_val_preds = segment_baseline.predict(df_val)
    segment_val_metrics = regression_metrics(val_actual, segment_val_preds)

    validation_benchmarks: dict[str, dict[str, dict[str, float]]] = {
        fmt: {} for fmt in TARGET_FORMULATIONS
    }
    fitted_selection_pipes: dict[tuple[str, str], Pipeline] = {}

    for fmt in TARGET_FORMULATIONS:
        if fmt == "total_price":
            y_train = np.log1p(df_train["Price"])
        else:
            y_train = np.log1p(df_train["Price"] / df_train["Area"])

        for m_name in CANDIDATE_MODELS:
            pipe = build_pipeline(m_name).fit(features_train, y_train)
            fitted_selection_pipes[(fmt, m_name)] = pipe

            val_raw_pred = pipe.predict(features_val)
            if fmt == "price_per_m2":
                val_pred_price = np.maximum(
                    np.expm1(val_raw_pred) * df_val["Area"].to_numpy(),
                    0.0,
                )
            else:
                val_pred_price = np.maximum(np.expm1(val_raw_pred), 0.0)

            validation_benchmarks[fmt][m_name] = regression_metrics(
                val_actual, val_pred_price
            )

    # Ghi nhận baseline segment vào kết quả validation
    for fmt in TARGET_FORMULATIONS:
        validation_benchmarks[fmt]["district_property_segment_median"] = segment_val_metrics

    # 3. Lựa chọn mô hình Champion dựa trên Validation MAE (và ghi nhận WAPE, Median AE)
    best_combo = None
    best_val_mae = float("inf")

    for fmt in TARGET_FORMULATIONS:
        for m_name in CANDIDATE_MODELS:
            mae = validation_benchmarks[fmt][m_name]["mae_million"]
            if mae < best_val_mae:
                best_val_mae = mae
                best_combo = (fmt, m_name)

    selected_target_fmt, selected_model_name = best_combo
    naive_val_mae = validation_benchmarks["total_price"]["naive_median"]["mae_million"]

    logger.info(
        "Kết thúc Phase A: Đã chọn Champion model '%s' (target=%s) với Val MAE=%.1f triệu (Naive=%.1f triệu).",
        selected_model_name,
        selected_target_fmt,
        best_val_mae,
        naive_val_mae,
    )

    return {
        "selected_model_name": selected_model_name,
        "selected_target_fmt": selected_target_fmt,
        "best_val_mae": best_val_mae,
        "naive_val_mae": naive_val_mae,
        "validation_benchmarks": validation_benchmarks,
        "reference_date_selection": feature_ctx.reference_date,
    }
