"""Chọn mô hình theo MAE trên tập xác thực, với mục tiêu log1p(Price)."""

from typing import Any

import numpy as np
import pandas as pd

from src.config import logger
from src.evaluation.metrics import regression_metrics
from src.features.builder import build_features
from src.features.context import FeatureContext

from .baselines import SegmentMedianBaseline
from .candidates import CANDIDATE_MODELS
from .pipelines import build_pipeline


def select_champion_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> dict[str, Any]:
    """Huấn luyện trên Train và chọn theo Validation; không đọc Calibration/Test."""
    logger.info("Chọn mô hình theo MAE trên tập xác thực")
    feature_ctx = FeatureContext.fit(df_train)
    features_train = build_features(df_train, context=feature_ctx)
    features_val = build_features(df_val, context=feature_ctx)
    y_train = np.log1p(df_train["Price"].to_numpy())
    val_actual = df_val["Price"].to_numpy()

    benchmarks = {}
    for model_name in CANDIDATE_MODELS:
        pipe = build_pipeline(model_name).fit(features_train, y_train)
        prediction = np.maximum(np.expm1(pipe.predict(features_val)), 0.0)
        benchmarks[model_name] = regression_metrics(val_actual, prediction)

    # Trung vị phân khúc chỉ là mốc so sánh, không tham gia chọn pipeline.
    segment_baseline = SegmentMedianBaseline().fit(df_train)
    benchmarks["district_property_segment_median"] = regression_metrics(
        val_actual, segment_baseline.predict(df_val)
    )
    selected_model_name = min(
        CANDIDATE_MODELS, key=lambda name: benchmarks[name]["mae_million"]
    )
    selected_metrics = benchmarks[selected_model_name]
    logger.info(
        "Đã chọn %s, MAE xác thực %.1f triệu VND",
        selected_model_name,
        selected_metrics["mae_million"],
    )
    return {
        "selected_model_name": selected_model_name,
        "best_val_mae": selected_metrics["mae_million"],
        "naive_val_mae": benchmarks["naive_median"]["mae_million"],
        "selected_validation_metrics": selected_metrics,
        "validation_benchmarks": benchmarks,
        "reference_date_selection": feature_ctx.reference_date,
    }
