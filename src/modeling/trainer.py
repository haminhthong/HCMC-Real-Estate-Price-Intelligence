"""Refit champion model trên tập gộp Train và Validation."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import NUMERIC_FEATURES, logger
from src.features.builder import build_features
from src.features.context import FeatureContext

from .baselines import NaiveMedianBaseline, SegmentMedianBaseline
from .pipelines import build_pipeline


def refit_champion_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    selected_model_name: str,
) -> dict[str, Any]:
    """Tái huấn luyện duy nhất Champion Model trên tập dữ liệu gộp Train + Validation (75%).

    NGUYÊN TẮC:
    1. 15% Validation sau khi dùng để chọn mô hình sẽ được gộp vào Train để tối đa hóa dữ liệu học.
    2. Cập nhật mốc thời gian tham chiếu `reference_date_final = max(listing_date của Train + Validation)`.
    3. Đóng băng `final_feature_context` làm quy chuẩn cho Calibration, Test và Serving.
    4. Lưu miền giá trị quan sát của feature để cảnh báo input nằm ngoài dữ liệu train.
    """
    logger.info("Huấn luyện lại %s trên Train + Validation", selected_model_name)

    df_train_dev = pd.concat([df_train, df_val], ignore_index=True)
    logger.info("Tập gộp Train + Validation có %d bản ghi.", len(df_train_dev))

    # 1. Cập nhật FeatureContext mới dựa trên toàn bộ 75% dữ liệu
    final_feature_context = FeatureContext.fit(df_train_dev)
    logger.info(
        "Mốc tham chiếu chuẩn hóa mới (reference_date_final): %s",
        final_feature_context.reference_date,
    )

    # 2. Xây dựng ma trận đặc trưng cho tập gộp
    features_train_dev = build_features(df_train_dev, context=final_feature_context)

    # 3. Chuẩn bị biến mục tiêu
    y_train_dev = np.log1p(df_train_dev["Price"].to_numpy())

    # 4. Refit pipeline
    champion_pipeline: Pipeline = build_pipeline(selected_model_name).fit(
        features_train_dev,
        y_train_dev,
    )
    logger.info(
        "Huấn luyện lại hoàn tất cho mô hình %s.",
        selected_model_name,
    )

    # 5. Fit các mô hình cơ sở thẩm định trên Train + Val phục vụ so sánh và bối cảnh
    naive_baseline = NaiveMedianBaseline().fit(df_train_dev)
    segment_baseline = SegmentMedianBaseline().fit(df_train_dev)

    # Bảng đơn giá trung vị theo phân khúc
    segment_unit_prices = (
        df_train_dev.assign(unit_price=df_train_dev["Price"] / df_train_dev["Area"])
        .groupby(["Property Type", "location_area"])["unit_price"]
        .median()
        .to_dict()
    )

    # Miền giá trị quan sát, chỉ dùng để phát cảnh báo input ngoại suy.
    training_ranges = {
        col: [
            float(features_train_dev[col].min()),
            float(features_train_dev[col].max()),
        ]
        for col in NUMERIC_FEATURES
        if features_train_dev[col].notna().any()
    }
    return {
        "champion_pipeline": champion_pipeline,
        "final_feature_context": final_feature_context,
        "df_train_dev": df_train_dev,
        "features_train_dev": features_train_dev,
        "naive_baseline": naive_baseline,
        "segment_baseline": segment_baseline,
        "segment_unit_prices": segment_unit_prices,
        "training_ranges": training_ranges,
        "selected_model_name": selected_model_name,
    }
