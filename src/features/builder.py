"""Mô-đun xây dựng tập đặc trưng chính thức (Feature Builder) cho huấn luyện và phục vụ."""

from typing import Any
import numpy as np
import pandas as pd

from src.config import FLAG_FEATURES, MODEL_FEATURES
from .context import FeatureContext
from .geospatial import calculate_distance_to_cbd
from .structural import calculate_input_completeness
from .temporal import calculate_days_from_reference
from .text import add_text_flags


def build_features(
    df: pd.DataFrame,
    context: FeatureContext | None = None,
) -> pd.DataFrame:
    """Tạo tập đặc trưng hoàn chỉnh và khớp với schema mô hình yêu cầu.

    Quy trình chuẩn hóa:
    1. Trích xuất cờ tiện ích văn bản (có xử lý phủ định).
    2. Đo lường điểm hoàn thiện thông tin đầu vào (`input_completeness_score`).
    3. Tính độ lệch ngày so với mốc thời gian tham chiếu đóng băng (`days_from_train_reference`).
    4. Tính khoảng cách Haversine tới trung tâm TP.HCM (`distance_to_cbd_km`).
    5. Căn chỉnh chính xác danh sách và thứ tự các cột đặc trưng mô hình yêu cầu.

    Args:
        df: DataFrame chứa thông tin bất động sản.
        context: Gói FeatureContext đã được fit từ tập huấn luyện. Nếu None, sẽ fit tự động.

    Returns:
        DataFrame chỉ chứa các cột đặc trưng mô hình, không chứa giá hoặc target.
    """
    if context is None:
        context = FeatureContext.fit(df)

    out = df.copy()

    # 1. Cờ tiện ích văn bản
    out = add_text_flags(out)

    # 2. Điểm hoàn thiện dữ liệu
    completeness = calculate_input_completeness(out)
    out["input_completeness_score"] = completeness
    out["data_quality_score"] = completeness

    # 3. Đặc trưng thời gian
    listing_dates = out["listing_date"] if "listing_date" in out else pd.Series(pd.NaT, index=out.index)
    out["days_from_train_reference"] = calculate_days_from_reference(
        listing_dates,
        context.reference_date,
    )
    out["listing_age_days"] = out["days_from_train_reference"]

    # 4. Đặc trưng địa không gian
    lat = out["Latitude"] if "Latitude" in out else pd.Series(np.nan, index=out.index)
    lon = out["Longitude"] if "Longitude" in out else pd.Series(np.nan, index=out.index)
    out["distance_to_cbd_km"] = calculate_distance_to_cbd(
        lat,
        lon,
        cbd_lat=context.cbd_latitude,
        cbd_lon=context.cbd_longitude,
    )

    # 5. Căn chỉnh đầy đủ các cột đặc trưng mô hình
    target_features = context.model_features if context else MODEL_FEATURES
    for col in target_features:
        if col not in out:
            out[col] = 0 if col in FLAG_FEATURES else np.nan

    return out[target_features].replace([np.inf, -np.inf], np.nan)


def add_quality_features(
    df: pd.DataFrame,
    reference_date: Any = None,
) -> pd.DataFrame:
    """Hàm phụ trợ tương thích ngược tính toán độ hoàn thiện và tuổi tin đăng."""
    out = df.copy()
    completeness = calculate_input_completeness(out)
    out["input_completeness_score"] = completeness
    out["data_quality_score"] = completeness

    listing_dates = out["listing_date"] if "listing_date" in out else pd.Series(pd.NaT, index=out.index)
    out["days_from_train_reference"] = calculate_days_from_reference(listing_dates, reference_date)
    out["listing_age_days"] = out["days_from_train_reference"]
    return out


def make_features(
    df: pd.DataFrame,
    reference_date: Any = None,
) -> pd.DataFrame:
    """Hàm ủy nhiệm tương thích ngược tạo đặc trưng với mốc reference_date đơn lẻ."""
    if reference_date is not None:
        ref_str = (
            reference_date.isoformat()
            if hasattr(reference_date, "isoformat")
            else str(reference_date)
        )
        ctx = FeatureContext(reference_date=ref_str)
    else:
        ctx = FeatureContext.fit(df)

    return build_features(df, context=ctx)
