"""Mô-đun xây dựng tập đặc trưng chính thức (Feature Builder) cho huấn luyện và phục vụ."""

from typing import Any

import numpy as np
import pandas as pd

from src.config import FLAG_FEATURES, MISSING_INDICATOR_FEATURES, MODEL_FEATURES

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

    # 2. Điểm hoàn thiện dữ liệu (input_completeness_score là chuẩn canonical)
    completeness = calculate_input_completeness(out)
    out["input_completeness_score"] = completeness
    # Alias tương thích ngược cũ (đã deprecate)
    out["data_quality_score"] = completeness

    # 3. Cờ chỉ báo khuyết thiếu có chủ đích (Missingness Indicators)
    lat_val = out["Latitude"] if "Latitude" in out else pd.Series(np.nan, index=out.index)
    lon_val = out["Longitude"] if "Longitude" in out else pd.Series(np.nan, index=out.index)
    out["gps_missing"] = (lat_val.isna() | lon_val.isna()).astype(int)
    out["width_missing"] = (out["Width"].isna() if "Width" in out else pd.Series(1, index=out.index)).astype(int)
    out["length_missing"] = (out["Length"].isna() if "Length" in out else pd.Series(1, index=out.index)).astype(int)
    out["bedrooms_missing"] = (out["Bedrooms"].isna() if "Bedrooms" in out else pd.Series(1, index=out.index)).astype(int)
    out["bathrooms_missing"] = (out["Bathrooms"].isna() if "Bathrooms" in out else pd.Series(1, index=out.index)).astype(int)
    out["floors_missing"] = (out["Floors"].isna() if "Floors" in out else pd.Series(1, index=out.index)).astype(int)
    out["road_type_missing"] = (out["Road Type"].isna() if "Road Type" in out else pd.Series(1, index=out.index)).astype(int)
    out["alley_width_missing"] = (out["Alley Width"].isna() if "Alley Width" in out else pd.Series(1, index=out.index)).astype(int)

    # 4. Đặc trưng thời gian và mốc định giá tham chiếu
    as_of = out.get("as_of_date", out.get("valuation_date"))
    listing_dates = out["listing_date"] if "listing_date" in out else pd.Series(pd.NaT, index=out.index)
    
    has_as_of = as_of is not None and not (isinstance(as_of, pd.Series) and as_of.isna().all()) and not (not isinstance(as_of, pd.Series) and pd.isna(as_of))
    if has_as_of:
        from .temporal import calculate_market_time_offset
        time_offset = calculate_market_time_offset(listing_dates, context.reference_date, as_of_date=as_of)
    else:
        time_offset = calculate_days_from_reference(listing_dates, context.reference_date)

    out["days_from_train_reference"] = time_offset
    out["market_time_offset_days"] = time_offset
    out["listing_age_days"] = time_offset

    # 5. Đặc trưng địa không gian
    out["distance_to_cbd_km"] = calculate_distance_to_cbd(
        lat_val,
        lon_val,
        cbd_lat=context.cbd_latitude,
        cbd_lon=context.cbd_longitude,
    )

    # 6. Căn chỉnh đầy đủ các cột đặc trưng mô hình
    target_features = context.model_features if context else MODEL_FEATURES
    zero_fill_cols = set(FLAG_FEATURES) | set(MISSING_INDICATOR_FEATURES)
    for col in target_features:
        if col not in out:
            out[col] = 0 if col in zero_fill_cols else np.nan

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
