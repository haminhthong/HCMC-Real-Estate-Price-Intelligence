"""Mô-đun xây dựng tập đặc trưng chính thức (Feature Builder) cho huấn luyện và phục vụ."""

import numpy as np
import pandas as pd

from src.config import FLAG_FEATURES, MISSING_INDICATOR_FEATURES

from .context import FeatureContext
from .geospatial import calculate_distance_to_cbd
from .structural import calculate_input_completeness
from .temporal import calculate_days_from_reference, calculate_market_time_offset
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

    # 3. Cờ chỉ báo khuyết thiếu có chủ đích (Missingness Indicators)
    lat_val = (
        out["Latitude"] if "Latitude" in out else pd.Series(np.nan, index=out.index)
    )
    lon_val = (
        out["Longitude"] if "Longitude" in out else pd.Series(np.nan, index=out.index)
    )
    out["gps_missing"] = (lat_val.isna() | lon_val.isna()).astype(int)

    missing_col_mapping = {
        "Width": "width_missing",
        "Length": "length_missing",
        "Bedrooms": "bedrooms_missing",
        "Bathrooms": "bathrooms_missing",
        "Floors": "floors_missing",
        "Road Type": "road_type_missing",
        "Alley Width": "alley_width_missing",
    }
    for col, flag_name in missing_col_mapping.items():
        series = out[col] if col in out else pd.Series(np.nan, index=out.index)
        out[flag_name] = series.isna().astype(int)

    # 4. Đặc trưng thời gian và mốc định giá tham chiếu
    as_of = out.get("as_of_date")
    listing_dates = (
        out["listing_date"]
        if "listing_date" in out
        else pd.Series(pd.NaT, index=out.index)
    )

    has_as_of = (
        as_of is not None
        and not (isinstance(as_of, pd.Series) and as_of.isna().all())
        and not (not isinstance(as_of, pd.Series) and pd.isna(as_of))
    )
    if has_as_of:
        time_offset = calculate_market_time_offset(
            listing_dates, context.reference_date, as_of_date=as_of
        )
    else:
        time_offset = calculate_days_from_reference(
            listing_dates, context.reference_date
        )

    out["days_from_train_reference"] = time_offset

    # 5. Đặc trưng địa không gian
    out["distance_to_cbd_km"] = calculate_distance_to_cbd(
        lat_val,
        lon_val,
        cbd_lat=context.cbd_latitude,
        cbd_lon=context.cbd_longitude,
    )

    # 6. Căn chỉnh đầy đủ các cột đặc trưng mô hình
    target_features = context.model_features
    zero_fill_cols = set(FLAG_FEATURES) | set(MISSING_INDICATOR_FEATURES)
    for col in target_features:
        if col not in out:
            out[col] = 0 if col in zero_fill_cols else np.nan

    return out[target_features].replace([np.inf, -np.inf], np.nan)
