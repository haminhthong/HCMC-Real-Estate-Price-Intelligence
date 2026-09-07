"""Kiểm thử đơn vị cho mô-đun Features và FeatureContext."""

import pandas as pd
from src.features.builder import build_features
from src.features.context import FeatureContext
from src.features.geospatial import calculate_distance_to_cbd
from src.features.text import add_text_flags


def test_distance_to_cbd_calculation():
    # Tọa độ ngay Chợ Bến Thành
    lat = pd.Series([10.7769])
    lon = pd.Series([106.7009])
    dist = calculate_distance_to_cbd(lat, lon, cbd_lat=10.7769, cbd_lon=106.7009)
    assert dist.iloc[0] < 0.01, "Khoảng cách tại tọa độ trung tâm phải xấp xỉ 0 km!"


def test_text_flags_with_negation():
    df = pd.DataFrame(
        [
            {"Title": "Nhà", "Description": "nhà không có nội thất, hẻm xe hơi"},
            {"Title": "Nhà", "Description": "full nội thất gỗ cao cấp"},
        ]
    )
    flagged = add_text_flags(df)
    assert flagged["has_furniture"].iloc[0] == 0
    assert flagged["car_alley"].iloc[0] == 1
    assert flagged["has_furniture"].iloc[1] == 1


def test_feature_context_serialization():
    ctx = FeatureContext(reference_date="2025-09-01T00:00:00")
    d = ctx.to_dict()
    restored = FeatureContext.from_dict(d)
    assert restored.reference_date == ctx.reference_date
    assert restored.cbd_latitude == ctx.cbd_latitude


def test_as_of_date_and_missing_indicators():
    from src.features.builder import build_features
    from src.config import MISSING_INDICATOR_FEATURES

    ctx = FeatureContext(
        reference_date="2025-01-01T00:00:00",
        missing_indicator_features=list(MISSING_INDICATOR_FEATURES),
    )
    df = pd.DataFrame(
        [
            {
                "Property Type": "Nhà riêng",
                "location_area": "Quận 1",
                "Area": 80.0,
                # Missing bedrooms, bathrooms, width, length, GPS
                "as_of_date": "2025-04-11T00:00:00",
            }
        ]
    )
    feats = build_features(df, context=ctx)
    assert feats["days_from_train_reference"].iloc[0] == 100.0
    assert feats["gps_missing"].iloc[0] == 1
    assert feats["width_missing"].iloc[0] == 1
    assert feats["length_missing"].iloc[0] == 1
    assert feats["bedrooms_missing"].iloc[0] == 1
    assert feats["bathrooms_missing"].iloc[0] == 1

