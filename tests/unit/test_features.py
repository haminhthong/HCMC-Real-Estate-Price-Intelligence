"""Kiểm thử đơn vị cho mô-đun Features, FeatureContext và Pipeline Preprocessing."""

import numpy as np
import pandas as pd

from src.config import MISSING_INDICATOR_FEATURES, MODEL_FEATURES
from src.features.builder import build_features
from src.features.context import FeatureContext
from src.features.geospatial import calculate_distance_to_cbd
from src.features.text import add_text_flags
from src.modeling.pipelines import build_pipeline


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


def test_target_is_not_a_feature(sample_raw_dataframe):
    features = build_features(sample_raw_dataframe)
    assert "Price" not in features.columns
    assert list(features.columns) == MODEL_FEATURES


def test_preprocessing_has_no_nonfinite_values(sample_raw_dataframe):
    features = build_features(sample_raw_dataframe)
    transformed = build_pipeline().named_steps["preprocessor"].fit_transform(features)
    assert np.isfinite(transformed).all()


def test_feature_count_is_stable_after_transform(sample_raw_dataframe):
    features = build_features(sample_raw_dataframe)
    prep = build_pipeline().named_steps["preprocessor"].fit(features)
    assert prep.transform(features).shape[1] == len(prep.get_feature_names_out())


def test_supplied_amenities_are_not_overwritten():
    frame = pd.DataFrame([{"has_furniture": True, "car_alley": True}])
    features = build_features(frame)
    assert features.loc[0, "has_furniture"] == 1
    assert features.loc[0, "car_alley"] == 1
