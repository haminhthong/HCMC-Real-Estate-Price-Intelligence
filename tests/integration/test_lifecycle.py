"""Kiểm thử tích hợp các nguyên tắc chống rò rỉ dữ liệu cốt lõi (Leakage Prevention Invariants)."""

import pandas as pd
import pytest

from src.calibration.conformal import calibrate_conformal
from src.config import MODEL_FEATURES
from src.data.cleaning import clean_data
from src.data.split import split_group_indices
from src.features.builder import build_features
from src.features.context import FeatureContext
from src.modeling.selector import select_champion_model
from src.modeling.trainer import refit_champion_model


@pytest.fixture
def lifecycle_dataset() -> pd.DataFrame:
    """Tạo tập dữ liệu mẫu đủ 100 nhóm có tin đăng lại để kiểm thử toàn bộ lifecycle."""
    records = []
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    areas = ["Quận 1", "Quận 3", "Quận 7", "Quận Bình Thạnh"]
    p_types = ["Nhà riêng", "Căn hộ chung cư", "Nhà mặt tiền"]

    for i in range(100):
        records.append(
            {
                "Price": 5000.0 + (i * 50.0),
                "Area": 50.0 + (i % 50),
                "Property Type": p_types[i % len(p_types)],
                "Location": f"{10 + i} Đường Số {i % 10}, Phường {i % 5}, {areas[i % len(areas)]}, TP.HCM",
                "Listing ID": f"list-{i}",
                "Bedrooms": 2 + (i % 4),
                "Bathrooms": 1 + (i % 3),
                "Floors": 1 + (i % 3),
                "Width": 4.0 + (i % 3),
                "Length": 12.0 + (i % 5),
                "Alley Width": 3.0,
                "Direction": "Đông",
                "Position": "Trong hẻm",
                "Latitude": 10.7769 + (i * 0.0001),
                "Longitude": 106.7009 + (i * 0.0001),
                "Last Updated Date": dates[i].strftime("%d/%m/%Y %H:%M"),
            }
        )

    # Thêm 5 tin đăng lại của cùng căn nhà ở mốc thời gian khác
    for i in range(5):
        repost = dict(records[i])
        repost["Listing ID"] = f"repost-{i}"
        repost["Price"] += 300.0
        repost["Last Updated Date"] = (dates[i] + pd.Timedelta(days=60)).strftime(
            "%d/%m/%Y %H:%M"
        )
        records.append(repost)

    return pd.DataFrame(records)


def test_same_property_never_crosses_splits(lifecycle_dataset):
    """Cùng một bất động sản không bao giờ xuất hiện ở hai tập split."""
    clean = clean_data(lifecycle_dataset)
    train_idx, val_idx, calib_idx, test_idx = split_group_indices(clean)

    train_groups = set(clean.iloc[train_idx]["property_group_id"])
    val_groups = set(clean.iloc[val_idx]["property_group_id"])
    calib_groups = set(clean.iloc[calib_idx]["property_group_id"])
    test_groups = set(clean.iloc[test_idx]["property_group_id"])

    assert train_groups.isdisjoint(val_groups), "Rò rỉ giữa Train và Validation!"
    assert train_groups.isdisjoint(calib_groups), "Rò rỉ giữa Train và Calibration!"
    assert train_groups.isdisjoint(test_groups), "Rò rỉ giữa Train và Test!"
    assert val_groups.isdisjoint(calib_groups), "Rò rỉ giữa Validation và Calibration!"
    assert val_groups.isdisjoint(test_groups), "Rò rỉ giữa Validation và Test!"
    assert calib_groups.isdisjoint(test_groups), "Rò rỉ giữa Calibration và Test!"


def test_test_split_never_affects_model_selection(lifecycle_dataset):
    """Việc tuyển chọn champion chỉ dựa vào Train và Validation."""
    clean = clean_data(lifecycle_dataset)
    train_idx, val_idx, _, _ = split_group_indices(clean)

    df_train = clean.iloc[train_idx]
    df_val = clean.iloc[val_idx]

    selection_result = select_champion_model(df_train, df_val)

    assert "selected_model_name" in selection_result
    assert "best_val_mae" in selection_result
    # Không hề chứa bất kỳ thông tin nào của Test
    assert "test" not in str(selection_result["validation_benchmarks"]).lower()


def test_training_and_serving_feature_vectors_match(lifecycle_dataset):
    """Vector đặc trưng của training và serving phải khớp hoàn toàn với MODEL_FEATURES."""
    clean = clean_data(lifecycle_dataset)
    ctx = FeatureContext.fit(clean)

    train_feats = build_features(clean.iloc[:5], context=ctx)
    serving_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 85.0,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Floors": 2,
        "Width": 4.5,
        "Length": 18.0,
        "Alley Width": 3.5,
        "Latitude": 10.7769,
        "Longitude": 106.7009,
    }
    serving_feats = build_features(pd.DataFrame([serving_input]), context=ctx)

    assert train_feats.columns.tolist() == serving_feats.columns.tolist()
    assert train_feats.columns.tolist() == MODEL_FEATURES
    assert len(train_feats.columns) == len(serving_feats.columns)


def test_champion_is_refit_before_calibration(lifecycle_dataset):
    """Refit champion trên tập Train+Val trước khi tính conformal calibration."""
    clean = clean_data(lifecycle_dataset)
    train_idx, val_idx, calib_idx, _ = split_group_indices(clean)

    df_train = clean.iloc[train_idx]
    df_val = clean.iloc[val_idx]
    df_calib = clean.iloc[calib_idx]

    selection_result = select_champion_model(df_train, df_val)

    refit_result = refit_champion_model(
        df_train=df_train,
        df_val=df_val,
        selected_model_name=selection_result["selected_model_name"],
    )

    champion_pipe = refit_result["champion_pipeline"]
    final_ctx = refit_result["final_feature_context"]

    assert len(refit_result["df_train_dev"]) == len(train_idx) + len(val_idx)

    calib_feats = build_features(df_calib, context=final_ctx)
    calib_result = calibrate_conformal(
        champion_pipe,
        calib_feats,
        df_calib,
    )
    assert calib_result["residual_log_quantile"] > 0
    assert calib_result["calibration_samples"] == len(calib_idx)
