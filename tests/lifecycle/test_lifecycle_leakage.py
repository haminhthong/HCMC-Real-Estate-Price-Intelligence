"""Bộ kiểm thử 5 nguyên tắc Lifecycle & Chống rò rỉ dữ liệu (Anti-Leakage) cốt lõi."""

import pandas as pd

from src.calibration.conformal import calibrate_conformal
from src.config import DATA_PATH, MODEL_FEATURES
from src.data.cleaning import clean_data
from src.data.loader import load_raw_dataset
from src.data.split import split_group_indices
from src.features.builder import build_features, make_features
from src.features.context import FeatureContext
from src.modeling.selector import select_champion_model
from src.modeling.trainer import refit_champion_model


def test_same_property_never_crosses_splits():
    """1. NGUYÊN TẮC: Cùng một bất động sản tuyệt đối không bao giờ xuất hiện ở 2 tập split khác nhau."""
    raw = load_raw_dataset(DATA_PATH)
    clean = clean_data(raw)
    train_idx, val_idx, calib_idx, test_idx = split_group_indices(clean)

    train_groups = set(clean.iloc[train_idx]["property_group_id"])
    val_groups = set(clean.iloc[val_idx]["property_group_id"])
    calib_groups = set(clean.iloc[calib_idx]["property_group_id"])
    test_groups = set(clean.iloc[test_idx]["property_group_id"])

    assert train_groups.isdisjoint(val_groups), "Leakage giữa Train và Validation!"
    assert train_groups.isdisjoint(calib_groups), "Leakage giữa Train và Calibration!"
    assert train_groups.isdisjoint(test_groups), "Leakage giữa Train và Test!"
    assert val_groups.isdisjoint(calib_groups), "Leakage giữa Validation và Calibration!"
    assert val_groups.isdisjoint(test_groups), "Leakage giữa Validation và Test!"
    assert calib_groups.isdisjoint(test_groups), "Leakage giữa Calibration và Test!"


def test_repeated_property_listings_are_not_removed_as_duplicates():
    """2. NGUYÊN TẮC: Các tin đăng lặp lại theo thời gian của cùng một căn nhà KHÔNG bị drop làm mất nhóm.

    Chỉ loại bỏ tin đăng trùng lặp hoàn toàn (cùng property_group_id, listing_date, Price).
    """
    raw = load_raw_dataset(DATA_PATH)
    clean = clean_data(raw)
    audit = getattr(clean, "attrs", {}).get("data_audit", {})

    # Trong tập dữ liệu thực tế, số lượng clean listings phải lớn hơn unique_property_groups
    unique_groups = clean["property_group_id"].nunique()
    total_clean = len(clean)
    assert total_clean > unique_groups, (
        f"Lỗi logic: clean listings ({total_clean}) không được bằng unique groups ({unique_groups}). "
        f"Multi-listing của cùng 1 property phải được giữ lại!"
    )
    assert audit.get("multi_listing_groups_count", 0) > 0


def test_test_split_never_affects_model_selection():
    """3. NGUYÊN TẮC: Việc lựa chọn Champion model chỉ dựa vào Validation, Test set hoàn toàn cô lập."""
    raw = load_raw_dataset(DATA_PATH)
    clean = clean_data(raw)
    train_idx, val_idx, _, _ = split_group_indices(clean)

    df_train = clean.iloc[train_idx]
    df_val = clean.iloc[val_idx]

    # Hàm select_champion_model chỉ nhận df_train và df_val
    selection_result = select_champion_model(df_train, df_val)

    assert "selected_model_name" in selection_result
    assert "selected_target_fmt" in selection_result
    assert "best_val_mae" in selection_result

    # Không hề chứa bất kỳ metric nào của Test trong quá trình selection
    assert "test" not in str(selection_result["validation_benchmarks"]).lower()


def test_training_and_serving_feature_vectors_match():
    """4. NGUYÊN TẮC: Vector đặc trưng khi huấn luyện và khi serving khớp nhau 100% (chống skew)."""
    raw = load_raw_dataset(DATA_PATH)
    clean = clean_data(raw)
    ctx = FeatureContext.fit(clean)

    # 1. Feature frame trong huấn luyện
    train_feats = build_features(clean.iloc[:5], context=ctx)

    # 2. Feature frame trong serving
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
    serving_feats = make_features(pd.DataFrame([serving_input]), reference_date=ctx.reference_date)

    assert train_feats.columns.tolist() == serving_feats.columns.tolist(), "Cột đặc trưng không khớp giữa Train và Serving!"
    assert train_feats.columns.tolist() == MODEL_FEATURES, "Không khớp với danh sách MODEL_FEATURES chuẩn!"
    assert len(train_feats.columns) == len(serving_feats.columns)


def test_champion_is_refit_before_calibration():
    """5. NGUYÊN TẮC: Champion model phải được refit trên Train+Val (75%) trước khi đưa vào Calibration."""
    raw = load_raw_dataset(DATA_PATH)
    clean = clean_data(raw)
    train_idx, val_idx, calib_idx, _ = split_group_indices(clean)

    df_train = clean.iloc[train_idx]
    df_val = clean.iloc[val_idx]
    df_calib = clean.iloc[calib_idx]

    # Phase A: Tuyển chọn
    selection_result = select_champion_model(df_train, df_val)

    # Phase B: Refit Champion trên Train + Validation
    refit_result = refit_champion_model(
        df_train=df_train,
        df_val=df_val,
        selected_model_name=selection_result["selected_model_name"],
        selected_target_fmt=selection_result["selected_target_fmt"],
    )

    champion_pipe = refit_result["champion_pipeline"]
    final_ctx = refit_result["final_feature_context"]

    # Số mẫu của tập Train+Val phải bằng tổng len(train_idx) + len(val_idx)
    assert len(refit_result["df_train_dev"]) == len(train_idx) + len(val_idx)

    # Conformal Calibration sử dụng model đã refit
    calib_feats = build_features(df_calib, context=final_ctx)
    calib_result = calibrate_conformal(
        champion_pipe,
        calib_feats,
        df_calib,
        target_formulation=selection_result["selected_target_fmt"],
    )
    assert calib_result["residual_log_quantile"] > 0
    assert calib_result["calibration_samples"] == len(calib_idx)
