"""Mô-đun phân chia tập dữ liệu cô lập nhóm bất động sản theo thứ tự thời gian (Group-isolated Temporal Ordering Split)."""

from typing import Any

import numpy as np
import pandas as pd


def _positions(mask: pd.Series) -> np.ndarray:
    """Đổi boolean mask thành vị trí nguyên để dùng an toàn với ``DataFrame.iloc``."""
    return np.flatnonzero(mask.to_numpy(dtype=bool))


def split_group_indices(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chia tập dữ liệu theo nhóm bất động sản và mốc thời gian (Group-isolated Temporal Ordering Split 60/15/10/15).

    NGUYÊN LÝ CHỐNG LEAKAGE & CÔ LẬP NHÓM BẤT ĐỘNG SẢN:
    1. Nhóm toàn bộ tin đăng theo `property_group_id`.
    2. Lấy ngày đăng bài muộn nhất của từng nhóm (`latest listing_date per group`) và sắp xếp tăng dần.
    3. Phân chia danh sách nhóm theo tỷ lệ 60/15/10/15:
       - Train (60%): Huấn luyện ứng viên và fit tiền xử lý.
       - Validation (15%): So sánh đa mô hình, chọn target formulation.
       - Calibration (10%): Tính phần dư conformal prediction trong log space.
       - Test (15%): Đánh giá kiểm thử độc lập cuối cùng.

    Bảo đảm: Toàn bộ tin đăng của cùng một bất động sản luôn nằm chung trong 1 split duy nhất,
    tương ứng với ngày đăng mới nhất của bất động sản đó (Group Isolation).
    LƯU Ý NGỮ NGHĨA: Đây là Group-isolated temporal ordering split, không phải strict row-level temporal holdout,
    vì nếu một căn nhà có tin đăng cũ vào tháng 1 và tin đăng mới nhất vào tháng 9, toàn bộ tin đăng của căn đó
    sẽ được gán theo mốc tháng 9.
    """
    if "property_group_id" not in df or "listing_date" not in df:
        raise ValueError("DataFrame phải chứa các cột 'property_group_id' và 'listing_date' để thực hiện split.")

    # Date unknown không được phép rơi vào cuối chuỗi như thể là dữ liệu mới nhất.
    dated_df = df.loc[df["listing_date"].notna()].copy()
    if dated_df.empty:
        raise ValueError("Không thể split: không có listing_date hợp lệ.")

    ordered_groups = (
        dated_df.groupby("property_group_id", as_index=False)["listing_date"]
        .max()
        .sort_values(["listing_date", "property_group_id"])
    )
    number_of_groups = len(ordered_groups)
    if number_of_groups < 7:
        raise ValueError(
            "Cần ít nhất 7 property_group_id có ngày hợp lệ để tạo đủ 4 tập temporal không rỗng."
        )
    train_end = int(number_of_groups * 0.60)
    validation_end = int(number_of_groups * 0.75)
    calibration_end = int(number_of_groups * 0.85)

    train_groups: set[str] = set(
        ordered_groups.iloc[:train_end]["property_group_id"]
    )
    validation_groups: set[str] = set(
        ordered_groups.iloc[train_end:validation_end]["property_group_id"]
    )
    calibration_groups: set[str] = set(
        ordered_groups.iloc[validation_end:calibration_end]["property_group_id"]
    )
    test_groups: set[str] = set(
        ordered_groups.iloc[calibration_end:]["property_group_id"]
    )

    train_idx = _positions(df["property_group_id"].isin(train_groups) & df["listing_date"].notna())
    validation_idx = _positions(
        df["property_group_id"].isin(validation_groups) & df["listing_date"].notna()
    )
    calibration_idx = _positions(
        df["property_group_id"].isin(calibration_groups) & df["listing_date"].notna()
    )
    test_idx = _positions(df["property_group_id"].isin(test_groups) & df["listing_date"].notna())

    # Kiểm tra tính toàn vẹn (Disjointness test giữa cả 4 tập)
    group_sets = [
        set(df.iloc[idx]["property_group_id"])
        for idx in (train_idx, validation_idx, calibration_idx, test_idx)
    ]
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[0] & group_sets[3]
        or group_sets[1] & group_sets[2]
        or group_sets[1] & group_sets[3]
        or group_sets[2] & group_sets[3]
    ):
        raise RuntimeError("CẢNH BÁO LEAKAGE: Phát hiện nhóm bất động sản bị trùng giữa các tập split!")

    return train_idx, validation_idx, calibration_idx, test_idx


def split_strict_temporal_purged(
    df: pd.DataFrame,
    train_val_ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Protocol B: Phân chia theo mốc thời gian tuyệt đối (Strict Temporal Cutoff) kèm loại bỏ rò rỉ nhóm.

    Quy trình:
    1. Xác định mốc thời gian phân tách theo phân vị `train_val_ratio` của ngày đăng `listing_date`.
    2. Tập Train sơ bộ: các tin đăng có `listing_date < cutoff_date`.
    3. Tập Test sơ bộ: các tin đăng có `listing_date >= cutoff_date`.
    4. Purge (Loại trừ): Xác định các `property_group_id` xuất hiện ở CẢ HAI phía mốc cutoff,
       và loại bỏ chúng khỏi tập Test để triệt tiêu hoàn toàn rò rỉ thông tin quá khứ-tương lai của cùng căn nhà.

    Returns:
        tuple gồm (train_indices, test_purged_indices, audit_dict).
    """
    if "property_group_id" not in df or "listing_date" not in df:
        raise ValueError("DataFrame phải chứa 'property_group_id' và 'listing_date'.")

    dated_df = df.loc[df["listing_date"].notna()].copy()
    if dated_df.empty:
        raise ValueError("Không thể split: không có listing_date hợp lệ.")
    if not 0 < train_val_ratio < 1:
        raise ValueError("train_val_ratio phải nằm trong khoảng (0, 1).")

    sorted_dates = dated_df["listing_date"].sort_values().reset_index(drop=True)
    cutoff_idx = min(max(int(len(sorted_dates) * train_val_ratio), 0), len(sorted_dates) - 1)
    cutoff_date = sorted_dates.iloc[cutoff_idx]

    train_mask = df["listing_date"].notna() & (df["listing_date"] < cutoff_date)
    test_mask = df["listing_date"].notna() & (df["listing_date"] >= cutoff_date)

    train_groups = set(df.loc[train_mask, "property_group_id"])
    raw_test_groups = set(df.loc[test_mask, "property_group_id"])
    overlapping_groups = train_groups & raw_test_groups

    # Purge overlapping groups from test set to preserve strict temporal anti-leakage
    clean_test_mask = test_mask & (~df["property_group_id"].isin(overlapping_groups))

    train_indices = _positions(train_mask)
    test_indices = _positions(clean_test_mask)

    audit = {
        "protocol": "strict_temporal_purged",
        "cutoff_date": str(cutoff_date),
        "train_rows": len(train_indices),
        "test_raw_rows": int(test_mask.sum()),
        "test_purged_rows": len(test_indices),
        "purged_overlapping_groups_count": len(overlapping_groups),
        "purged_rows_count": int(test_mask.sum() - len(test_indices)),
    }

    return train_indices, test_indices, audit


def split_canonical_temporal_purged(
    df: pd.DataFrame,
    development_ratio: float = 0.70,
    calibration_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Chạy alternative experiment 70/10/20 theo thời gian tuyệt đối.

    Các property xuất hiện ở block trước sẽ bị purge khỏi block sau. Dòng ngày
    unknown bị loại khỏi benchmark thay vì được coi là ngày mới nhất. Tên hàm
    được giữ để tương thích ngược; master pipeline không gọi protocol này.
    """
    required = {"property_group_id", "listing_date"}
    if not required.issubset(df.columns):
        raise ValueError("DataFrame phải chứa 'property_group_id' và 'listing_date'.")
    if development_ratio <= 0 or calibration_ratio <= 0 or development_ratio + calibration_ratio >= 1:
        raise ValueError("Tỷ lệ development/calibration không hợp lệ.")

    dated = df.loc[df["listing_date"].notna()].copy()
    if dated.empty:
        raise ValueError("Không thể split: không có listing_date hợp lệ.")

    unique_dates = np.sort(dated["listing_date"].drop_duplicates().to_numpy())
    dev_cutoff = unique_dates[min(max(int(len(unique_dates) * development_ratio), 0), len(unique_dates) - 1)]
    calibration_cutoff = unique_dates[
        min(max(int(len(unique_dates) * (development_ratio + calibration_ratio)), 0), len(unique_dates) - 1)
    ]

    development_mask = dated["listing_date"] < dev_cutoff
    calibration_mask = (dated["listing_date"] >= dev_cutoff) & (dated["listing_date"] < calibration_cutoff)
    test_mask = dated["listing_date"] >= calibration_cutoff

    development_groups = set(dated.loc[development_mask, "property_group_id"])
    calibration_groups = set(dated.loc[calibration_mask, "property_group_id"])
    calibration_groups -= development_groups
    test_groups = set(dated.loc[test_mask, "property_group_id"])
    test_groups -= development_groups | calibration_groups

    dated_positions = np.flatnonzero(df["listing_date"].notna().to_numpy(dtype=bool))
    development_indices = dated_positions[development_mask.to_numpy(dtype=bool)]
    calibration_indices = dated_positions[
        (calibration_mask & dated["property_group_id"].isin(calibration_groups)).to_numpy(dtype=bool)
    ]
    test_indices = dated_positions[
        (test_mask & dated["property_group_id"].isin(test_groups)).to_numpy(dtype=bool)
    ]

    return development_indices, calibration_indices, test_indices, {
        "protocol": "strict_temporal_property_purged_70_10_20",
        "development_cutoff": str(dev_cutoff),
        "calibration_cutoff": str(calibration_cutoff),
        "development_rows": len(development_indices),
        "calibration_rows": len(calibration_indices),
        "test_rows": len(test_indices),
        "unknown_date_rows_excluded": int(df["listing_date"].isna().sum()),
        "calibration_groups_purged": int((set(dated.loc[calibration_mask, "property_group_id"]) - calibration_groups).__len__()),
        "test_groups_purged": int((set(dated.loc[test_mask, "property_group_id"]) - test_groups).__len__()),
    }


def compare_split_protocols(df: pd.DataFrame) -> dict[str, Any]:
    """So sánh đặc tính phân phối giữa Protocol A (Group-isolated Temporal) và Protocol B (Strict Temporal Purged)."""
    t_idx, v_idx, c_idx, te_idx = split_group_indices(df)
    b_train_idx, b_test_idx, b_audit = split_strict_temporal_purged(df, train_val_ratio=0.75)

    return {
        "protocol_a_group_isolated": {
            "name": "Group-isolated temporal ordering split",
            "train_rows": len(t_idx),
            "validation_rows": len(v_idx),
            "calibration_rows": len(c_idx),
            "test_rows": len(te_idx),
            "guarantees": "100% group isolation across all 4 sets; ordered by latest observed date per property",
        },
        "protocol_b_strict_temporal": {
            "name": "Strict temporal holdout with overlapping group purge",
            "cutoff_date": b_audit["cutoff_date"],
            "train_dev_rows": len(b_train_idx),
            "test_rows": len(b_test_idx),
            "purged_groups_count": b_audit["purged_overlapping_groups_count"],
            "guarantees": "Strict chronological separation with zero group leakage across time boundary",
        },
    }
