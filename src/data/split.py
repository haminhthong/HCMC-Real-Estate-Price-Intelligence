"""Mô-đun chia tập dữ liệu cô lập nhóm bất động sản theo dòng thời gian (Grouped Temporal Split)."""

import numpy as np
import pandas as pd


def split_group_indices(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chia tập dữ liệu theo nhóm bất động sản và mốc thời gian (Grouped Temporal Split 60/15/10/15).

    NGUYÊN LÝ CHỐNG LEAKAGE & CÔ LẬP NHÓM BẤT ĐỘNG SẢN:
    1. Nhóm toàn bộ tin đăng theo `property_group_id`.
    2. Lấy ngày đăng bài muộn nhất của từng nhóm (`latest listing_date per group`) và sắp xếp tăng dần.
    3. Phân chia danh sách nhóm theo tỷ lệ 60/15/10/15:
       - Train (60%): Huấn luyện ứng viên và fit tiền xử lý.
       - Validation (15%): So sánh đa mô hình, chọn target formulation.
       - Calibration (10%): Tính phần dư conformal prediction trong log space.
       - Test (15%): Đánh giá kiểm thử độc lập cuối cùng.

    Bảo đảm: Toàn bộ tin đăng của cùng một bất động sản luôn nằm chung trong 1 split duy nhất,
    tương ứng với ngày đăng mới nhất của bất động sản đó.
    """
    if "property_group_id" not in df or "listing_date" not in df:
        raise ValueError("DataFrame phải chứa các cột 'property_group_id' và 'listing_date' để thực hiện split.")

    ordered_groups = (
        df.groupby("property_group_id", as_index=False)["listing_date"]
        .max()
        .sort_values(["listing_date", "property_group_id"])
    )
    number_of_groups = len(ordered_groups)
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

    train_idx = df.index[df["property_group_id"].isin(train_groups)].to_numpy()
    validation_idx = df.index[
        df["property_group_id"].isin(validation_groups)
    ].to_numpy()
    calibration_idx = df.index[
        df["property_group_id"].isin(calibration_groups)
    ].to_numpy()
    test_idx = df.index[df["property_group_id"].isin(test_groups)].to_numpy()

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
