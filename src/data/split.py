"""Group-isolated temporal split duy nhất của project."""

import numpy as np
import pandas as pd


def _positions(mask: pd.Series) -> np.ndarray:
    """Đổi boolean mask thành vị trí nguyên để dùng an toàn với ``DataFrame.iloc``."""
    return np.flatnonzero(mask.to_numpy(dtype=bool))


def split_group_indices(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.15,
    calibration_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chia theo ngày listing muộn nhất của từng property group.

    Mọi listing của cùng một ``property_group_id`` ở cùng một tập. Listing
    không có ngày bị loại khỏi benchmark thay vì bị gán một ngày giả.
    """
    required = {"property_group_id", "listing_date"}
    if not required.issubset(df.columns):
        raise ValueError(
            "DataFrame phải chứa 'property_group_id' và 'listing_date' để split."
        )
    if (
        train_ratio <= 0
        or validation_ratio <= 0
        or calibration_ratio <= 0
        or train_ratio + validation_ratio + calibration_ratio >= 1
    ):
        raise ValueError("Tỷ lệ split không hợp lệ.")

    dated = df.loc[df["listing_date"].notna()].copy()
    if dated.empty:
        raise ValueError("Không thể split: không có listing_date hợp lệ.")

    latest_by_group = (
        dated.groupby("property_group_id", sort=False)["listing_date"]
        .max()
        .sort_values()
        .reset_index()
    )
    number_of_groups = len(latest_by_group)
    if number_of_groups < 7:
        raise ValueError("Cần ít nhất 7 property groups để tạo đủ bốn tập.")

    train_end = max(1, int(number_of_groups * train_ratio))
    validation_end = max(
        train_end + 1, int(number_of_groups * (train_ratio + validation_ratio))
    )
    calibration_end = max(
        validation_end + 1,
        int(number_of_groups * (train_ratio + validation_ratio + calibration_ratio)),
    )
    calibration_end = min(calibration_end, number_of_groups - 1)
    validation_end = min(validation_end, calibration_end - 1)
    train_end = min(train_end, validation_end - 1)

    train_groups = set(latest_by_group.iloc[:train_end]["property_group_id"])
    validation_groups = set(
        latest_by_group.iloc[train_end:validation_end]["property_group_id"]
    )
    calibration_groups = set(
        latest_by_group.iloc[validation_end:calibration_end]["property_group_id"]
    )
    test_groups = set(latest_by_group.iloc[calibration_end:]["property_group_id"])

    train_idx = _positions(
        df["property_group_id"].isin(train_groups) & df["listing_date"].notna()
    )
    validation_idx = _positions(
        df["property_group_id"].isin(validation_groups) & df["listing_date"].notna()
    )
    calibration_idx = _positions(
        df["property_group_id"].isin(calibration_groups) & df["listing_date"].notna()
    )
    test_idx = _positions(
        df["property_group_id"].isin(test_groups) & df["listing_date"].notna()
    )

    groups = [
        {df.iloc[index]["property_group_id"] for index in split}
        for split in (train_idx, validation_idx, calibration_idx, test_idx)
    ]
    for left_index, left_groups in enumerate(groups):
        for right_groups in groups[left_index + 1 :]:
            if left_groups.intersection(right_groups):
                raise RuntimeError("Phát hiện property group bị trùng giữa các tập.")

    return train_idx, validation_idx, calibration_idx, test_idx


__all__ = ["split_group_indices"]
