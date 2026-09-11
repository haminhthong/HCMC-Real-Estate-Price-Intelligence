"""Kiểm thử đơn vị cho thuật toán Grouped Temporal Split."""

import pandas as pd
import pytest

from src.data.split import split_group_indices


def test_split_group_isolation_and_temporal_order():
    """Kiểm tra: Bốn tập không trùng property_group_id và tuân thủ thứ tự thời gian."""
    frame = pd.DataFrame(
        {
            "property_group_id": [f"group-{index}" for index in range(100)],
            "listing_date": pd.date_range("2025-01-01", periods=100),
        }
    )
    train, validation, calibration, test = split_group_indices(frame)
    train_groups = set(frame.iloc[train].property_group_id)
    val_groups = set(frame.iloc[validation].property_group_id)
    calib_groups = set(frame.iloc[calibration].property_group_id)
    test_groups = set(frame.iloc[test].property_group_id)

    assert train_groups.isdisjoint(val_groups), "Trùng group giữa Train và Validation"
    assert train_groups.isdisjoint(
        calib_groups
    ), "Trùng group giữa Train và Calibration"
    assert train_groups.isdisjoint(test_groups), "Trùng group giữa Train và Test"
    assert val_groups.isdisjoint(
        calib_groups
    ), "Trùng group giữa Validation và Calibration"
    assert val_groups.isdisjoint(test_groups), "Trùng group giữa Validation và Test"
    assert calib_groups.isdisjoint(test_groups), "Trùng group giữa Calibration và Test"

    assert (
        frame.iloc[train].listing_date.max()
        <= frame.iloc[validation].listing_date.min()
    )
    assert (
        frame.iloc[validation].listing_date.max()
        <= frame.iloc[calibration].listing_date.min()
    )
    assert (
        frame.iloc[calibration].listing_date.max()
        <= frame.iloc[test].listing_date.min()
    )


def test_grouped_temporal_split_prioritizes_group_isolation():
    """Bất động sản có nhiều tin đăng rải rác đi vào tập ứng với ngày muộn nhất."""
    dates = pd.date_range("2025-01-01", periods=100)
    records = [
        {"property_group_id": f"group-{i}", "listing_date": dates[i]}
        for i in range(100)
    ]
    # Tin cũ và tin mới của cùng một nhóm
    records.append(
        {
            "property_group_id": "group-multi-listing",
            "listing_date": pd.Timestamp("2025-01-02"),
        }
    )
    records.append(
        {
            "property_group_id": "group-multi-listing",
            "listing_date": pd.Timestamp("2025-10-15"),
        }
    )
    frame = pd.DataFrame(records)
    train, _val, _calib, test = split_group_indices(frame)

    multi_indices = frame.index[
        frame["property_group_id"] == "group-multi-listing"
    ].to_numpy()
    assert set(multi_indices).issubset(
        set(test)
    ), "Mọi tin đăng phải cùng nằm trong tập mới nhất"
    assert set(multi_indices).isdisjoint(
        set(train)
    ), "Không rò rỉ tin cũ sang tập Train"


def test_split_returns_iloc_positions_for_non_default_index():
    """Đảm bảo trả về vị trí iloc nguyên kể cả khi DataFrame có index tùy ý."""
    frame = pd.DataFrame(
        {
            "property_group_id": [f"group-{index}" for index in range(14)],
            "listing_date": pd.date_range("2025-01-01", periods=14),
        },
        index=[index * 10 for index in range(14)],
    )

    train, validation, calibration, test = split_group_indices(frame)
    all_indices = (*train, *validation, *calibration, *test)
    assert all(index < len(frame) for index in all_indices)
    selected = pd.concat(
        [
            frame.iloc[train],
            frame.iloc[validation],
            frame.iloc[calibration],
            frame.iloc[test],
        ]
    )
    assert set(selected.index) == set(frame.index)


def test_split_invalid_ratios_or_missing_columns():
    """Kiểm tra báo lỗi khi thiếu cột hoặc tỷ lệ không hợp lệ."""
    df = pd.DataFrame({"listing_date": [pd.Timestamp("2025-01-01")]})
    with pytest.raises(ValueError, match="DataFrame phải chứa"):
        split_group_indices(df)

    valid_df = pd.DataFrame(
        {
            "property_group_id": [f"g{i}" for i in range(10)],
            "listing_date": pd.date_range("2025-01-01", periods=10),
        }
    )
    with pytest.raises(ValueError, match="Tỷ lệ split không hợp lệ"):
        split_group_indices(
            valid_df, train_ratio=0.8, validation_ratio=0.15, calibration_ratio=0.1
        )
