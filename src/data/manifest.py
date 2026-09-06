"""Mô-đun quản lý Manifest (Provenance) cho Dataset và Split."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import pandas as pd


@dataclass
class DatasetManifest:
    """Provenance và metadata theo vết của bộ dữ liệu."""

    source_file: str
    snapshot_id: str
    rows_raw: int
    rows_valid: int
    rows_clean: int
    property_groups: int
    date_min: str
    date_max: str
    schema_version: str = "2.0.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SplitManifest:
    """Provenance và metadata theo vết của quá trình phân chia dữ liệu."""

    protocol: str
    train_groups_count: int
    validation_groups_count: int
    calibration_groups_count: int
    test_groups_count: int
    train_rows: int
    validation_rows: int
    calibration_rows: int
    test_rows: int
    train_date_range: tuple[str, str]
    validation_date_range: tuple[str, str]
    calibration_date_range: tuple[str, str]
    test_date_range: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_dataset_manifest(
    df_clean: pd.DataFrame,
    source_path: Path | str,
    snapshot_id: str = "latest",
) -> DatasetManifest:
    """Tạo DatasetManifest từ DataFrame đã làm sạch."""
    audit = getattr(df_clean, "attrs", {}).get("data_audit", {})
    date_min = (
        df_clean["listing_date"].min().isoformat()
        if "listing_date" in df_clean and df_clean["listing_date"].notna().any()
        else "unknown"
    )
    date_max = (
        df_clean["listing_date"].max().isoformat()
        if "listing_date" in df_clean and df_clean["listing_date"].notna().any()
        else "unknown"
    )

    return DatasetManifest(
        source_file=Path(source_path).name,
        snapshot_id=snapshot_id,
        rows_raw=audit.get("rows_raw", len(df_clean)),
        rows_valid=audit.get("rows_valid", len(df_clean)),
        rows_clean=len(df_clean),
        property_groups=df_clean["property_group_id"].nunique() if "property_group_id" in df_clean else 0,
        date_min=date_min,
        date_max=date_max,
    )


def create_split_manifest(
    df: pd.DataFrame,
    train_idx: Any,
    validation_idx: Any,
    calibration_idx: Any,
    test_idx: Any,
) -> SplitManifest:
    """Tạo SplitManifest từ các chỉ mục tập chia."""
    train_df = df.iloc[train_idx]
    val_df = df.iloc[validation_idx]
    calib_df = df.iloc[calibration_idx]
    test_df = df.iloc[test_idx]

    def get_range(d: pd.DataFrame) -> tuple[str, str]:
        if "listing_date" in d and d["listing_date"].notna().any():
            return (d["listing_date"].min().isoformat(), d["listing_date"].max().isoformat())
        return ("unknown", "unknown")

    return SplitManifest(
        protocol="grouped_temporal_split_60_15_10_15_by_latest_group_listing_date",
        train_groups_count=int(train_df["property_group_id"].nunique()),
        validation_groups_count=int(val_df["property_group_id"].nunique()),
        calibration_groups_count=int(calib_df["property_group_id"].nunique()),
        test_groups_count=int(test_df["property_group_id"].nunique()),
        train_rows=len(train_idx),
        validation_rows=len(validation_idx),
        calibration_rows=len(calibration_idx),
        test_rows=len(test_idx),
        train_date_range=get_range(train_df),
        validation_date_range=get_range(val_df),
        calibration_date_range=get_range(calib_df),
        test_date_range=get_range(test_df),
    )
