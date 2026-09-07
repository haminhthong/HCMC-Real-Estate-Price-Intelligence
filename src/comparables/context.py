"""Mô-đun quản lý ngữ cảnh chuẩn hóa cho động cơ tra cứu bất động sản tương đồng (ComparableContext)."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class ComparableContext:
    """Ngữ cảnh chuẩn hóa đo lường khoảng cách tương đồng giữa các bất động sản.

    Lưu trữ các giá trị tỷ lệ (scalers) và thống kê phân khúc học được từ tập Train/Dev:
    - median_area, area_scale
    - cbd_scale
    - segment_counts (số lượng mẫu theo từng loại hình x quận)
    - allowed_types, allowed_areas
    """

    median_area: float = 75.0
    area_scale: float = 40.0
    cbd_scale: float = 8.0
    allowed_types: list[str] = field(default_factory=list)
    allowed_areas: list[str] = field(default_factory=list)
    segment_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, reference_df: pd.DataFrame) -> "ComparableContext":
        """Khởi tạo và fit ComparableContext từ dữ liệu tham chiếu (Train hoặc Train+Validation)."""
        if "Area" in reference_df:
            areas = pd.to_numeric(reference_df["Area"], errors="coerce").dropna()
            median_area = float(areas.median()) if len(areas) > 0 else 75.0
            q75 = float(areas.quantile(0.75)) if len(areas) > 0 else 100.0
            q25 = float(areas.quantile(0.25)) if len(areas) > 0 else 50.0
            area_scale = max(q75 - q25, 15.0)
        else:
            median_area = 75.0
            area_scale = 40.0

        if "distance_to_cbd_km" in reference_df:
            cbds = pd.to_numeric(reference_df["distance_to_cbd_km"], errors="coerce").dropna()
            cbd_scale = float(cbds.std()) if len(cbds) > 1 and float(cbds.std()) > 0 else 8.0
        else:
            cbd_scale = 8.0

        allowed_types = sorted(reference_df["Property Type"].dropna().unique().tolist()) if "Property Type" in reference_df else []
        allowed_areas = sorted(reference_df["location_area"].dropna().unique().tolist()) if "location_area" in reference_df else []

        segment_counts: dict[str, int] = {}
        if "Property Type" in reference_df and "location_area" in reference_df:
            grouped = reference_df.groupby(["Property Type", "location_area"]).size()
            for (p_type, loc_area), count in grouped.items():
                segment_counts[f"{p_type} | {loc_area}"] = int(count)

        return cls(
            median_area=round(median_area, 1),
            area_scale=round(area_scale, 1),
            cbd_scale=round(cbd_scale, 1),
            allowed_types=allowed_types,
            allowed_areas=allowed_areas,
            segment_counts=segment_counts,
        )

    def to_dict(self) -> dict[str, Any]:
        """Xuất sang dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComparableContext":
        """Nạp từ dictionary."""
        return cls(
            median_area=float(data.get("median_area", 75.0)),
            area_scale=float(data.get("area_scale", 40.0)),
            cbd_scale=float(data.get("cbd_scale", 8.0)),
            allowed_types=list(data.get("allowed_types", [])),
            allowed_areas=list(data.get("allowed_areas", [])),
            segment_counts=dict(data.get("segment_counts", {})),
        )

    def save(self, filepath: Path | str) -> None:
        """Lưu ngữ cảnh ra tệp JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Path | str) -> "ComparableContext":
        """Nạp ngữ cảnh từ tệp JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
