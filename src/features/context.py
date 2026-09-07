"""Mô-đun quản lý ngữ cảnh đặc trưng (FeatureContext) chống rò rỉ và training-serving skew."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    FLAG_FEATURES,
    NUMERIC_FEATURES,
)


@dataclass
class FeatureContext:
    """Ngữ cảnh đặc trưng đóng băng từ tập huấn luyện (Train/Dev).

    Lưu giữ các tham số môi trường: mốc thời gian tham chiếu, tọa độ trung tâm,
    danh sách đặc trưng chuẩn hóa và phiên bản schema.
    """

    reference_date: str
    cbd_latitude: float = 10.7769
    cbd_longitude: float = 106.7009
    feature_schema_version: int = 2
    numeric_features: list[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    categorical_features: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    flag_features: list[str] = field(default_factory=lambda: list(FLAG_FEATURES))
    missing_indicator_features: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        cbd_latitude: float = 10.7769,
        cbd_longitude: float = 106.7009,
    ) -> "FeatureContext":
        """Khởi tạo FeatureContext từ tập dữ liệu huấn luyện.

        Args:
            df: DataFrame tập Train hoặc Train+Validation.
            cbd_latitude: Vĩ độ CBD Quận 1.
            cbd_longitude: Kinh độ CBD Quận 1.
        """
        if "listing_date" in df and df["listing_date"].notna().any():
            max_date = df["listing_date"].max()
            ref_str = max_date.isoformat() if hasattr(max_date, "isoformat") else str(max_date)
        else:
            ref_str = datetime.now().isoformat()

        return cls(
            reference_date=ref_str,
            cbd_latitude=cbd_latitude,
            cbd_longitude=cbd_longitude,
        )

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi thành dictionary có thể serialize sang JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureContext":
        """Khôi phục FeatureContext từ dictionary đã lưu."""
        return cls(
            reference_date=data["reference_date"],
            cbd_latitude=data.get("cbd_latitude", 10.7769),
            cbd_longitude=data.get("cbd_longitude", 106.7009),
            feature_schema_version=data.get("feature_schema_version", 2),
            numeric_features=data.get("numeric_features", list(NUMERIC_FEATURES)),
            categorical_features=data.get("categorical_features", list(CATEGORICAL_FEATURES)),
            flag_features=data.get("flag_features", list(FLAG_FEATURES)),
            missing_indicator_features=data.get("missing_indicator_features", []),
        )

    @property
    def model_features(self) -> list[str]:
        return (
            self.numeric_features
            + self.categorical_features
            + self.flag_features
            + self.missing_indicator_features
        )
