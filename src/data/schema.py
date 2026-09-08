"""Mô-đun định nghĩa hợp đồng dữ liệu (Schema Contract) và đối tượng PropertyRecord."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

REQUIRED_RAW_COLUMNS: set[str] = {"Price", "Area", "Property Type", "Location"}


@dataclass
class PropertyRecord:
    """Bản ghi chuẩn hóa trung gian biểu diễn một bất động sản hoặc tin đăng.

    Cầu nối giữa offline raw dataset và online serving API request.
    """

    property_type: str
    location_area: str
    area: float
    bedrooms: int
    bathrooms: int | None = None
    floors: int | None = None
    width: float | None = None
    length: float | None = None
    alley_width: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    direction: str = "Không rõ"
    position: str = "Không rõ"
    has_furniture: int = 0
    car_alley: int = 0
    near_market: int = 0
    near_school: int = 0
    is_urgent_sale: int = 0
    listing_date: pd.Timestamp | None = None
    listing_id: Any = None
    property_group_id: str | None = None
    price_million: float | None = None


def validate_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra sự tồn tại của các cột bắt buộc trong dữ liệu thô.

    Args:
        df: DataFrame đầu vào.

    Returns:
        DataFrame gốc nếu hợp lệ.

    Raises:
        ValueError: Nếu thiếu một hoặc nhiều cột bắt buộc.
    """
    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc trong dữ liệu đầu vào: {sorted(missing)}")
    return df
