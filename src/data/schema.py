"""Mô-đun định nghĩa hợp đồng cột đầu vào của dữ liệu thô."""

import pandas as pd

REQUIRED_RAW_COLUMNS: set[str] = {"Price", "Area", "Property Type", "Location"}


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
        raise ValueError(
            f"Thiếu các cột bắt buộc trong dữ liệu đầu vào: {sorted(missing)}"
        )
    return df
