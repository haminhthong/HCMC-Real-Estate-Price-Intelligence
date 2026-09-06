"""Mô-đun tương thích ngược cho tiền xử lý và làm sạch dữ liệu.

Tệp này chuyển tiếp các lời gọi hàm sang gói kiến trúc mới `src.data` để đảm bảo
100% tương thích ngược với mã nguồn hiện tại và các bài test.
"""

from typing import Any
import pandas as pd

from src.data.cleaning import clean_data as _clean_data
from src.data.geo import extract_area, _normalize_text
from src.data.identity import assign_property_group, make_property_signature
from src.data.validation import CRAWL_DATE

# Khả năng tương thích ngược các hàm private cũ
def _make_property_group_id(df: pd.DataFrame) -> pd.Series:
    """Tạo khóa nhóm duy nhất cho từng bất động sản dựa trên chữ ký đặc trưng."""
    return make_property_signature(df)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Hàm clean_data ủy nhiệm sang src.data.cleaning.clean_data."""
    return _clean_data(raw)


__all__ = [
    "CRAWL_DATE",
    "_make_property_group_id",
    "_normalize_text",
    "assign_property_group",
    "clean_data",
    "extract_area",
    "make_property_signature",
]
