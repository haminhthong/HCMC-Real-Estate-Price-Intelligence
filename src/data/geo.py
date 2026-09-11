"""Mô-đun xử lý địa lý, trích xuất quận/huyện và kiểm tra hợp lệ tọa độ GPS TP.HCM."""

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from src.config import MAX_LATITUDE, MAX_LONGITUDE, MIN_LATITUDE, MIN_LONGITUDE


def _normalize_text(value: Any) -> str:
    """Chuẩn hóa chuỗi văn bản Unicode NFC, chuyển chữ thường và gộp khoảng trắng."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value)).lower()
    return re.sub(r"\s+", " ", text).strip()


def extract_area(location: Any) -> str:
    """Rút gọn và gán nhãn địa chỉ về đúng đơn vị hành chính TP.HCM.

    Nhận diện các quận từ Quận 1 đến Quận 12, các quận đặt tên (Bình Thạnh, Gò Vấp...),
    TP. Thủ Đức (gộp Quận 2, Quận 9, Thủ Đức), và các huyện ngoại thành.

    Args:
        location: Chuỗi thông tin địa chỉ thô từ tin đăng.

    Returns:
        Tên quận/huyện tiêu chuẩn hoặc "Unknown" nếu không trích xuất được.
    """
    text = _normalize_text(location)

    # Quy hoạch TP. Thủ Đức (bao gồm Quận 2, Quận 9, Thủ Đức cũ)
    if any(name in text for name in ("quận 2", "quận 9", "thủ đức")):
        return "TP. Thủ Đức"

    # Trích xuất các quận bằng số (Quận 1 - 12)
    numbered = re.search(r"quận\s+(1[0-2]|[1-8])(?:\D|$)", text)
    if numbered:
        return f"Quận {numbered.group(1)}"

    # Ánh xạ các quận/huyện có tên riêng
    named_areas: dict[str, str] = {
        "bình tân": "Quận Bình Tân",
        "bình thạnh": "Quận Bình Thạnh",
        "gò vấp": "Quận Gò Vấp",
        "phú nhuận": "Quận Phú Nhuận",
        "tân bình": "Quận Tân Bình",
        "tân phú": "Quận Tân Phú",
        "bình chánh": "Huyện Bình Chánh",
        "cần giờ": "Huyện Cần Giờ",
        "củ chi": "Huyện Củ Chi",
        "hóc môn": "Huyện Hóc Môn",
        "nhà bè": "Huyện Nhà Bè",
    }
    return next(
        (area for keyword, area in named_areas.items() if keyword in text),
        "Unknown",
    )


def validate_gps_coordinates(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
) -> pd.DataFrame:
    """Kiểm tra tính hợp lệ của tọa độ GPS trong phạm vi khu vực TP.HCM.

    Phạm vi địa lý TP.HCM: Vĩ độ [MIN_LATITUDE, MAX_LATITUDE], Kinh độ [MIN_LONGITUDE, MAX_LONGITUDE].
    Các tọa độ ngoài khoảng sẽ được chuyển thành NaN.
    """
    out = df.copy()
    if lat_col in out and lon_col in out:
        valid_coords = out[lat_col].between(MIN_LATITUDE, MAX_LATITUDE) & out[
            lon_col
        ].between(MIN_LONGITUDE, MAX_LONGITUDE)
        out.loc[~valid_coords, [lat_col, lon_col]] = np.nan
    return out
