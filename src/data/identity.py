"""Mô-đun định danh bất động sản vật lý (Property Identity) và nhóm tin đăng.

Tạo khóa nhóm duy nhất (property_group_id) dựa trên chữ ký đặc trưng bất động sản
nhằm phục vụ cơ chế chia tập Grouped Temporal Split chống rò rỉ dữ liệu (Data Leakage).
"""

from typing import Any
import pandas as pd

from .geo import _normalize_text


def make_property_signature(df: pd.DataFrame) -> pd.Series:
    """Tạo chữ ký bất động sản vật lý độc lập với tin đăng, giá, ngày đăng và môi giới.

    Các yếu tố cấu thành chữ ký:
    - location (chuẩn hóa Unicode NFC)
    - property_type (chuẩn hóa)
    - area_rounded (làm tròn 0 chữ số thập phân)
    - width_rounded (làm tròn 1 chữ số thập phân)
    - length_rounded (làm tròn 1 chữ số thập phân)
    - bedrooms (số phòng ngủ)
    - bathrooms (số phòng vệ sinh)
    - latitude_rounded (làm tròn 4 chữ số thập phân ~ 11m)
    - longitude_rounded (làm tròn 4 chữ số thập phân ~ 11m)

    Args:
        df: DataFrame chứa thông tin bất động sản.

    Returns:
        pd.Series chứa chuỗi hash 64-bit biểu diễn mã định danh bất động sản vật lý.
    """
    location = df["Location"].map(_normalize_text) if "Location" in df else pd.Series("", index=df.index)
    property_type = (
        df["Property Type"].map(_normalize_text)
        if "Property Type" in df
        else pd.Series("", index=df.index)
    )

    area = pd.to_numeric(df.get("Area"), errors="coerce").round(0)
    width = pd.to_numeric(df.get("Width"), errors="coerce").round(1)
    length = pd.to_numeric(df.get("Length"), errors="coerce").round(1)

    latitude = pd.to_numeric(df.get("Latitude"), errors="coerce").round(4)
    longitude = pd.to_numeric(df.get("Longitude"), errors="coerce").round(4)

    bedrooms = pd.to_numeric(df.get("Bedrooms"), errors="coerce")
    bathrooms = pd.to_numeric(df.get("Bathrooms"), errors="coerce")

    signature = pd.DataFrame(
        {
            "location": location,
            "property_type": property_type,
            "area": area,
            "width": width,
            "length": length,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "latitude": latitude,
            "longitude": longitude,
        }
    )

    return pd.util.hash_pandas_object(
        signature.fillna("missing"),
        index=False,
    ).astype(str)


def assign_property_group(df: pd.DataFrame) -> pd.DataFrame:
    """Gán cột `property_group_id` cho DataFrame dựa trên chữ ký bất động sản."""
    out = df.copy()
    out["property_group_id"] = make_property_signature(out)
    return out
