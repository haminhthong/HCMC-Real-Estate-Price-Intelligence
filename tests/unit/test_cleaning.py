"""Kiểm thử đơn vị cho mô-đun Cleaning và Khử trùng lặp."""

import pandas as pd

from src.data.cleaning import clean_data


def test_clean_data_drops_exact_duplicates_only():
    """Kiểm tra: Tin trùng lặp hoàn toàn bị loại, nhưng tin khác ngày/giá được giữ lại."""
    base_listing = {
        "Price": 5000.0,
        "Area": 60.0,
        "Property Type": "Nhà riêng",
        "Location": "Quận 1, TP.HCM",
        "Bedrooms": 2,
        "Bathrooms": 2,
        "Width": 4.0,
        "Length": 15.0,
        "Latitude": 10.7769,
        "Longitude": 106.7009,
        "Last Updated Date": "01/01/2025 10:00",
    }

    # Tạo 3 dòng:
    # row 0: gốc
    # row 1: trùng hoàn toàn với row 0 -> PHẢI BỊ DROP
    # row 2: cùng bất động sản nhưng đăng lại ngày khác với giá khác -> PHẢI ĐƯỢC GIỮ
    row0 = dict(base_listing)
    row1 = dict(base_listing)
    row2 = dict(base_listing)
    row2["Price"] = 5500.0
    row2["Last Updated Date"] = "15/06/2025 14:00"

    df = pd.DataFrame([row0, row1, row2])
    cleaned = clean_data(df)

    assert len(cleaned) == 2, (
        "Chỉ 1 bản ghi trùng hoàn toàn bị loại, phải còn lại đúng 2 dòng!"
    )
    assert cleaned["property_group_id"].nunique() == 1, (
        "Cả 2 dòng phải chung 1 property_group_id!"
    )
