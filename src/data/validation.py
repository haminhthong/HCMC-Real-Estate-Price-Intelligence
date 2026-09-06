"""Mô-đun kiểm tra hợp lệ số học, lọc ngoại lệ và chuẩn hóa ngày đăng tin."""

import numpy as np
import pandas as pd

CRAWL_DATE: pd.Timestamp = pd.Timestamp("2025-09-30 23:59:59")

NUMERIC_COLUMNS: list[str] = [
    "Price",
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Width",
    "Length",
    "Alley Width",
    "Latitude",
    "Longitude",
]


def filter_numeric_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Ép kiểu số và loại bỏ các bản ghi có giá trị ngoại lệ phi thực tế trên thực địa.

    Điều kiện hợp lệ:
    - Giá: 100 triệu - 50 tỷ VND
    - Diện tích: 5m² - 500m²
    - Đơn giá: >= 10 triệu VND/m²
    - Số phòng ngủ: NaN hoặc 1 - 10 phòng
    - Số phòng vệ sinh: NaN hoặc 0 - 20 phòng
    - Số tầng: NaN hoặc 0 - 100 tầng
    - Mặt tiền (Width): NaN hoặc 0.1m - 100m
    - Chiều dài (Length): NaN hoặc 0.1m - 200m
    - Hẻm (Alley Width): NaN hoặc 0m - 30m
    """
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    unit_price = out["Price"] / out["Area"].replace(0, np.nan)
    valid = (
        out["Price"].between(100, 50_000, inclusive="left")
        & out["Area"].between(5, 500, inclusive="left")
        & unit_price.ge(10)
        & (out["Bedrooms"].isna() | out["Bedrooms"].between(1, 10))
        & (out["Bathrooms"].isna() | out["Bathrooms"].between(0, 20))
        & (out["Floors"].isna() | out["Floors"].between(0, 100))
        & (out["Width"].isna() | out["Width"].between(0.1, 100))
        & (out["Length"].isna() | out["Length"].between(0.1, 200))
        & (out["Alley Width"].isna() | out["Alley Width"].between(0, 30))
    )
    return out.loc[valid].copy()


def parse_listing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Phân tích ngày đăng tin và đánh dấu cờ dữ liệu ngày bị khuyết.

    Không tự tiện gán ngày thiếu thành `earliest_date` toàn cục để tránh làm sai lệch
    giả định dòng thời gian của grouped temporal split.
    Đánh dấu `listing_date_missing = 1` nếu khuyết ngày.
    """
    out = df.copy()
    if "Last Updated Date" in out:
        parsed_dates = pd.to_datetime(
            out["Last Updated Date"],
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )
    elif "listing_date" in out:
        parsed_dates = pd.to_datetime(out["listing_date"], errors="coerce")
    else:
        parsed_dates = pd.Series(pd.NaT, index=out.index)

    missing_mask = parsed_dates.isna()
    out["listing_date_missing"] = missing_mask.astype(int)

    # Nếu khuyết ngày, gán mốc crawl mặc định có kiểm soát thay vì giả định là tin cổ nhất
    out["listing_date"] = parsed_dates.fillna(CRAWL_DATE)
    return out
