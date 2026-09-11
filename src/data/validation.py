"""Mô-đun kiểm tra hợp lệ số học, lọc ngoại lệ và chuẩn hóa ngày đăng tin."""

import numpy as np
import pandas as pd

from src.config import (
    MAX_ALLEY_WIDTH_M,
    MAX_AREA_M2,
    MAX_BATHROOMS,
    MAX_BEDROOMS,
    MAX_FLOORS,
    MAX_LENGTH_M,
    MAX_PRICE_MILLION,
    MAX_WIDTH_M,
    MIN_ALLEY_WIDTH_M,
    MIN_AREA_M2,
    MIN_BATHROOMS,
    MIN_BEDROOMS,
    MIN_FLOORS,
    MIN_LENGTH_M,
    MIN_PRICE_MILLION,
    MIN_UNIT_PRICE_MILLION_M2,
    MIN_WIDTH_M,
)

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

    Điều kiện hợp lệ tuân thủ các ngưỡng chuẩn trong src.config:
    - Giá: [MIN_PRICE_MILLION, MAX_PRICE_MILLION) triệu VND
    - Diện tích: [MIN_AREA_M2, MAX_AREA_M2) m²
    - Đơn giá: >= MIN_UNIT_PRICE_MILLION_M2 triệu VND/m²
    - Số phòng ngủ: NaN hoặc [MIN_BEDROOMS, MAX_BEDROOMS]
    - Số phòng vệ sinh: NaN hoặc [MIN_BATHROOMS, MAX_BATHROOMS]
    - Số tầng: NaN hoặc [MIN_FLOORS, MAX_FLOORS]
    - Mặt tiền: NaN hoặc [MIN_WIDTH_M, MAX_WIDTH_M]
    - Chiều dài: NaN hoặc [MIN_LENGTH_M, MAX_LENGTH_M]
    - Hẻm: NaN hoặc [MIN_ALLEY_WIDTH_M, MAX_ALLEY_WIDTH_M]
    """
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    unit_price = out["Price"] / out["Area"].replace(0, np.nan)
    valid = (
        out["Price"].between(MIN_PRICE_MILLION, MAX_PRICE_MILLION, inclusive="left")
        & out["Area"].between(MIN_AREA_M2, MAX_AREA_M2, inclusive="left")
        & unit_price.ge(MIN_UNIT_PRICE_MILLION_M2)
        & (out["Bedrooms"].isna() | out["Bedrooms"].between(MIN_BEDROOMS, MAX_BEDROOMS))
        & (
            out["Bathrooms"].isna()
            | out["Bathrooms"].between(MIN_BATHROOMS, MAX_BATHROOMS)
        )
        & (out["Floors"].isna() | out["Floors"].between(MIN_FLOORS, MAX_FLOORS))
        & (out["Width"].isna() | out["Width"].between(MIN_WIDTH_M, MAX_WIDTH_M))
        & (out["Length"].isna() | out["Length"].between(MIN_LENGTH_M, MAX_LENGTH_M))
        & (
            out["Alley Width"].isna()
            | out["Alley Width"].between(MIN_ALLEY_WIDTH_M, MAX_ALLEY_WIDTH_M)
        )
    )
    return out.loc[valid].copy()


def parse_listing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Phân tích ngày đăng tin và đánh dấu cờ dữ liệu ngày bị khuyết.

    Không tự tiện gán ngày thiếu thành ngày crawl cố định. Nếu có ngày cập nhật
    thì dùng ngày đó; nếu không có nhưng có thời điểm scrape thì dùng scrape time
    như proxy quan sát và đánh dấu rõ nguồn ngày.
    """
    out = df.copy()
    if "Last Updated Date" in out:
        listing_dates = pd.to_datetime(
            out["Last Updated Date"],
            format="%d/%m/%Y %H:%M",
            errors="coerce",
            utc=True,
        )
    elif "listing_date" in out:
        listing_dates = pd.to_datetime(out["listing_date"], errors="coerce", utc=True)
    else:
        listing_dates = pd.Series(pd.NaT, index=out.index)

    if "Scraped At" in out:
        observed_at = pd.to_datetime(out["Scraped At"], errors="coerce", utc=True)
    elif "observed_at" in out:
        observed_at = pd.to_datetime(out["observed_at"], errors="coerce", utc=True)
    else:
        observed_at = pd.Series(pd.NaT, index=out.index)

    proxy_mask = listing_dates.isna() & observed_at.notna()
    unknown_mask = listing_dates.isna() & observed_at.isna()
    out["observed_at"] = observed_at
    out["listing_date"] = listing_dates.where(~proxy_mask, observed_at)
    out["listing_date_missing"] = unknown_mask.astype(int)
    out["date_source"] = np.select(
        [listing_dates.notna(), proxy_mask],
        ["listing_date", "scrape_time_proxy"],
        default="unknown",
    )
    out["temporal_status"] = np.select(
        [listing_dates.notna(), proxy_mask],
        ["known", "proxy"],
        default="unknown",
    )
    return out
