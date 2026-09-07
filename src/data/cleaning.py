"""Mô-đun điều phối quy trình làm sạch dữ liệu, lọc phạm vi thị trường và khử trùng cấp độ tin đăng.

Chính sách lọc dữ liệu:
Áp dụng các quy tắc hợp lệ xác định trước có nhận biết biến mục tiêu (Predefined target-aware
validity rules used to define the supported market scope; thresholds are fixed before model
development and are not tuned on Validation/Test performance). Ví dụ: 100 triệu <= asking price <= 50 tỷ VND
và diện tích 5 - 500 m² xác định phạm vi phân khúc thị trường được hỗ trợ bởi nền tảng, không phải
là đặc trưng mô hình.
"""

import pandas as pd

from src.config import RESIDENTIAL_TYPES, logger
from .geo import extract_area, validate_gps_coordinates
from .identity import resolve_property_identities
from .schema import validate_raw_schema
from .validation import filter_numeric_outliers, parse_listing_dates


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Tiền xử lý, chuẩn hóa, lọc phạm vi thị trường và deduplicate ở cấp độ tin đăng.

    QUY TRÌNH CHUẨN HÓA DATA LIFECYCLE:
    1. Kiểm tra Schema (các cột bắt buộc: Price, Area, Property Type, Location).
    2. Lọc loại hình nhà ở dân dụng (`RESIDENTIAL_TYPES`).
    3. Chuẩn hóa khu vực hành chính TP.HCM (`location_area`).
    4. Ép kiểu số và áp dụng quy tắc phạm vi thị trường (Predefined target-aware validity rules).
    5. Kiểm tra tọa độ GPS trong ranh giới TP.HCM.
    6. Chuẩn hóa ngày đăng `listing_date` và cờ khuyết ngày `listing_date_missing`.
    7. Định danh bất động sản đa tầng (Multi-level Property Identity Resolution) gán `property_group_id`.
    8. KHỬ TRÙNG CẤP ĐỘ TIN ĐĂNG (Listing-level Dedup):
       Chỉ loại bỏ tin đăng trùng hoàn toàn (`property_group_id`, `listing_date`, `Price`).
       GIỮ LẠI các tin đăng lặp lại theo dòng thời gian của cùng một căn nhà (ví dụ đăng lại
       đổi giá theo chu kỳ thị trường) để phục vụ Group-isolated Temporal Split an toàn và chuẩn xác.

    Args:
        raw: DataFrame dữ liệu thô.

    Returns:
        DataFrame dữ liệu sạch với đầy đủ audit statistics trong `attrs["data_audit"]`.
    """
    logger.info("Bắt đầu quy trình làm sạch dữ liệu thô...")
    df = raw.copy()
    rows_raw = len(df)

    # 1. Kiểm tra Schema bắt buộc
    validate_raw_schema(df)

    # 2. Lọc loại hình bất động sản dân dụng
    df = df[df["Property Type"].isin(RESIDENTIAL_TYPES)].copy()
    rows_after_type_filter = len(df)

    # 3. Trích xuất khu vực hành chính TP.HCM
    df["location_area"] = df["Location"].map(extract_area)

    # 4. Ép kiểu và áp dụng quy tắc phạm vi thị trường
    df = filter_numeric_outliers(df)
    rows_valid = len(df)

    # 5. Kiểm tra tính hợp lệ của tọa độ GPS
    df = validate_gps_coordinates(df)

    # 6. Chuẩn hóa ngày đăng tin
    df = parse_listing_dates(df)

    # 7. Định danh bất động sản đa tầng bằng Union-Find
    df, identity_audit = resolve_property_identities(df)

    # 8. Khử trùng lặp ở cấp độ tin đăng (Listing-level deduplication)
    rows_before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["property_group_id", "listing_date", "Price"]
    ).reset_index(drop=True)
    rows_clean = len(df)
    unique_properties = df["property_group_id"].nunique()

    exact_listing_duplicates_removed = rows_before_dedup - rows_clean
    unsupported_type_removed = rows_raw - rows_after_type_filter
    numeric_outliers_removed = rows_after_type_filter - rows_valid

    audit_stats = {
        "rows_raw": rows_raw,
        "rows_valid": rows_valid,
        "rows_clean": rows_clean,
        "unique_property_groups": unique_properties,
        "multi_listing_groups_count": int(df[df.duplicated("property_group_id", keep=False)]["property_group_id"].nunique()),
        "largest_group_size": identity_audit.get("largest_group_size", 1),
        "identity_level_counts": identity_audit.get("level_counts", {}),
        "rows_removed_by_reason": {
            "unsupported_property_type": unsupported_type_removed,
            "numeric_or_price_outliers": numeric_outliers_removed,
            "exact_duplicate_listings": exact_listing_duplicates_removed,
        },
        "duplicate_listing_percent": float(round(exact_listing_duplicates_removed / max(rows_raw, 1) * 100, 2)),
    }
    df.attrs["data_audit"] = audit_stats

    logger.info(
        "Hoàn tất làm sạch dữ liệu: Raw listings=%d -> Valid listings=%d -> Clean listings=%d -> Unique properties=%d.",
        rows_raw,
        rows_valid,
        rows_clean,
        unique_properties,
    )
    return df
