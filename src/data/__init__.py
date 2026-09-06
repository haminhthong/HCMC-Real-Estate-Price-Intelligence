"""Gói xử lý dữ liệu (Data Pipeline) cho HCMC Real Estate Price Intelligence.

Bao gồm các mô-đun:
- loader: Tải dữ liệu thô
- schema: Kiểm tra schema và hợp đồng dữ liệu
- geo: Xử lý địa lý, chuẩn hóa quận/huyện và kiểm tra tọa độ GPS
- identity: Tạo chữ ký và định danh nhóm bất động sản (property_group_id)
- validation: Kiểm tra hợp lệ số học, lọc ngoại lệ, xử lý ngày tháng
- cleaning: Quy trình làm sạch dữ liệu và deduplication cấp độ tin đăng
- manifest: Khai báo DatasetManifest và SplitManifest
- split: Chia tập Grouped Temporal Split chống rò rỉ dữ liệu
"""

from .cleaning import clean_data
from .geo import extract_area, validate_gps_coordinates
from .identity import assign_property_group, make_property_signature
from .loader import load_raw_dataset
from .manifest import DatasetManifest, SplitManifest, create_dataset_manifest, create_split_manifest
from .schema import REQUIRED_RAW_COLUMNS, PropertyRecord, validate_raw_schema
from .split import split_group_indices
from .validation import CRAWL_DATE, filter_numeric_outliers, parse_listing_dates

__all__ = [
    "CRAWL_DATE",
    "REQUIRED_RAW_COLUMNS",
    "DatasetManifest",
    "PropertyRecord",
    "SplitManifest",
    "assign_property_group",
    "clean_data",
    "create_dataset_manifest",
    "create_split_manifest",
    "extract_area",
    "filter_numeric_outliers",
    "load_raw_dataset",
    "make_property_signature",
    "parse_listing_dates",
    "split_group_indices",
    "validate_gps_coordinates",
    "validate_raw_schema",
]
