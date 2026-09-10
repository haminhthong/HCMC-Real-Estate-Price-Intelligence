"""Cấu hình chung cho toàn bộ dự án HCMC Real Estate Price Intelligence.

Tệp này quản lý các đường dẫn hệ thống, phiên bản mô hình, danh mục bất động sản,
tập đặc trưng (features) và cấu hình logging chuẩn hóa.
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Cấu hình ghi nhật ký
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hcmc_price_intelligence")

# ---------------------------------------------------------------------------
# Đường dẫn thư mục, dữ liệu và mô hình
# ---------------------------------------------------------------------------
# Đường dẫn thư mục gốc của dự án
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Đường dẫn dữ liệu đầu vào
DATA_PATH: Path = ROOT_DIR / "data" / "sample" / "data_public_sample.csv"

# Artifact phẳng: một mô hình, một context, một calibration và một bảng tham chiếu.
ARTIFACT_DIR: Path = ROOT_DIR / "artifacts"
MODEL_PATH: Path = ARTIFACT_DIR / "model.joblib"
FEATURE_CONTEXT_PATH: Path = ARTIFACT_DIR / "feature_context.json"
CALIBRATION_PATH: Path = ARTIFACT_DIR / "calibration.json"
COMPARABLES_PATH: Path = ARTIFACT_DIR / "comparables.csv"

# Báo cáo là nguồn sự thật duy nhất cho kết quả đánh giá và data summary.
REPORT_DIR: Path = ROOT_DIR / "reports"
METRICS_PATH: Path = REPORT_DIR / "metrics.json"
DATA_SUMMARY_PATH: Path = REPORT_DIR / "data_summary.json"
ERROR_ANALYSIS_PATH: Path = REPORT_DIR / "error_analysis.json"

# Nhãn mô hình và hạt giống ngẫu nhiên để tái lập kết quả.
MODEL_VERSION: str = "1.2.0"
RANDOM_STATE: int = 42
CANONICAL_SPLIT_PROTOCOL: str = (
    "group_isolated_temporal_split_60_15_10_15_by_latest_property_date"
)

# ---------------------------------------------------------------------------
# Danh mục loại bất động sản và khu vực được hỗ trợ tại TP.HCM
# ---------------------------------------------------------------------------
RESIDENTIAL_TYPES: list[str] = [
    "Nhà riêng",
    "Nhà mặt tiền",
    "Căn hộ chung cư",
    "Biệt thự liền kề",
    "Nhà biệt thự",
]

SUPPORTED_AREAS: list[str] = [
    "Quận 1",
    "Quận 3",
    "Quận 4",
    "Quận 5",
    "Quận 6",
    "Quận 7",
    "Quận 8",
    "Quận 10",
    "Quận 11",
    "Quận 12",
    "Quận Bình Tân",
    "Quận Bình Thạnh",
    "Quận Gò Vấp",
    "Quận Phú Nhuận",
    "Quận Tân Bình",
    "Quận Tân Phú",
    "TP. Thủ Đức",
    "Huyện Bình Chánh",
    "Huyện Cần Giờ",
    "Huyện Củ Chi",
    "Huyện Hóc Môn",
    "Huyện Nhà Bè",
    "Unknown",
]

# ---------------------------------------------------------------------------
# Danh sách đặc trưng phục vụ huấn luyện
# ---------------------------------------------------------------------------
# Các đặc trưng dạng số
NUMERIC_FEATURES: list[str] = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Width",
    "Length",
    "Alley Width",
    "Latitude",
    "Longitude",
    "distance_to_cbd_km",
    "days_from_train_reference",
    "input_completeness_score",
]

# Các đặc trưng phân loại
CATEGORICAL_FEATURES: list[str] = [
    "Property Type",
    "location_area",
    "Direction",
    "Position",
]

# Các cờ nhị phân trích xuất từ tiện ích và nội dung tin
FLAG_FEATURES: list[str] = [
    "has_furniture",
    "car_alley",
    "near_market",
    "near_school",
    "is_urgent_sale",
]

# Các cờ chỉ báo khuyết thiếu dữ liệu có chủ đích (Missingness Indicators)
MISSING_INDICATOR_FEATURES: list[str] = [
    "gps_missing",
    "width_missing",
    "length_missing",
    "bedrooms_missing",
    "bathrooms_missing",
    "floors_missing",
    "road_type_missing",
    "alley_width_missing",
]

# Hợp đồng đặc trưng duy nhất dùng cho cả huấn luyện và serving.
# Missingness là tín hiệu có ý nghĩa trong dữ liệu tin đăng bất động sản,
# vì vậy phải được đưa vào model thay vì chỉ tạo ra rồi bỏ quên.
MODEL_FEATURES: list[str] = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES + MISSING_INDICATOR_FEATURES
)
EXTENDED_MODEL_FEATURES: list[str] = list(MODEL_FEATURES)
