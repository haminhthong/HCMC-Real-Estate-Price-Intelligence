"""Gói trích xuất đặc trưng (Feature Engineering) cho HCMC Real Estate Price Intelligence.

Bao gồm:
- context: FeatureContext đóng băng môi trường đặc trưng
- geospatial: Khoảng cách Haversine tới trung tâm TP.HCM
- temporal: Độ lệch thời gian so với mốc reference
- text: Cờ nhị phân tiện ích có xử lý phủ định
- structural: Độ hoàn thiện dữ liệu đo lường
- builder: Hàm build_features chuẩn hóa và make_features tương thích ngược
"""

from .builder import add_quality_features, build_features, make_features
from .context import FeatureContext
from .geospatial import calculate_distance_to_cbd
from .structural import calculate_input_completeness
from .temporal import calculate_days_from_reference
from .text import KEYWORDS, NEGATION_PATTERN, add_text_flags

# Tọa độ mặc định CBD để tương thích ngược
CBD_LATITUDE: float = 10.7769
CBD_LONGITUDE: float = 106.7009

__all__ = [
    "CBD_LATITUDE",
    "CBD_LONGITUDE",
    "KEYWORDS",
    "NEGATION_PATTERN",
    "FeatureContext",
    "add_quality_features",
    "add_text_flags",
    "build_features",
    "calculate_days_from_reference",
    "calculate_distance_to_cbd",
    "calculate_input_completeness",
    "make_features",
]
