"""Gói trích xuất đặc trưng (Feature Engineering) cho HCMC Real Estate Price Intelligence.

Bao gồm:
- context: FeatureContext đóng băng môi trường đặc trưng
- geospatial: Khoảng cách Haversine tới trung tâm TP.HCM
- temporal: Độ lệch thời gian so với mốc reference
- text: Cờ nhị phân tiện ích có xử lý phủ định
- structural: Độ hoàn thiện dữ liệu đo lường
- builder: Hàm build_features dùng chung cho huấn luyện và serving
"""

from .builder import build_features
from .context import FeatureContext
from .geospatial import calculate_distance_to_cbd
from .structural import calculate_input_completeness
from .temporal import calculate_days_from_reference
from .text import KEYWORDS, NEGATION_PATTERN, add_text_flags

__all__ = [
    "KEYWORDS",
    "NEGATION_PATTERN",
    "FeatureContext",
    "add_text_flags",
    "build_features",
    "calculate_days_from_reference",
    "calculate_distance_to_cbd",
    "calculate_input_completeness",
]
