"""Mô-đun tương thích ngược cho trích xuất đặc trưng (Feature Engineering).

Ủy nhiệm toàn bộ chức năng sang gói chuyên biệt `src.features`.
"""

from src.features import (
    CBD_LATITUDE,
    CBD_LONGITUDE,
    KEYWORDS,
    NEGATION_PATTERN,
    FeatureContext,
    add_quality_features,
    add_text_flags,
    build_features,
    calculate_days_from_reference,
    calculate_distance_to_cbd,
    calculate_input_completeness,
    make_features,
)

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
