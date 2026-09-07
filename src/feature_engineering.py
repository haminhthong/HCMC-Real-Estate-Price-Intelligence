"""[DEPRECATED LEGACY WRAPPER] Mô-đun tương thích ngược cho trích xuất đặc trưng (Feature Engineering).

LƯU Ý KIẾN TRÚC:
Toàn bộ mã nguồn chính thức và các bài test đã được chuyển dịch hoàn toàn sang gói `src.features`.
Tệp này chỉ được duy trì làm proxy chuyển tiếp cho các notebook và script cũ.
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
