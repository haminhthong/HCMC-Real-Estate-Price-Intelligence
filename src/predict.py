"""[DEPRECATED LEGACY WRAPPER] Mô-đun tương thích ngược cho dự báo giá và giải thích.

LƯU Ý KIẾN TRÚC:
Toàn bộ mã nguồn chính thức, API FastAPI và ứng dụng Streamlit đã được chuyển dịch sang
`src.serving.predictor` và `src.artifacts.loader`.
Tệp này chỉ được duy trì làm proxy chuyển tiếp cho các notebook và Docker command cũ.
"""

from typing import Any

from src.artifacts.loader import clear_model_cache, load_production_model
from src.serving import (
    check_ood_guards,
    explain_top_features,
    find_comparables,
    friendly_feature_name,
    predict_one as _predict_one,
)


load_model = load_production_model


def predict_one(
    values: dict[str, Any],
    include_explanation: bool = False,
) -> dict[str, Any]:
    """Dự báo giá cho một bất động sản."""
    return _predict_one(values, include_explanation=include_explanation)


__all__ = [
    "clear_model_cache",
    "check_ood_guards",
    "explain_top_features",
    "find_comparables",
    "friendly_feature_name",
    "load_model",
    "load_production_model",
    "predict_one",
]
