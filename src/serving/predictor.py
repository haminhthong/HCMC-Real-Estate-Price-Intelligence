"""Dự báo giá và lấy các tin đăng tương đồng cho một bất động sản."""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.artifacts.loader import load_model
from src.comparables import find_comparables
from src.config import MISSING_INDICATOR_FEATURES
from src.features.builder import build_features
from src.features.context import FeatureContext

from .explain import explain_top_features
from .input_validation import build_prediction_warnings, validate_supported_input


def _json_safe(value: Any) -> Any:
    """Chuẩn hóa số numpy/pandas và loại NaN trước khi trả qua API."""
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    return value


def predict_one(
    values: dict[str, Any],
    include_explanation: bool = False,
    model_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Trả về giá điểm, khoảng conformal, cảnh báo và tin đăng tương đồng."""
    if model_package is None:
        model_package = load_model()

    validate_supported_input(
        values,
        model_package.get("supported_areas", []),
        model_package.get("supported_property_types", []),
    )

    area_value = values.get("Area")
    if pd.isna(area_value) or not 5 <= float(area_value) <= 500:
        raise ValueError("Area phải nằm trong phạm vi hỗ trợ 5–500 m².")

    as_of_date = values.get("as_of_date", values.get("valuation_date"))
    if as_of_date is None or pd.isna(as_of_date):
        as_of_date = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat()
    as_of_timestamp = pd.to_datetime(as_of_date, errors="coerce", utc=True)
    if pd.isna(as_of_timestamp):
        raise ValueError("as_of_date phải là ngày hợp lệ theo ISO format.")
    as_of_date = as_of_timestamp.date().isoformat()

    reference_timestamp = pd.to_datetime(
        model_package.get("reference_date"), errors="coerce", utc=True
    )
    market_reference_date = None
    market_age_days = None
    if pd.notna(reference_timestamp):
        market_reference_date = reference_timestamp.date().isoformat()
        market_age_days = int((as_of_timestamp - reference_timestamp).days)

    row = pd.DataFrame([{**values, "as_of_date": as_of_date}])
    context_data = model_package.get("feature_context")
    if context_data:
        feature_context = FeatureContext.from_dict(context_data)
    else:
        package_features = set(model_package.get("features", []))
        feature_context = FeatureContext(
            reference_date=as_of_date,
            missing_indicator_features=(
                list(MISSING_INDICATOR_FEATURES)
                if package_features.intersection(MISSING_INDICATOR_FEATURES)
                else []
            ),
        )
    feature_frame = build_features(row, context=feature_context)

    completeness = float(
        feature_frame.iloc[0].get(
            "input_completeness_score",
            feature_frame.iloc[0].get("data_quality_score", 100.0),
        )
    )
    raw_prediction = float(model_package["pipeline"].predict(feature_frame)[0])
    predicted_price = max(float(np.expm1(raw_prediction)), 0.0)
    residual_quantile = float(model_package["residual_log_quantile"])
    target_coverage = float(model_package["target_coverage"])
    lower_bound = max(float(np.expm1(raw_prediction - residual_quantile)), 0.0)
    upper_bound = float(np.expm1(raw_prediction + residual_quantile))

    warnings = build_prediction_warnings(
        feature_frame=feature_frame,
        values=values,
        model_package=model_package,
        predicted_price=predicted_price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        input_completeness_score=completeness,
        as_of_date=as_of_date,
    )

    comparable_values = {**values, "as_of_date": as_of_date}
    comparables, _summary = find_comparables(
        model_package, comparable_values, n_matches=4
    )
    should_explain = include_explanation or values.get("include_explanation", False)
    contributions = (
        explain_top_features(model_package, feature_frame) if should_explain else []
    )

    response = {
        "predicted_price_million": round(predicted_price, 1),
        "prediction_interval": {
            "lower_million": round(lower_bound, 1),
            "upper_million": round(upper_bound, 1),
            "coverage": target_coverage,
        },
        "comparables": comparables,
        "warnings": warnings,
        "as_of_date": as_of_date,
        "market_reference_date": market_reference_date,
        "market_age_days": market_age_days,
        "model_version": model_package.get("version", "unknown"),
        "top_contributions": contributions,
        "disclaimer": (
            "Giá niêm yết tham khảo từ dữ liệu tin đăng, không phải giá giao dịch "
            "hoặc thẩm định pháp lý."
        ),
    }
    return _json_safe(response)


__all__ = ["predict_one"]
