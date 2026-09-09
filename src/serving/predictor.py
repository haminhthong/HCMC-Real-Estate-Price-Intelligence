"""Mô-đun điều phối dịch vụ dự báo giá (Price Intelligence Serving Predictor)."""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.artifacts.loader import load_production_model
from src.comparables import find_comparables
from src.config import MISSING_INDICATOR_FEATURES
from src.features.builder import build_features
from src.features.context import FeatureContext
from src.reliability.guards import check_reliability_guards

from .explain import explain_top_features


def _json_safe(value: Any) -> Any:
    """Chuẩn hóa response về các kiểu JSON hợp lệ trước khi trả qua API."""
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        return numeric_value if np.isfinite(numeric_value) else None
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
    """Dự báo giá cho một bất động sản và trả về gói Price Intelligence Response hoàn chỉnh.

    CẤU TRÚC 5 TRỤ CỘT CHUẨN MỰC:
    1. VALUATION: Ước lượng giá điểm (Point Listing-Price Estimate).
    2. UNCERTAINTY: Khoảng dự báo Conformal Interval (Coverage 80%).
    3. MARKET CONTEXT: Trung vị đơn giá phân khúc và trung vị các căn tương đồng.
    4. COMPARABLES: Danh sách các bất động sản tương đồng tham chiếu Top-K.
    5. RELIABILITY: Đánh giá độ tin cậy, rào chắn miền dữ liệu, rủi ro khoảng và điểm hoàn thiện.

    Args:
        values: Dictionary chứa các thuộc tính của bất động sản (có thể có as_of_date hoặc valuation_date).
        include_explanation: Cờ yêu cầu tính toán SHAP values.
        model_package: Tùy chọn gói mô hình đã tải sẵn (nếu có).

    Returns:
        Dictionary kết quả phân tầng: valuation, market_context, reliability, comparables, model.
    """
    if model_package is None:
        model_package = load_production_model()

    area_value = values.get("Area")
    if pd.isna(area_value) or not 5 <= float(area_value) <= 500:
        raise ValueError(
            "UNSUPPORTED_MARKET_SCOPE: Area phải nằm trong phạm vi hỗ trợ 5–500 m²."
        )

    reference_date = model_package.get("reference_date")
    as_of_date = values.get("as_of_date", values.get("valuation_date"))
    if as_of_date is None or pd.isna(as_of_date):
        as_of_date = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat()
    as_of_timestamp = pd.to_datetime(as_of_date, errors="coerce", utc=True)
    if pd.isna(as_of_timestamp):
        raise ValueError("as_of_date phải là ngày hợp lệ theo ISO format.")
    as_of_date = as_of_timestamp.date().isoformat()
    reference_timestamp = pd.to_datetime(reference_date, errors="coerce", utc=True)
    market_age_days = (
        int((as_of_timestamp - reference_timestamp).days)
        if pd.notna(reference_timestamp)
        else None
    )

    row = pd.DataFrame([values])
    row["as_of_date"] = as_of_date

    context_data = model_package.get("feature_context")
    if context_data:
        feature_context = FeatureContext.from_dict(context_data)
    else:
        package_features = set(model_package.get("features", []))
        feature_context = FeatureContext(
            reference_date=reference_date or as_of_date,
            missing_indicator_features=(
                list(MISSING_INDICATOR_FEATURES)
                if package_features.intersection(MISSING_INDICATOR_FEATURES)
                else []
            ),
        )
    feature_frame = build_features(row, context=feature_context)

    input_completeness = float(
        feature_frame.iloc[0].get(
            "input_completeness_score",
            feature_frame.iloc[0].get("data_quality_score", 100.0),
        )
    )
    target_formulation = model_package.get("target_formulation", "total_price")
    area_val = float(area_value)

    raw_pred = float(model_package["pipeline"].predict(feature_frame)[0])
    if target_formulation == "price_per_m2":
        predicted_price = max(float(np.expm1(raw_pred) * area_val), 0.0)
    else:
        predicted_price = max(float(np.expm1(raw_pred)), 0.0)

    # 2. Xác định khoảng dự báo Conformal Prediction (Prediction Interval)
    error_quantile = float(model_package.get("residual_log_quantile", 0.25))
    target_coverage = float(model_package.get("target_coverage", 0.8))
    if target_formulation == "price_per_m2":
        lower_bound = max(float(np.expm1(raw_pred - error_quantile) * area_val), 0.0)
        upper_bound = float(np.expm1(raw_pred + error_quantile) * area_val)
    else:
        lower_bound = max(float(np.expm1(raw_pred - error_quantile)), 0.0)
        upper_bound = float(np.expm1(raw_pred + error_quantile))

    # 3. Kiểm tra rào chắn miền và phân rã độ tin cậy
    warnings, domain_support, interval_risk, reliability_level = check_reliability_guards(
        feature_frame=feature_frame,
        values=values,
        model_package=model_package,
        predicted_price=predicted_price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        input_completeness_score=input_completeness,
        as_of_date=as_of_date,
    )

    # 4. Tra cứu đơn giá trung vị cùng phân khúc
    property_type = values.get("Property Type")
    location_area = values.get("location_area")
    segment_prices = model_package.get("segment_unit_prices", {})
    segment_unit_price = segment_prices.get((property_type, location_area))
    if segment_unit_price is None:
        segment_unit_price = segment_prices.get(f"{property_type} | {location_area}")

    # 5. Tìm kiếm bất động sản tương đồng (Comparable Properties Engine)
    comparable_values = dict(values)
    comparable_values["as_of_date"] = as_of_date
    comparables, comp_summary = find_comparables(
        model_package, comparable_values, n_matches=4
    )

    # 6. Tính SHAP nếu được yêu cầu
    should_explain = include_explanation or values.get("include_explanation", False)
    contributions = (
        explain_top_features(model_package, feature_frame)
        if should_explain
        else []
    )

    response = {
        # Cấu trúc phân tầng tiêu chuẩn 5 trụ cột
        "valuation": {
            "point_estimate_million": round(predicted_price, 1),
            "prediction_interval": {
                "lower_bound_million": round(lower_bound, 1),
                "upper_bound_million": round(upper_bound, 1),
                "target_coverage": target_coverage,
            },
        },
        "uncertainty": {
            "target_coverage": target_coverage,
            "lower_bound_million": round(lower_bound, 1),
            "upper_bound_million": round(upper_bound, 1),
            "interval_width_million": round(upper_bound - lower_bound, 1),
            "relative_interval_width": round((upper_bound - lower_bound) / max(predicted_price, 1.0), 3),
        },
        "market_context": {
            "segment_median_unit_price_million_m2": (
                round(float(segment_unit_price), 1)
                if segment_unit_price is not None
                else None
            ),
            "comparable_median_price_million": comp_summary["median_price_million"],
            "comparable_median_unit_price_million_m2": comp_summary["median_unit_price_million_m2"],
        },
        "reliability": {
            "overall": reliability_level,
            "reliability_level": reliability_level,
            "input_completeness_score": round(input_completeness, 1),
            "domain_support": domain_support,
            "interval_risk": interval_risk,
            "warnings": warnings,
        },
        "comparables": comparables,
        "explanation": {
            "top_contributions": contributions,
            "explanation_space": "log_target_space",
            "note": "Giá trị SHAP thể hiện mức độ đóng góp của đặc trưng trên thang log-target của mô hình, không đại diện cho quan hệ nhân quả.",
        },
        "model": {
            "version": model_package["version"],
            "model_type": model_package.get("model_type", "ExtraTreesRegressor"),
            "target_formulation": target_formulation,
            "model_status": model_package.get("model_status", "production_ready"),
            "valuation_as_of": as_of_date,
            "model_market_reference": reference_date,
            "market_age_days": market_age_days,
        },
        # Các trường phẳng tương thích ngược (Flat aliases)
        "predicted_price_million": round(predicted_price, 1),
        "lower_bound_million": round(lower_bound, 1),
        "upper_bound_million": round(upper_bound, 1),
        "confidence": reliability_level,
        "reliability_level": reliability_level,
        "model_version": model_package["version"],
        "model_status": model_package.get("model_status", "production_ready"),
        "valuation_as_of": as_of_date,
        "model_market_reference": reference_date,
        "market_age_days": market_age_days,
        "warnings": warnings,
        "data_quality_score": round(input_completeness, 1),
        "input_completeness_score": round(input_completeness, 1),
        "top_contributions": contributions,
        "segment_median_unit_price_million_m2": (
            round(float(segment_unit_price), 1)
            if segment_unit_price is not None
            else None
        ),
        "disclaimer": (
            "Kết quả là giá đăng tham khảo từ mô hình Machine Learning, "
            "không phải giá giao dịch thực tế hoặc văn bản thẩm định giá chuyên nghiệp."
        ),
    }
    return _json_safe(response)
