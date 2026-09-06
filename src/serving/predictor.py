"""Mô-đun điều phối dịch vụ dự báo giá (Price Intelligence Serving Predictor)."""

from typing import Any
import numpy as np
import pandas as pd

from src.artifacts.loader import load_production_model
from src.features.builder import make_features
from .comparables import find_comparables
from .explain import explain_top_features
from .ood import check_ood_guards


def predict_one(
    values: dict[str, Any],
    include_explanation: bool = False,
    model_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dự báo giá cho một bất động sản và trả về gói Price Intelligence Response hoàn chỉnh.

    Quy trình 7 bước Serving:
    1. Tải gói mô hình hoạt động (Active Version từ production.json hoặc LRU cache).
    2. Shared Feature Engineering với reference_date cố định từ tập Train/Dev.
    3. Dự báo điểm trung tâm (point estimate) và quy đổi từ log-scale.
    4. Xác định khoảng dự báo Conformal Prediction (Prediction Interval 80%) không phụ thuộc phân phối.
    5. Kiểm tra cảnh báo Out-Of-Distribution (OOD) bằng phân vị robust P01–P99 và cảnh báo phân khúc hạng sang.
    6. Đánh giá độ tin cậy phân rã (Decomposed Reliability: Overall, Data Completeness, Domain Support, Interval Risk).
    7. Trích xuất top 5 đặc trưng SHAP (nếu yêu cầu) và tra cứu bất động sản tương đồng (Comparable Properties).

    Args:
        values: Dictionary chứa các thuộc tính của bất động sản.
        include_explanation: Cờ yêu cầu tính toán SHAP values.
        model_package: Tùy chọn gói mô hình đã tải sẵn (nếu có).

    Returns:
        Dictionary kết quả phân tầng: valuation, market_context, reliability, comparables, model.
    """
    if model_package is None:
        model_package = load_production_model()

    reference_date = model_package.get("reference_date")
    row = pd.DataFrame([values])
    feature_frame = make_features(row, reference_date=reference_date)

    data_quality_score = float(
        feature_frame.iloc[0].get(
            "input_completeness_score",
            feature_frame.iloc[0].get("data_quality_score", 100.0),
        )
    )
    target_formulation = model_package.get("target_formulation", "total_price")
    area_val = (
        float(values.get("Area", 1.0))
        if pd.notna(values.get("Area")) and float(values.get("Area")) > 0
        else 1.0
    )

    raw_pred = float(model_package["pipeline"].predict(feature_frame)[0])
    if target_formulation == "price_per_m2":
        predicted_price = max(float(np.expm1(raw_pred) * area_val), 0.0)
    else:
        predicted_price = max(float(np.expm1(raw_pred)), 0.0)

    # 4. Xác định khoảng dự báo Conformal Prediction (Prediction Interval)
    error_quantile = float(model_package.get("residual_log_quantile", 0.25))
    target_coverage = float(model_package.get("target_coverage", 0.8))
    if target_formulation == "price_per_m2":
        lower_bound = max(float(np.expm1(raw_pred - error_quantile) * area_val), 0.0)
        upper_bound = float(np.expm1(raw_pred + error_quantile) * area_val)
    else:
        lower_bound = max(float(np.expm1(raw_pred - error_quantile)), 0.0)
        upper_bound = float(np.expm1(raw_pred + error_quantile))

    # 5 & 6. Kiểm tra cảnh báo OOD và độ tin cậy phân rã
    warnings, domain_support, interval_risk, reliability_level = check_ood_guards(
        feature_frame=feature_frame,
        values=values,
        model_package=model_package,
        predicted_price=predicted_price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        data_quality_score=data_quality_score,
    )

    # Tra cứu đơn giá trung vị cùng phân khúc
    property_type = values.get("Property Type")
    location_area = values.get("location_area")
    segment_unit_price = model_package.get("segment_unit_prices", {}).get(
        (property_type, location_area)
    )

    # Tìm kiếm bất động sản tương đồng (Comparable Properties Engine)
    comparables, comp_summary = find_comparables(model_package, values, n_matches=4)

    # Tính SHAP nếu được yêu cầu
    should_explain = include_explanation or values.get("include_explanation", False)
    contributions = (
        explain_top_features(model_package, feature_frame)
        if should_explain
        else []
    )

    return {
        # Cấu trúc phân tầng tiêu chuẩn
        "valuation": {
            "point_estimate_million": round(predicted_price, 1),
            "prediction_interval": {
                "lower_bound_million": round(lower_bound, 1),
                "upper_bound_million": round(upper_bound, 1),
                "target_coverage": target_coverage,
            },
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
            "input_completeness_score": round(data_quality_score, 1),
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
        },
        # Các trường phẳng tương thích ngược (Flat aliases)
        "predicted_price_million": round(predicted_price, 1),
        "lower_bound_million": round(lower_bound, 1),
        "upper_bound_million": round(upper_bound, 1),
        "confidence": reliability_level,
        "reliability_level": reliability_level,
        "model_version": model_package["version"],
        "warnings": warnings,
        "data_quality_score": round(data_quality_score, 1),
        "input_completeness_score": round(data_quality_score, 1),
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
