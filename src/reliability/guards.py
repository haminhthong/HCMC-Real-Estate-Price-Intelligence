"""Mô-đun rào chắn miền và kiểm soát độ tin cậy kết quả dự báo (Domain & Reliability Guardrails).

Thay thế cho khái niệm OOD Detector thống kê giả định bằng một hệ thống rào chắn thực tiễn:
1. Phạm vi phân vị đặc trưng P01 - P99 từ tập huấn luyện.
2. Kiểm tra hỗ trợ danh mục (khu vực hành chính, loại hình nhà ở).
3. Kiểm tra tính toàn vẹn đầu vào (tọa độ GPS, điểm hoàn thiện input completeness).
4. Phân khúc hỗ trợ thấp (LOW_SUPPORT_SEGMENT) và ngoại suy phân khúc hạng sang (P90/P95 target percentile).
5. Ngoại suy thời gian (TEMPORAL_EXTRAPOLATION) khi ngày định giá as_of_date cách xa kỳ huấn luyện.
6. Phân rã rủi ro khoảng dự báo Conformal Interval (tight, moderate, wide).
"""

from typing import Any

import numpy as np
import pandas as pd


def check_reliability_guards(
    feature_frame: pd.DataFrame,
    values: dict[str, Any],
    model_package: dict[str, Any],
    predicted_price: float,
    lower_bound: float,
    upper_bound: float,
    input_completeness_score: float,
    as_of_date: Any = None,
) -> tuple[list[str], str, str, str]:
    """Kiểm tra toàn diện các rào chắn miền và phân rã độ tin cậy.

    Args:
        feature_frame: Vector đặc trưng 1 dòng đã qua Feature Builder.
        values: Dictionary dữ liệu đầu vào gốc.
        model_package: Artifact package nạp từ models/artifacts.
        predicted_price: Giá dự báo điểm (triệu VND).
        lower_bound: Cận dưới Conformal (triệu VND).
        upper_bound: Cận trên Conformal (triệu VND).
        input_completeness_score: Điểm hoàn thiện đầu vào (0 - 100%).
        as_of_date: Ngày định giá tham chiếu.

    Returns:
        tuple gồm (warnings, domain_support, interval_risk, reliability_level).
    """
    warnings: list[str] = []

    # 1. Rào chắn phân vị P01 - P99 của các biến số học
    quantiles_dict = model_package.get(
        "training_quantiles",
        model_package.get("training_ranges", {}),
    )
    for feature, bounds in quantiles_dict.items():
        if feature in feature_frame.columns:
            val = feature_frame.iloc[0].get(feature)
            if (
                pd.notna(val)
                and np.isfinite(float(val))
                and not (bounds[0] <= float(val) <= bounds[1])
            ):
                warnings.append(
                    f"CẢNH BÁO PHẠM VI (OOD): Đặc trưng '{feature}'={val} nằm ngoài phân vị huấn luyện P01–P99 "
                    f"({bounds[0]:g}–{bounds[1]:g})."
                )

    # 2. Rào chắn hỗ trợ danh mục khu vực và loại hình
    location_area = values.get("location_area")
    supported_areas = model_package.get("supported_areas", [])
    if location_area and supported_areas and location_area not in supported_areas:
        warnings.append(
            "CẢNH BÁO KHU VỰC: Khu vực này chưa xuất hiện trong tập huấn luyện."
        )

    property_type = values.get("Property Type")
    supported_types = model_package.get("supported_property_types", [])
    if property_type and supported_types and property_type not in supported_types:
        warnings.append(
            "CẢNH BÁO LOẠI HÌNH: Loại bất động sản này chưa xuất hiện trong tập huấn luyện."
        )

    # 3. Rào chắn tọa độ GPS và độ hoàn thiện thông tin
    lat_val = values.get("Latitude")
    lon_val = values.get("Longitude")
    if lat_val is None or pd.isna(lat_val) or lon_val is None or pd.isna(lon_val):
        warnings.append(
            "CẢNH BÁO TỌA ĐỘ: Thiếu GPS (vĩ độ/kinh độ) nên mô hình không dùng được khoảng cách CBD."
        )

    if input_completeness_score < 60.0:
        warnings.append(
            f"CẢNH BÁO DỮ LIỆU THIẾU: Dữ liệu đầu vào chưa đầy đủ (Điểm hoàn thiện {input_completeness_score:.0f}/100)."
        )

    # 4. Rào chắn dữ liệu hóa phân khúc cao cấp (P90 Target Percentile) và phân khúc hỗ trợ thấp
    target_p90 = float(
        model_package.get(
            "target_p90",
            model_package.get("data_card", {})
            .get("target_percentiles", {})
            .get("p90", 22740.0),
        )
    )
    if predicted_price > target_p90:
        warnings.append(
            f"CẢNH BÁO NGOẠI SUY (LUXURY): Giá ước tính ({predicted_price:,.0f}M) vượt phân vị P90 tập huấn luyện "
            f"({target_p90:,.0f}M); thuộc vùng phân khúc cao cấp mật độ mẫu thưa, độ bất định tăng cao."
        )

    # Kiểm tra phân khúc mẫu ít (LOW_SUPPORT_SEGMENT)
    segment_counts = model_package.get("segment_sample_counts", {})
    seg_key = f"{property_type} | {location_area}"
    if segment_counts and seg_key in segment_counts and segment_counts[seg_key] < 5:
        warnings.append(
            f"CẢNH BÁO MẪU ÍT (LOW_SUPPORT_SEGMENT): Phân khúc '{seg_key}' chỉ có {segment_counts[seg_key]} mẫu "
            f"trong dữ liệu tham chiếu."
        )

    # 5. Rào chắn ngoại suy thời gian (TEMPORAL_EXTRAPOLATION)
    if "market_time_offset_days" in feature_frame.columns:
        raw_offset = feature_frame.iloc[0]["market_time_offset_days"]
    elif "days_from_train_reference" in feature_frame.columns:
        raw_offset = feature_frame.iloc[0]["days_from_train_reference"]
    else:
        raw_offset = 0.0

    offset_days = float(raw_offset) if pd.notna(raw_offset) else 0.0

    if pd.notna(offset_days) and abs(offset_days) > 180:
        warnings.append(
            f"CẢNH BÁO NGOẠI SUY THỜI GIAN (TEMPORAL_EXTRAPOLATION): Mốc thời gian định giá chênh lệch "
            f"{offset_days:+.0f} ngày so với chu kỳ huấn luyện; biến động vĩ mô thị trường có thể làm sai lệch giá."
        )

    # Tuổi thị trường được tính theo ngày định giá thực tế, không phải ngày
    # mặc định của feature vector. Đây là cảnh báo riêng cho model stale.
    reference_date = pd.to_datetime(
        model_package.get("reference_date"), errors="coerce", utc=True
    )
    valuation_date = pd.to_datetime(as_of_date, errors="coerce", utc=True)
    if pd.notna(reference_date) and pd.notna(valuation_date):
        market_age_days = int((valuation_date - reference_date).days)
        if market_age_days > 180:
            warnings.append(
                f"STALE_MARKET_MODEL: Model tham chiếu đã cách ngày định giá {market_age_days} ngày (>180 ngày)."
            )

    # 6. Phân rã rủi ro khoảng Conformal Interval và xếp hạng độ tin cậy
    relative_width = (upper_bound - lower_bound) / max(predicted_price, 1.0)
    interval_risk = (
        "wide_interval"
        if relative_width > 0.8
        else ("moderate" if relative_width > 0.4 else "tight")
    )
    # Hỗ trợ cả 'warning' và 'warning_ood' để giữ tính tương thích ngược với các assertion test cũ
    has_domain_warning = any(
        "CẢNH BÁO" in warning or warning.startswith("STALE_") for warning in warnings
    )
    domain_support = "warning_ood" if has_domain_warning else "in_domain"

    if (
        domain_support == "warning_ood"
        or interval_risk == "wide_interval"
        or input_completeness_score < 60.0
    ):
        reliability_level = "low"
    elif interval_risk == "moderate":
        reliability_level = "medium"
    else:
        reliability_level = "high"

    return warnings, domain_support, interval_risk, reliability_level
