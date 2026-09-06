"""Mô-đun kiểm tra cảnh báo Out-Of-Distribution (OOD) và phân rã độ tin cậy."""

from typing import Any
import numpy as np
import pandas as pd


def check_ood_guards(
    feature_frame: pd.DataFrame,
    values: dict[str, Any],
    model_package: dict[str, Any],
    predicted_price: float,
    lower_bound: float,
    upper_bound: float,
    data_quality_score: float,
) -> tuple[list[str], str, str, str]:
    """Kiểm tra các ngưỡng bảo vệ OOD và phân rã độ tin cậy kết quả dự báo.

    Returns:
        tuple gồm (warnings, domain_support, interval_risk, reliability_level).
    """
    warnings: list[str] = []

    # 1. Phân vị P01 - P99 của các đặc trưng số học
    quantiles_dict = model_package.get(
        "training_quantiles",
        model_package.get("training_ranges", {}),
    )
    for feature, bounds in quantiles_dict.items():
        if feature in feature_frame.columns:
            val = feature_frame.iloc[0].get(feature)
            if pd.notna(val) and np.isfinite(float(val)) and not (
                bounds[0] <= float(val) <= bounds[1]
            ):
                warnings.append(
                    f"CẢNH BÁO PHẠM VI (OOD): Đặc trưng '{feature}'={val} nằm ngoài phân vị huấn luyện P01–P99 "
                    f"({bounds[0]:g}–{bounds[1]:g})."
                )

    # 2. Kiểm tra thuộc danh mục huấn luyện
    location_area = values.get("location_area")
    if location_area and location_area not in model_package.get("supported_areas", []):
        warnings.append("CẢNH BÁO KHU VỰC: Khu vực này chưa xuất hiện trong tập huấn luyện.")

    property_type = values.get("Property Type")
    if property_type and property_type not in model_package.get("supported_property_types", []):
        warnings.append("CẢNH BÁO LOẠI HÌNH: Loại bất động sản này chưa xuất hiện trong tập huấn luyện.")

    if values.get("Latitude") is None or values.get("Longitude") is None:
        warnings.append("CẢNH BÁO TỌA ĐỘ: Thiếu GPS (vĩ độ/kinh độ) nên mô hình không dùng được khoảng cách CBD.")

    if data_quality_score < 60:
        warnings.append(
            f"CẢNH BÁO DỮ LIỆU THIẾU: Dữ liệu đầu vào chưa đầy đủ (Điểm hoàn thiện {data_quality_score:.0f}/100)."
        )

    # 3. Cảnh báo ngoại suy phân khúc cao cấp (> 15 tỷ VND)
    if predicted_price > 15_000:
        warnings.append(
            "CẢNH BÁO NGOẠI SUY (LUXURY): Giá dự báo > 15 tỷ VND thuộc vùng phân khúc cao cấp dữ liệu thưa; "
            "khoảng dự báo mở rộng và mức độ hiệu chuẩn hạn chế."
        )

    # 4. Phân rã độ tin cậy (Decomposed Reliability)
    relative_width = (upper_bound - lower_bound) / max(predicted_price, 1.0)
    interval_risk = (
        "wide_interval"
        if relative_width > 0.8
        else ("moderate" if relative_width > 0.4 else "tight")
    )
    domain_support = "warning_ood" if any("CẢNH BÁO" in w for w in warnings) else "in_domain"

    if domain_support == "warning_ood" or interval_risk == "wide_interval" or data_quality_score < 60:
        reliability_level = "low"
    elif interval_risk == "moderate":
        reliability_level = "medium"
    else:
        reliability_level = "high"

    return warnings, domain_support, interval_risk, reliability_level
