"""Mô-đun giải thích mô hình bằng SHAP TreeExplainer."""

from typing import Any

import numpy as np
import pandas as pd


def friendly_feature_name(raw_name: str) -> str:
    """Ánh xạ tên đặc trưng thô từ scikit-learn sang nhãn tiếng Việt thân thiện."""
    mapping = {
        "num__Area": "Diện tích đất (Area)",
        "num__Bedrooms": "Số phòng ngủ",
        "num__Bathrooms": "Số phòng vệ sinh",
        "num__Floors": "Số tầng",
        "num__Width": "Chiều rộng mặt tiền",
        "num__Length": "Chiều dài",
        "num__Alley Width": "Độ rộng hẻm",
        "num__distance_to_cbd_km": "Khoảng cách tới Quận 1 (CBD)",
        "num__days_from_train_reference": "Thời gian đăng tin",
        "num__input_completeness_score": "Độ đầy đủ thông tin",
        "num__has_furniture": "Nội thất",
        "num__car_alley": "Hẻm xe hơi",
        "num__near_market": "Gần chợ",
        "num__near_school": "Gần trường học",
        "num__is_urgent_sale": "Cần bán gấp",
    }
    if raw_name in mapping:
        return mapping[raw_name]

    if raw_name.startswith("cat__Property Type_"):
        return f"Loại hình: {raw_name.replace('cat__Property Type_', '')}"
    if raw_name.startswith("cat__location_area_"):
        return f"Khu vực: {raw_name.replace('cat__location_area_', '')}"
    if raw_name.startswith("cat__Direction_"):
        return f"Hướng nhà: {raw_name.replace('cat__Direction_', '')}"
    if raw_name.startswith("cat__Position_"):
        return f"Vị trí: {raw_name.replace('cat__Position_', '')}"

    return raw_name.replace("num__", "").replace("cat__", "")


def explain_top_features(
    model_package: dict[str, Any],
    feature_frame: pd.DataFrame,
) -> list[dict[str, str | float]]:
    """Sử dụng SHAP TreeExplainer trích xuất 5 đặc trưng ảnh hưởng lớn nhất trong không gian log-target."""
    pipeline = model_package["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(feature_frame)
    feature_names = preprocessor.get_feature_names_out()
    model = pipeline.named_steps["model"]

    # Chỉ hỗ trợ giải thích mô hình cây; mô hình tuyến tính và cơ sở trả rỗng.
    if not hasattr(model, "feature_importances_"):
        return []

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
        scores = np.asarray(shap_values)[0]
    except ImportError:
        # Thiếu SHAP thì không có giải thích cục bộ cho lần dự báo này.
        return []

    top_indices = np.argsort(np.abs(scores))[-5:][::-1]

    return [
        {
            "feature": str(feature_names[index]),
            "friendly_name": friendly_feature_name(str(feature_names[index])),
            "shap_value": round(float(scores[index]), 4),
        }
        for index in top_indices
    ]
