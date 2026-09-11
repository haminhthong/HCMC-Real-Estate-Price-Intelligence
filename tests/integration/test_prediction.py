"""Kiểm thử tích hợp cho quy trình dự báo (Prediction Serving Workflow)."""

import json

import numpy as np
import pytest

from src.artifacts.loader import load_model
from src.serving.predictor import predict_one


@pytest.fixture(autouse=True)
def clear_cache():
    load_model.cache_clear()
    yield
    load_model.cache_clear()


def test_real_model_prediction_schema():
    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    result = predict_one(sample_input, include_explanation=False)

    assert result["predicted_price_million"] >= 0
    assert result["prediction_interval"]["lower_million"] >= 0
    assert (
        result["prediction_interval"]["upper_million"]
        >= result["prediction_interval"]["lower_million"]
    )
    assert result["top_contributions"] == []


def test_prediction_response_is_strict_json_serializable():
    result = predict_one(
        {
            "Property Type": "Nhà riêng",
            "location_area": "Quận 1",
            "Area": 75.0,
            "Bedrooms": 3,
        }
    )
    serialized = json.dumps(result, allow_nan=False)
    assert serialized is not None


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_shap_explanation_enabled_returns_top_contributions():
    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    result = predict_one(sample_input, include_explanation=True)
    assert isinstance(result["top_contributions"], list)
    assert len(result["top_contributions"]) <= 5


def test_input_at_min_max_bounds():
    min_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 5.0,
        "Bedrooms": 1,
        "Width": 0.1,
        "Length": 0.1,
    }
    max_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 500.0,
        "Bedrooms": 10,
        "Width": 100.0,
        "Length": 200.0,
    }
    result_min = predict_one(min_input)
    result_max = predict_one(max_input)
    assert result_min["predicted_price_million"] >= 0
    assert result_max["predicted_price_million"] >= 0


def test_unseen_category_in_train_is_rejected():
    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận Chưa Có",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    with pytest.raises(ValueError, match="chưa được hỗ trợ"):
        predict_one(sample_input)


def test_optional_fields_all_missing():
    minimal_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 50.0,
        "Bedrooms": 2,
    }
    result = predict_one(minimal_input)
    assert result["predicted_price_million"] > 0
    assert len(result["warnings"]) > 0


def test_conformal_residual_space_matches_inference_space():
    """Kiểm chứng residual conformal khớp với inference log-space:

    1. Mô hình dự báo trên log1p(Price).
    2. Conformal residual được tính trên thang log.
    3. Phân vị quantile q áp dụng đối xứng trên log-space: [y_hat_log - q, y_hat_log + q].
    4. Cận trên/dưới trong price-space: lower = expm1(log_pred - q), upper = expm1(log_pred + q).
    5. Do hàm mũ lồi, khoảng giá VND là bất đối xứng: upper - pred > pred - lower.
    """
    pkg = load_model()
    q = pkg["residual_log_quantile"]
    assert q > 0

    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    result = predict_one(sample_input)
    pred_price = result["predicted_price_million"]
    lower_price = result["prediction_interval"]["lower_million"]
    upper_price = result["prediction_interval"]["upper_million"]

    diff_upper = upper_price - pred_price
    diff_lower = pred_price - lower_price
    assert (
        diff_upper > diff_lower
    ), "Do hàm expm1 lồi, khoảng cách cận trên phải lớn hơn cận dưới"

    log_pred = np.log1p(pred_price)
    expected_lower = np.expm1(log_pred - q)
    expected_upper = np.expm1(log_pred + q)
    assert np.isclose(lower_price, expected_lower, rtol=1e-2)
    assert np.isclose(upper_price, expected_upper, rtol=1e-2)
