"""Kiểm tra artifact sai không được âm thầm dùng giá trị mặc định."""

import json

import pytest

from src.artifacts import loader
from src.serving.predictor import predict_one


@pytest.fixture(autouse=True)
def clear_artifact_cache():
    loader.clear_model_cache()
    yield
    loader.clear_model_cache()


@pytest.mark.parametrize(
    "field,value",
    [
        ("residual_log_quantile", -1),
        ("residual_log_quantile", float("nan")),
        ("target_coverage", 0),
        ("target_coverage", 1),
    ],
)
def test_invalid_calibration_fails_loading(monkeypatch, field, value):
    read_json = loader._read_json

    def invalid_calibration(path):
        payload = read_json(path)
        if path == loader.CALIBRATION_PATH:
            payload[field] = value
        return payload

    monkeypatch.setattr(loader, "_read_json", invalid_calibration)
    with pytest.raises(ValueError):
        loader.load_model()


def test_reference_listings_are_strict_json():
    json.dumps(loader.load_model()["reference_listings"], allow_nan=False)


def test_missing_feature_context_does_not_create_request_context():
    package = dict(loader.load_model())
    package.pop("feature_context")
    with pytest.raises(KeyError, match="feature_context"):
        predict_one(
            {"Property Type": "Nhà riêng", "location_area": "Quận 1", "Area": 75},
            model_package=package,
        )
