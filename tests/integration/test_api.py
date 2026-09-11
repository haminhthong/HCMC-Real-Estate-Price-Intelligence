"""Kiểm thử tích hợp cho FastAPI endpoints (/health, /predict)."""

from fastapi.testclient import TestClient


def test_health_schema(api_client: TestClient):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert {"status", "model_loaded", "model_version", "model_type"}.issubset(data)
    assert data["model_loaded"] is True
    assert data["status"] == "ok"


def test_health_rejects_corrupt_model(api_client: TestClient, monkeypatch):
    """File tồn tại nhưng không đọc được phải làm readiness thất bại (503)."""

    def broken_model():
        raise EOFError("Artifact bị cắt ngắn")

    monkeypatch.setattr("api.main.load_model", broken_model)
    assert api_client.get("/health").status_code == 503


def test_invalid_area_is_rejected(api_client: TestClient):
    payload = {
        "Property Type": "Nhà riêng",
        "location_area": "Hà Nội",
        "Area": 50,
        "Bedrooms": 2,
    }
    assert api_client.post("/predict", json=payload).status_code == 422


def test_missing_required_value_is_rejected(api_client: TestClient):
    payload = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Bedrooms": 2,
    }
    assert api_client.post("/predict", json=payload).status_code == 422


def test_nonfinite_area_is_rejected(api_client: TestClient):
    payload = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": "NaN",
        "Bedrooms": 2,
    }
    assert api_client.post("/predict", json=payload).status_code == 422


def test_prediction_response_schema(api_client: TestClient, monkeypatch):
    expected = {
        "predicted_price_million": 7850.0,
        "prediction_interval": {
            "lower_million": 6400.0,
            "upper_million": 9300.0,
            "coverage": 0.8,
        },
        "comparables": [],
        "warnings": [],
        "as_of_date": "2026-09-10",
        "market_reference_date": None,
        "market_age_days": None,
        "model_version": "1.2.0",
        "top_contributions": [],
        "disclaimer": "Giá tham khảo.",
    }
    monkeypatch.setattr("api.main.predict_one", lambda _val, **_kwargs: expected)
    response = api_client.post(
        "/predict",
        json={
            "Property Type": "Nhà riêng",
            "location_area": "Quận 1",
            "Area": 50,
            "Bedrooms": 2,
        },
    )
    assert response.status_code == 200
    assert response.json()["predicted_price_million"] == 7850.0
    assert response.json()["prediction_interval"]["coverage"] == 0.8


def test_real_prediction_returns_interval_and_comparables(api_client: TestClient):
    response = api_client.post(
        "/predict",
        json={
            "Property Type": "Nhà riêng",
            "location_area": "Quận 1",
            "Area": 75.0,
            "Bedrooms": 3,
            "as_of_date": "2026-09-10",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert (
        data["prediction_interval"]["upper_million"]
        >= data["prediction_interval"]["lower_million"]
    )
    assert len(data["comparables"]) >= 1
    assert "comparable_summary" in data
    assert "input_quality" in data
    assert data["input_quality"]["completeness_score"] > 0


def test_unknown_endpoint_returns_404(api_client: TestClient):
    assert api_client.get("/market/districts").status_code == 404
