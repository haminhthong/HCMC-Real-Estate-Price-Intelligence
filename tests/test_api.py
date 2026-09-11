from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_schema():
    response = client.get("/health")
    assert response.status_code == 200
    assert {"status", "model_loaded"}.issubset(response.json())
    assert response.json()["model_loaded"] is True
    assert response.json()["model_type"]


def test_health_rejects_corrupt_model(monkeypatch):
    """File tồn tại nhưng không đọc được phải làm readiness thất bại."""

    def broken_model():
        raise EOFError("Artifact bị cắt ngắn")

    monkeypatch.setattr("api.main.load_model", broken_model)
    assert client.get("/health").status_code == 503


def test_invalid_area_is_rejected():
    payload = {
        "Property Type": "Nhà riêng",
        "location_area": "Hà Nội",
        "Area": 50,
        "Bedrooms": 2,
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_required_value_is_rejected():
    payload = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Bedrooms": 2,
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_nonfinite_area_is_rejected():
    payload = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": "NaN",
        "Bedrooms": 2,
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_prediction_response_schema(monkeypatch):
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
    response = client.post(
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


def test_real_prediction_returns_interval_and_comparables():
    response = client.post(
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


def test_unknown_endpoint_is_not_part_of_public_contract():
    assert client.get("/market/districts").status_code == 404
