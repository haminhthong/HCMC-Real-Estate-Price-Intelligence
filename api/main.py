"""FastAPI cho ước lượng giá niêm yết nhà ở tại TP.HCM."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import MODEL_PATH, MODEL_VERSION, RESIDENTIAL_TYPES, SUPPORTED_AREAS
from src.serving.predictor import predict_one

app = FastAPI(
    title="HCMC Real Estate Price Intelligence API",
    description=(
        "Ước lượng giá niêm yết tham khảo từ listing data, kèm khoảng conformal "
        "và các listing tương đồng tồn tại trước thời điểm định giá."
    ),
    version=MODEL_VERSION,
)


class PredictionRequest(BaseModel):
    """Các thuộc tính chính của bất động sản cần ước lượng."""

    model_config = ConfigDict(populate_by_name=True, allow_inf_nan=False)

    property_type: str = Field(..., alias="Property Type")
    location_area: str
    area: float = Field(..., ge=5, le=500, alias="Area")
    bedrooms: int | None = Field(default=None, ge=1, le=10, alias="Bedrooms")
    bathrooms: int | None = Field(default=None, ge=0, le=20, alias="Bathrooms")
    floors: int | None = Field(default=None, ge=0, le=100, alias="Floors")
    width: float | None = Field(default=None, gt=0, le=100, alias="Width")
    length: float | None = Field(default=None, gt=0, le=200, alias="Length")
    alley_width: float | None = Field(default=None, ge=0, le=30, alias="Alley Width")
    latitude: float | None = Field(default=None, ge=10.3, le=11.2, alias="Latitude")
    longitude: float | None = Field(default=None, ge=106.3, le=107.0, alias="Longitude")
    direction: str = Field(default="Không rõ", alias="Direction")
    position: str = Field(default="Không rõ", alias="Position")
    title: str | None = Field(default=None, alias="Title")
    description: str | None = Field(default=None, alias="Description")
    has_furniture: bool | None = None
    car_alley: bool | None = None
    near_market: bool | None = None
    near_school: bool | None = None
    is_urgent_sale: bool | None = None
    as_of_date: str | None = None
    include_explanation: bool = False

    @field_validator("property_type")
    @classmethod
    def supported_type(cls, value: str) -> str:
        """Chỉ nhận loại hình có trong phạm vi dữ liệu."""
        if value not in RESIDENTIAL_TYPES:
            raise ValueError(f"Loại bất động sản '{value}' chưa được hỗ trợ.")
        return value

    @field_validator("location_area")
    @classmethod
    def supported_area(cls, value: str) -> str:
        """Chỉ nhận khu vực có trong phạm vi dữ liệu."""
        if value not in SUPPORTED_AREAS:
            raise ValueError(f"Khu vực '{value}' chưa được hỗ trợ.")
        return value


class PredictionInterval(BaseModel):
    """Khoảng dự báo conformal trên thang triệu VND."""

    lower_million: float
    upper_million: float
    coverage: float = 0.8


class ComparableProperty(BaseModel):
    """Một listing tương đồng được lấy từ dữ liệu tham chiếu lịch sử."""

    property_type: str
    location_area: str
    area: float | None = None
    price_million: float | None = None
    unit_price_million_m2: float | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    floors: int | None = None
    distance_to_cbd_km: float | None = None
    similarity_score: float = 1.0


class PredictionResponse(BaseModel):
    """Kết quả gồm giá dự báo, khoảng bất định và tin đăng tham chiếu."""

    predicted_price_million: float
    prediction_interval: PredictionInterval
    comparables: list[ComparableProperty] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    as_of_date: str
    market_reference_date: str | None = None
    market_age_days: int | None = None
    model_version: str
    top_contributions: list[dict[str, Any]] = Field(default_factory=list)
    disclaimer: str


@app.get("/health", summary="Kiểm tra API và artifact model")
def health() -> dict[str, Any]:
    """Kiểm tra server có thấy model artifact hiện tại hay chưa."""
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "service": "HCMC Real Estate Price Intelligence API",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Ước lượng giá niêm yết",
)
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Trả về giá điểm, khoảng conformal, cảnh báo và tin đăng tương đồng."""
    try:
        payload = request.model_dump(by_alias=True)
        include_explanation = payload.pop("include_explanation", False)
        return predict_one(payload, include_explanation=include_explanation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


__all__ = ["app"]
