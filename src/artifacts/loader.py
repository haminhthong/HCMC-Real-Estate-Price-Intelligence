"""Nạp bốn artifact phẳng dùng chung cho API và Streamlit."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    CALIBRATION_PATH,
    COMPARABLES_PATH,
    DATA_SUMMARY_PATH,
    FEATURE_CONTEXT_PATH,
    MODEL_PATH,
)


def _read_json(path: Path) -> dict[str, Any]:
    """Đọc JSON UTF-8 và báo lỗi rõ ràng nếu artifact thiếu hoặc hỏng."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Artifact JSON không hợp lệ: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Artifact JSON phải là object: {path}")
    return value


def _load_reference_listings() -> list[dict[str, Any]]:
    """Nạp bảng comparables và chuyển NaN thành None trước khi phục vụ API."""
    if not COMPARABLES_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy bảng comparables: {COMPARABLES_PATH}")
    reference_df = pd.read_csv(COMPARABLES_PATH)
    return reference_df.where(pd.notna(reference_df), None).to_dict(orient="records")


@lru_cache(maxsize=1)
def load_model(model_override_path: Path | str | None = None) -> dict[str, Any]:
    """Nạp model cùng các context cần thiết cho inference.

    Mọi thành phần của một lần dự báo đều đọc từ thư mục ``artifacts/`` hiện tại;
    không có fallback sang artifact cũ.
    """
    model_path = Path(model_override_path) if model_override_path else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Chưa có model artifact: {model_path}. Hãy chạy python -m src.pipeline train."
        )

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        pipeline = loaded["pipeline"]
    else:
        pipeline = loaded
    if not hasattr(pipeline, "predict"):
        raise ValueError(
            "Model artifact không đúng cấu trúc: cần phương thức predict()."
        )

    feature_context = _read_json(FEATURE_CONTEXT_PATH)
    calibration = _read_json(CALIBRATION_PATH)
    data_summary = _read_json(DATA_SUMMARY_PATH)
    comparable_context = calibration.get("comparable_context", {})
    references = _load_reference_listings()

    segment_prices: dict[tuple[str, str], float] = {}
    for key, value in data_summary.get("segment_unit_prices", {}).items():
        if " | " in key:
            property_type, location_area = key.split(" | ", 1)
            segment_prices[(property_type, location_area)] = float(value)

    return {
        "pipeline": pipeline,
        "version": data_summary.get("model_version", "unknown"),
        "model_type": data_summary.get("model_type", "ExtraTreesRegressor"),
        "target_formulation": data_summary.get("target_formulation", "total_price"),
        "features": (
            feature_context.get("numeric_features", [])
            + feature_context.get("categorical_features", [])
            + feature_context.get("flag_features", [])
            + feature_context.get("missing_indicator_features", [])
        ),
        "feature_context": feature_context,
        "reference_date": feature_context.get("reference_date"),
        "supported_areas": data_summary.get("district_coverage", []),
        "supported_property_types": data_summary.get("property_types", []),
        "training_ranges": data_summary.get("training_ranges", {}),
        "residual_log_quantile": calibration["residual_log_quantile"],
        "target_coverage": calibration.get("target_coverage", 0.8),
        "split_protocol": data_summary.get("split_protocol", "group_isolated_temporal"),
        "reference_listings": references,
        "comparable_context": comparable_context,
        "segment_unit_prices": segment_prices,
        "segment_sample_counts": comparable_context.get("segment_counts", {}),
    }


def clear_model_cache() -> None:
    """Xóa cache để lần dự báo sau đọc lại artifact mới."""
    load_model.cache_clear()


__all__ = ["clear_model_cache", "load_model"]
