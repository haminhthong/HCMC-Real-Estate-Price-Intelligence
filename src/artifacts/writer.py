"""Ghi model, calibration, comparables và các báo cáo của lần train hiện tại."""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    CALIBRATION_PATH,
    COMPARABLES_PATH,
    DATA_SUMMARY_PATH,
    FEATURE_CONTEXT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    REPORT_DIR,
)


def _write_json(data: dict[str, Any], destination: Path) -> None:
    """Ghi JSON UTF-8 có format ổn định để dễ đọc và review bằng Git."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def save_atomic_joblib(obj: Any, destination: Path) -> None:
    """Ghi joblib qua file tạm để không để lại model ghi dở."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        joblib.dump(obj, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        # Dọn file tạm nếu dump hoặc replace thất bại giữa chừng.
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _export_comparables(reference_df: pd.DataFrame) -> None:
    """Xuất bảng tham chiếu tối giản, không phụ thuộc Parquet/pyarrow."""
    rows: list[dict[str, Any]] = []
    for row in reference_df.to_dict(orient="records"):
        area = row.get("Area")
        price = row.get("Price")
        area_value = float(area) if pd.notna(area) and float(area) > 0 else None
        price_value = float(price) if pd.notna(price) and float(price) > 0 else None
        rows.append(
            {
                "property_group_id": str(row.get("property_group_id", "")),
                "property_type": str(row.get("Property Type", "Nhà riêng")),
                "location_area": str(row.get("location_area", "Unknown")),
                "area": area_value,
                "price_million": price_value,
                "unit_price_million_m2": (
                    round(price_value / area_value, 1)
                    if price_value and area_value
                    else None
                ),
                "bedrooms": int(row["Bedrooms"])
                if pd.notna(row.get("Bedrooms"))
                else None,
                "bathrooms": int(row["Bathrooms"])
                if pd.notna(row.get("Bathrooms"))
                else None,
                "floors": int(row["Floors"]) if pd.notna(row.get("Floors")) else None,
                "latitude": float(row["Latitude"])
                if pd.notna(row.get("Latitude"))
                else None,
                "longitude": float(row["Longitude"])
                if pd.notna(row.get("Longitude"))
                else None,
                "distance_to_cbd_km": round(float(row["distance_to_cbd_km"]), 2)
                if pd.notna(row.get("distance_to_cbd_km"))
                else None,
                "listing_date": str(row.get("listing_date", "")),
                "has_furniture": int(row.get("has_furniture", 0)),
                "car_alley": int(row.get("car_alley", 0)),
            }
        )
    COMPARABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(COMPARABLES_PATH, index=False)


def save_model_artifacts(
    model_version: str,
    pipeline: Any,
    feature_context: Any,
    calibration_result: dict[str, Any],
    evaluation_result: dict[str, Any],
    selection_result: dict[str, Any],
    data_summary: dict[str, Any],
    reference_df: pd.DataFrame,
    training_ranges: dict[str, list[float]],
    segment_unit_prices: dict[tuple[str, str], float],
    comparable_context: Any,
) -> dict[str, Path]:
    """Ghi đúng một snapshot model và ba báo cáo đọc được.

    Artifact runtime chỉ gồm model, feature context, calibration và comparables.
    Metrics, data summary và error analysis nằm trong ``reports/`` để tách code
    chạy dự báo khỏi phần trình bày kết quả.
    """
    save_atomic_joblib(pipeline, MODEL_PATH)
    _write_json(feature_context.to_dict(), FEATURE_CONTEXT_PATH)

    calibration_payload = dict(calibration_result)
    calibration_payload["comparable_context"] = comparable_context.to_dict()
    _write_json(calibration_payload, CALIBRATION_PATH)
    _export_comparables(reference_df)

    split_summary = data_summary.get("split_summary", {})
    metrics = {
        "model_version": model_version,
        "model_type": selection_result["selected_model_name"],
        "target_formulation": "total_price",
        "split_protocol": data_summary["split_protocol"],
        "train_rows": split_summary.get("train_rows", 0),
        "validation_rows": split_summary.get("validation_rows", 0),
        "calibration_rows": split_summary.get("calibration_rows", 0),
        "test_rows": split_summary.get("test_rows", 0),
        "validation": selection_result["selected_validation_metrics"],
        "test": evaluation_result["champion_metrics"],
        "baselines": evaluation_result["baselines"],
        "interval": evaluation_result["interval_metrics"],
    }
    _write_json(metrics, METRICS_PATH)

    summary_payload = dict(data_summary)
    summary_payload.update(
        {
            "model_version": model_version,
            "model_type": selection_result["selected_model_name"],
            "target_formulation": "total_price",
            "training_ranges": training_ranges,
            "segment_unit_prices": {
                f"{key[0]} | {key[1]}": float(value)
                for key, value in segment_unit_prices.items()
            },
            "comparable_reference_rows": len(reference_df),
        }
    )
    _write_json(summary_payload, DATA_SUMMARY_PATH)
    _write_json(evaluation_result["slice_analysis"], REPORT_DIR / "error_analysis.json")

    return {
        "model_path": MODEL_PATH,
        "feature_context_path": FEATURE_CONTEXT_PATH,
        "calibration_path": CALIBRATION_PATH,
        "comparables_path": COMPARABLES_PATH,
        "metrics_path": METRICS_PATH,
        "data_summary_path": DATA_SUMMARY_PATH,
        "error_analysis_path": REPORT_DIR / "error_analysis.json",
    }


__all__ = ["save_atomic_joblib", "save_model_artifacts"]
