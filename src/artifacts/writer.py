"""Mô-đun ghi và lưu trữ Artifacts chuẩn hóa, có quản lý phiên bản và tách rời Comparables."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from src.config import (
    DATA_CARD_PATH,
    ERROR_ANALYSIS_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    ROOT_DIR,
    logger,
)


def _write_json(data: dict[str, Any], destination: Path) -> None:
    """Ghi tệp JSON định dạng UTF-8 có căn lề thụt dòng."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_atomic_joblib(obj: Any, destination: Path) -> None:
    """Ghi file joblib nguyên tử (Atomic Write) qua tệp tạm `.tmp`."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(obj, temp_path)
    os.replace(temp_path, destination)


def _sha256(path: Path) -> str:
    """Tính SHA256 theo từng block để manifest tái lập được với file lớn."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_artifacts(
    version: str,
    pipeline: Any,
    feature_context: Any,
    calibration_result: dict[str, Any],
    dataset_manifest: Any,
    split_manifest: Any,
    evaluation_result: dict[str, Any],
    selection_result: dict[str, Any],
    promotion_result: dict[str, Any],
    data_card: dict[str, Any],
    reference_df: pd.DataFrame,
    training_ranges: dict[str, list[float]],
    training_quantiles: dict[str, list[float]],
    segment_unit_prices: dict[tuple[str, str], float],
    comparable_context: Any = None,
) -> dict[str, Path]:
    """Ghi toàn bộ artifacts theo cấu trúc thư mục versioned và decoupled comparables."""
    v_tag = f"v{version}" if not version.startswith("v") else version
    models_dir = ROOT_DIR / "models" / v_tag
    reference_dir = ROOT_DIR / "reference" / v_tag
    now_hcmc = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    timestamp_tag = now_hcmc.strftime("%Y%m%d_%H%M%S")
    runs_dir = ROOT_DIR / "artifacts" / "runs" / f"run_{timestamp_tag}"

    # Version đã tồn tại là immutable: không được chạy lại rồi âm thầm ghi đè.
    if models_dir.exists() or reference_dir.exists():
        raise FileExistsError(
            f"Release {v_tag} đã tồn tại; hãy dùng version/build mới để bảo toàn artifact immutable."
        )
    models_dir.mkdir(parents=True, exist_ok=False)
    reference_dir.mkdir(parents=True, exist_ok=False)
    runs_dir.mkdir(parents=True, exist_ok=False)

    # 1. Lưu pipeline vào thư mục version
    save_atomic_joblib(pipeline, models_dir / "model.joblib")

    # 2. Lưu feature context
    _write_json(feature_context.to_dict(), models_dir / "feature_context.json")

    # 3. Lưu calibration info
    _write_json(calibration_result, models_dir / "calibration.json")

    # 4. Lưu manifest
    manifest_bundle = {
        "dataset_manifest": dataset_manifest.to_dict(),
        "split_manifest": split_manifest.to_dict(),
    }
    _write_json(manifest_bundle, models_dir / "manifest.json")

    # 5. Lưu comparable context nếu có
    if comparable_context is not None:
        comp_ctx_dict = comparable_context.to_dict() if hasattr(comparable_context, "to_dict") else comparable_context
        _write_json(comp_ctx_dict, models_dir / "comparable_context.json")
        _write_json(comp_ctx_dict, runs_dir / "comparable_context.json")
        _write_json(comp_ctx_dict, ROOT_DIR / "artifacts" / "comparable_context.json")

    # 6. Lưu metadata version
    target_p90 = float(data_card.get("target_percentiles", {}).get("p90", 22740.0))
    metadata = {
        "version": version,
        "model_type": selection_result["selected_model_name"],
        "target_formulation": selection_result["selected_target_fmt"],
        "promotion": promotion_result,
        "supported_areas": sorted(reference_df["location_area"].unique().tolist()) if "location_area" in reference_df else [],
        "supported_property_types": sorted(reference_df["Property Type"].unique().tolist()) if "Property Type" in reference_df else [],
        "training_ranges": training_ranges,
        "training_quantiles": training_quantiles,
        "target_p90": target_p90,
        "segment_unit_prices": {
            f"{key[0]} | {key[1]}": float(value)
            for key, value in segment_unit_prices.items()
        },
        "created_at": now_hcmc.isoformat(),
    }
    _write_json(metadata, models_dir / "metadata.json")

    # 6. Tách rời bảng tham chiếu Comparables (Decoupled Reference Dataset)
    ref_clean_rows = []
    for _, row in reference_df.iterrows():
        area_val = float(row["Area"]) if pd.notna(row.get("Area")) and float(row.get("Area")) > 0 else None
        price_val = float(row["Price"]) if pd.notna(row.get("Price")) and float(row.get("Price")) > 0 else None
        unit_price = round(price_val / area_val, 1) if (price_val and area_val) else None
        ref_clean_rows.append(
            {
                "property_group_id": str(row.get("property_group_id", "")),
                "property_type": str(row.get("Property Type", "Nhà riêng")),
                "location_area": str(row.get("location_area", "Unknown")),
                "area": area_val,
                "price_million": price_val,
                "unit_price_million_m2": unit_price,
                "bedrooms": int(row["Bedrooms"]) if pd.notna(row.get("Bedrooms")) else None,
                "bathrooms": int(row["Bathrooms"]) if pd.notna(row.get("Bathrooms")) else None,
                "floors": int(row["Floors"]) if pd.notna(row.get("Floors")) else None,
                "latitude": float(row["Latitude"]) if pd.notna(row.get("Latitude")) else None,
                "longitude": float(row["Longitude"]) if pd.notna(row.get("Longitude")) else None,
                "distance_to_cbd_km": round(float(row["distance_to_cbd_km"]), 2) if pd.notna(row.get("distance_to_cbd_km")) else None,
                "listing_date": str(row.get("listing_date", "")),
                "has_furniture": int(row.get("has_furniture", 0)),
                "car_alley": int(row.get("car_alley", 0)),
            }
        )
    ref_export_df = pd.DataFrame(ref_clean_rows)
    try:
        ref_export_df.to_parquet(reference_dir / "comparables.parquet", index=False)
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("Không thể xuất file parquet (sẽ xuất CSV): %s", exc)
    ref_export_df.to_csv(reference_dir / "comparables.csv", index=False)

    # 7. Ghi checksum của release. manifest.json tự loại khỏi danh sách hash
    # vì hash của chính nó sẽ tạo vòng lặp không thể ổn định.
    checksum_files: dict[str, str] = {}
    for root in (models_dir, reference_dir):
        for path in root.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                checksum_files[str(path.relative_to(ROOT_DIR))] = _sha256(path)
    manifest_bundle["release"] = {
        "version": v_tag,
        "immutable": True,
        "checksums_sha256": checksum_files,
        "manifest_excludes": ["manifest.json"],
    }
    _write_json(manifest_bundle, models_dir / "manifest.json")

    # Candidate luôn được ghi nhận. Chỉ artifact đạt Release Gate mới được
    # promote sang production; research-only không được thay đổi pointer hiện tại.
    release_ready = bool(
        promotion_result.get("release_gate", {}).get(
            "production_ready", promotion_result.get("deployment_approved", False)
        )
    )
    _write_json(
        {"candidate_version": v_tag, "production_ready": release_ready},
        ROOT_DIR / "models" / "candidate.json",
    )
    if release_ready:
        _write_json({"active_version": v_tag}, ROOT_DIR / "models" / "production.json")
    else:
        logger.warning(
            "Release %s là research_only; production.json được giữ nguyên.",
            v_tag,
        )

    # 8. Lưu snapshot vào runs/
    _write_json(dataset_manifest.to_dict(), runs_dir / "dataset_manifest.json")
    _write_json(split_manifest.to_dict(), runs_dir / "split_manifest.json")
    _write_json(selection_result["validation_benchmarks"], runs_dir / "model_selection.json")
    _write_json(calibration_result, runs_dir / "calibration.json")
    _write_json(evaluation_result["champion_metrics"], runs_dir / "test_metrics.json")
    _write_json(evaluation_result["slice_analysis"], runs_dir / "slice_metrics.json")
    _write_json(data_card, runs_dir / "data_card.json")

    # 9. Ghi các file root artifacts truyền thống cho tương thích ngược
    test_metrics = evaluation_result["champion_metrics"]
    int_metrics = evaluation_result["interval_metrics"]
    metrics_flat = {
        "model_version": version,
        "deployed_model": selection_result["selected_model_name"],
        "target_formulation": selection_result["selected_target_fmt"],
        "train_rows": split_manifest.train_rows,
        "validation_rows": split_manifest.validation_rows,
        "calibration_rows": split_manifest.calibration_rows,
        "test_rows": split_manifest.test_rows,
        **test_metrics,
        "prediction_interval_target_coverage": int_metrics["target_coverage"],
        "prediction_interval_test_coverage": int_metrics["actual_coverage"],
        "coverage_gap": int_metrics["coverage_gap"],
        "mean_interval_width_million": int_metrics["mean_interval_width_million"],
        "median_interval_width_million": int_metrics["median_interval_width_million"],
        "relative_interval_width": int_metrics["relative_interval_width"],
        "deployment_approved": promotion_result["deployment_approved"],
        "promotion_status": promotion_result["promotion_status"],
        "selection_status": promotion_result["selection_status"],
        "production_readiness": promotion_result["production_readiness"],
        "deployment_reason": promotion_result["promotion_reason"],
    }
    _write_json(metrics_flat, METRICS_PATH)

    comparison = {
        "selection_split": "validation",
        "selection_metric": "mae_million",
        "recommended_model": selection_result["selected_model_name"],
        "deployed_model": selection_result["selected_model_name"],
        "selected_target_formulation": selection_result["selected_target_fmt"],
        "deployment_approved": promotion_result["deployment_approved"],
        "promotion_status": promotion_result["promotion_status"],
        "selection_status": promotion_result["selection_status"],
        "production_readiness": promotion_result["production_readiness"],
        "deployment_reason": promotion_result["promotion_reason"],
        "validation_candidate_benchmarks": selection_result["validation_benchmarks"],
        "test_report_only": {
            selection_result["selected_model_name"]: test_metrics,
            "naive_median": evaluation_result["baselines"]["naive_median"],
            "district_property_segment_median": evaluation_result["baselines"]["district_property_segment_median"],
        },
    }
    _write_json(comparison, MODEL_COMPARISON_PATH)
    _write_json(evaluation_result["slice_analysis"], ERROR_ANALYSIS_PATH)
    _write_json(data_card, DATA_CARD_PATH)

    # 10. Ghi gói legacy chỉ khi release đã được duyệt. Loader production không
    # còn fallback âm thầm sang file này.
    legacy_artifact = {
        "pipeline": pipeline,
        "model_type": selection_result["selected_model_name"],
        "version": version,
        "artifact_schema_version": 3,
        "features": feature_context.model_features,
        "supported_areas": metadata["supported_areas"],
        "supported_property_types": metadata["supported_property_types"],
        "training_ranges": training_ranges,
        "training_quantiles": training_quantiles,
        "residual_log_quantile": calibration_result["residual_log_quantile"],
        "target_coverage": calibration_result["target_coverage"],
        "target_formulation": selection_result["selected_target_fmt"],
        "reference_date": feature_context.reference_date,
        "split_protocol": "grouped temporal split 60/15/10/15 with 2-phase champion refit",
        "segment_unit_prices": segment_unit_prices,
        "reference_listings": ref_clean_rows,
        "comparable_context": comparable_context.to_dict() if hasattr(comparable_context, "to_dict") else comparable_context,
        "target_p90": target_p90,
        "promotion_status": promotion_result,
    }
    if release_ready:
        save_atomic_joblib(legacy_artifact, MODEL_PATH)

    logger.info("Đã lưu trữ toàn diện Versioned Artifacts tại %s và Legacy Artifact tại %s.", models_dir, MODEL_PATH)
    return {
        "models_dir": models_dir,
        "reference_dir": reference_dir,
        "runs_dir": runs_dir,
        "legacy_model_path": MODEL_PATH,
    }
