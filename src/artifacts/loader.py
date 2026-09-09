"""Mô-đun tải mô hình có nhận biết phiên bản (Version-aware Artifact Loader)."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import ROOT_DIR


@lru_cache(maxsize=1)
def load_production_model(model_override_path: Path | str | None = None) -> dict[str, Any]:
    """Tải đúng release được production pointer chỉ định và fail-closed.

    File legacy chỉ còn dành cho override/test; production không được tự động
    chuyển sang một artifact cũ khi bundle hiện tại hỏng.
    """
    if model_override_path is not None:
        path = Path(model_override_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {path}")
        loaded = joblib.load(path)
        required_keys = {"pipeline", "version", "features"}
        if not isinstance(loaded, dict) or not required_keys.issubset(loaded):
            raise ValueError("Tệp mô hình không đúng cấu trúc hoặc đã bị hỏng.")
        return loaded

    prod_pointer_path = ROOT_DIR / "models" / "production.json"
    if not prod_pointer_path.exists():
        raise FileNotFoundError(
            "Chưa có production release. Hãy phát hành một bundle đạt Release Gate."
        )

    try:
        prod_info = json.loads(prod_pointer_path.read_text(encoding="utf-8"))
        active_version = prod_info.get("active_version")
        if not active_version:
            raise ValueError("production.json thiếu active_version.")

        v_dir = ROOT_DIR / "models" / str(active_version)
        ref_dir = ROOT_DIR / "reference" / str(active_version)
        required_files = [
            v_dir / "model.joblib",
            v_dir / "metadata.json",
            v_dir / "feature_context.json",
            v_dir / "calibration.json",
            v_dir / "manifest.json",
        ]
        missing = [str(path) for path in required_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Production bundle thiếu file: {missing}")

        manifest = json.loads((v_dir / "manifest.json").read_text(encoding="utf-8"))
        checksums = manifest.get("release", {}).get("checksums_sha256", {})
        for relative_path, expected_hash in checksums.items():
            artifact_path = ROOT_DIR / relative_path
            if not artifact_path.exists():
                raise FileNotFoundError(f"Artifact trong manifest không tồn tại: {relative_path}")
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError(f"Checksum artifact không khớp: {relative_path}")

        pipeline = joblib.load(v_dir / "model.joblib")
        metadata = json.loads((v_dir / "metadata.json").read_text(encoding="utf-8"))
        ctx = json.loads((v_dir / "feature_context.json").read_text(encoding="utf-8"))
        calib = json.loads((v_dir / "calibration.json").read_text(encoding="utf-8"))
        comparable_context = {}
        context_path = v_dir / "comparable_context.json"
        if context_path.exists():
            comparable_context = json.loads(context_path.read_text(encoding="utf-8"))

        ref_listings = []
        if (ref_dir / "comparables.parquet").exists():
            try:
                ref_df = pd.read_parquet(ref_dir / "comparables.parquet")
            except ImportError:
                # Parquet là artifact ưu tiên; CSV là representation hợp lệ
                # được writer tạo kèm để runtime không cần pyarrow.
                if not (ref_dir / "comparables.csv").exists():
                    raise
                ref_df = pd.read_csv(ref_dir / "comparables.csv")
        elif (ref_dir / "comparables.csv").exists():
            ref_df = pd.read_csv(ref_dir / "comparables.csv")
        else:
            raise FileNotFoundError("Production bundle thiếu comparables.parquet/csv.")
        ref_listings = ref_df.where(pd.notna(ref_df), None).to_dict(orient="records")

        segment_prices = {}
        for key, value in metadata.get("segment_unit_prices", {}).items():
            if " | " in key:
                property_type, location_area = key.split(" | ", 1)
                segment_prices[(property_type, location_area)] = value

        return {
            "pipeline": pipeline,
            "version": metadata.get("version", str(active_version)),
            "model_type": metadata.get("model_type", "ExtraTreesRegressor"),
            "target_formulation": metadata.get("target_formulation", "total_price"),
            "features": ctx.get("numeric_features", [])
            + ctx.get("categorical_features", [])
            + ctx.get("flag_features", [])
            + ctx.get("missing_indicator_features", []),
            "feature_context": ctx,
            "reference_date": ctx.get("reference_date"),
            "supported_areas": metadata.get("supported_areas", []),
            "supported_property_types": metadata.get("supported_property_types", []),
            "training_ranges": metadata.get("training_ranges", {}),
            "training_quantiles": metadata.get("training_quantiles", {}),
            "residual_log_quantile": calib.get("residual_log_quantile", 0.25),
            "target_coverage": calib.get("target_coverage", 0.8),
            "split_protocol": manifest.get("split_manifest", {}).get(
                "protocol", "strict_temporal_property_purged"
            ),
            "reference_listings": ref_listings,
            "comparable_context": comparable_context,
            "segment_unit_prices": segment_prices,
            "segment_sample_counts": comparable_context.get("segment_counts", {}),
            "promotion_status": metadata.get("promotion", {}),
            "model_status": metadata.get("promotion", {}).get(
                "production_readiness", "research_only"
            ),
        }
    except Exception as exc:
        raise RuntimeError(
            f"Production bundle {active_version!r} không hợp lệ; hệ thống fail-closed: {exc}"
        ) from exc


def clear_model_cache() -> None:
    """Xóa bộ nhớ đệm LRU của hàm load_production_model."""
    load_production_model.cache_clear()
