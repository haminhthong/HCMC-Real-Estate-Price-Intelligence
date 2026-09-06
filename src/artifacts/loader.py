"""Mô-đun tải mô hình có nhận biết phiên bản (Version-aware Artifact Loader)."""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any
import joblib
import pandas as pd

from src.config import MODEL_PATH, ROOT_DIR, logger


@lru_cache(maxsize=1)
def load_production_model(model_override_path: Path | str | None = None) -> dict[str, Any]:
    """Tải gói mô hình phục vụ từ active version được khai báo trong production.json.

    Nếu có chỉ định đường dẫn cụ thể hoặc không có production.json, sẽ tự động fallback về
    file legacy `models/price_model.joblib`.
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
    if prod_pointer_path.exists():
        try:
            prod_info = json.loads(prod_pointer_path.read_text(encoding="utf-8"))
            active_version = prod_info.get("active_version")
            if active_version:
                v_dir = ROOT_DIR / "models" / active_version
                ref_dir = ROOT_DIR / "reference" / active_version

                if (v_dir / "model.joblib").exists():
                    pipeline = joblib.load(v_dir / "model.joblib")
                    metadata = json.loads((v_dir / "metadata.json").read_text(encoding="utf-8")) if (v_dir / "metadata.json").exists() else {}
                    ctx = json.loads((v_dir / "feature_context.json").read_text(encoding="utf-8")) if (v_dir / "feature_context.json").exists() else {}
                    calib = json.loads((v_dir / "calibration.json").read_text(encoding="utf-8")) if (v_dir / "calibration.json").exists() else {}

                    # Đọc dữ liệu tham chiếu decoupled comparables
                    ref_listings = []
                    if (ref_dir / "comparables.parquet").exists():
                        ref_df = pd.read_parquet(ref_dir / "comparables.parquet")
                        ref_df = ref_df.where(pd.notna(ref_df), None)
                        ref_listings = ref_df.to_dict(orient="records")
                    elif (ref_dir / "comparables.csv").exists():
                        ref_df = pd.read_csv(ref_dir / "comparables.csv")
                        ref_df = ref_df.where(pd.notna(ref_df), None)
                        ref_listings = ref_df.to_dict(orient="records")

                    package = {
                        "pipeline": pipeline,
                        "version": metadata.get("version", "1.2.0"),
                        "model_type": metadata.get("model_type", "ExtraTreesRegressor"),
                        "target_formulation": metadata.get("target_formulation", "total_price"),
                        "features": ctx.get("numeric_features", []) + ctx.get("categorical_features", []) + ctx.get("flag_features", []),
                        "reference_date": ctx.get("reference_date"),
                        "supported_areas": metadata.get("supported_areas", []),
                        "supported_property_types": metadata.get("supported_property_types", []),
                        "training_ranges": metadata.get("training_ranges", {}),
                        "training_quantiles": metadata.get("training_quantiles", {}),
                        "residual_log_quantile": calib.get("residual_log_quantile", 0.25),
                        "target_coverage": calib.get("target_coverage", 0.8),
                        "split_protocol": "grouped temporal split 60/15/10/15 with 2-phase champion refit",
                        "reference_listings": ref_listings,
                        "promotion_status": metadata.get("promotion", {}),
                    }
                    return package
        except Exception as exc:
            logger.warning("Lỗi đọc versioned artifact từ %s: %s. Fallback về legacy file.", prod_pointer_path, exc)

    # Fallback về legacy file joblib
    if not MODEL_PATH.exists():
        logger.error("Không tìm thấy file mô hình tại đường dẫn: %s", MODEL_PATH)
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy: python -m src.pipeline train")

    model_package: dict[str, Any] = joblib.load(MODEL_PATH)
    return model_package


def clear_model_cache() -> None:
    """Xóa bộ nhớ đệm LRU của hàm load_production_model."""
    load_production_model.cache_clear()
