"""Mô-đun định nghĩa Schema cho Artifacts, Tiêu chí phê duyệt (Promotion Criteria) và Readiness Gate."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PromotionCriteria:
    """Tiêu chuẩn định lượng để phê duyệt mô hình lên môi trường phục vụ (Production Gate)."""

    min_baseline_improvement: float = 0.10  # MAE trên Validation phải tốt hơn Naive ít nhất 10%
    max_validation_wape: float = 0.30       # Sai số tỷ trọng tuyệt đối WAPE <= 30%
    min_calibration_coverage: float = 0.75  # Mức độ bao phủ thực tế trên Calibration/Test >= 75%


def evaluate_promotion(
    champion_val_mae: float,
    naive_val_mae: float,
    val_wape: float,
    actual_coverage: float,
    criteria: PromotionCriteria | None = None,
) -> dict[str, Any]:
    """Đánh giá trạng thái sẵn sàng triển khai (Production Readiness & Promotion Status).

    Thay thế cho cờ `deployment_approved = true` ngây thơ trước đây bằng một bộ
    quy tắc phân tầng chính xác và phản ánh đúng thực tế của bài toán ML bất động sản.
    """
    if criteria is None:
        criteria = PromotionCriteria()

    improvement = (naive_val_mae - champion_val_mae) / max(naive_val_mae, 1.0)
    beats_baseline = bool(improvement >= criteria.min_baseline_improvement)
    wape_acceptable = bool(val_wape <= criteria.max_validation_wape)
    interval_calibrated = bool(actual_coverage >= criteria.min_calibration_coverage)

    is_production_ready = beats_baseline and wape_acceptable and interval_calibrated

    selection_status = "champion"
    production_readiness = "production_ready" if is_production_ready else "research_only"

    promotion_status = {
        "model_selected": True,
        "beats_baseline": beats_baseline,
        "baseline_improvement_percent": round(improvement * 100, 2),
        "wape_acceptable": wape_acceptable,
        "interval_calibrated": interval_calibrated,
        "production_ready": is_production_ready,
    }

    reason = (
        f"Mô hình đạt danh hiệu Champion (vượt baseline {improvement * 100:.1f}%). "
        f"Trạng thái phục vụ: {production_readiness.upper()} "
        f"(WAPE={val_wape:.1f}% {'<= 30%' if wape_acceptable else '> 30%'}, "
        f"Coverage={actual_coverage * 100:.1f}% {'>= 75%' if interval_calibrated else '< 75%'})."
    )

    return {
        "selection_status": selection_status,
        "production_readiness": production_readiness,
        "promotion_status": promotion_status,
        "deployment_approved": is_production_ready,
        "promotion_reason": reason,
    }
