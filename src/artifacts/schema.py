"""Mô-đun định nghĩa Schema cho Artifacts, Tiêu chí phê duyệt phân tầng (Development Gate vs Release Gate).

NGUYÊN TẮC GOVERNANCE ML THỰC CHIẾN:
1. DEVELOPMENT GATE (Đánh giá trên tập Validation):
   - Tuyển chọn ứng viên tốt nhất, kiểm tra mức độ cải thiện so với Naive/Segment Baseline.
   - Khi đạt chuẩn Development Gate, Champion Model được ĐÓNG BĂNG để chuẩn bị refit.
2. RELEASE GATE (Đánh giá trên tập Locked Test & Conformal Calibration):
   - Đánh giá năng lực tổng quát hóa trên dữ liệu hoàn toàn chưa nhìn thấy.
   - Kiểm tra WAPE Test và độ bao phủ Conformal Interval thực tế.
   - NGUYÊN TẮC BẤT DI BẤT DỊCH: Nếu Release Gate không đạt (research_only), TUYỆT ĐỐI KHÔNG
     được dùng kết quả Test để quay lại tinh chỉnh hyperparameter. Mọi thay đổi bắt buộc phải
     mở một experiment/phiên bản mới và tái huấn luyện từ đầu.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DevelopmentGateCriteria:
    """Tiêu chuẩn định lượng để phê duyệt Champion Model trên tập Validation."""

    min_baseline_improvement: float = (
        0.10  # MAE trên Validation phải tốt hơn Naive ít nhất 10%
    )
    max_validation_wape: float = 0.35  # WAPE trên Validation chấp nhận được


@dataclass
class ReleaseGateCriteria:
    """Tiêu chuẩn định lượng để phê duyệt mô hình lên môi trường phục vụ (Release Gate)."""

    max_test_wape: float = 0.30  # WAPE trên Test <= 30% để sẵn sàng sản xuất
    min_test_coverage: float = 0.75  # Độ bao phủ thực tế trên Test set >= 75%
    max_relative_interval_width: float = (
        1.50  # Bề rộng tương đối khoảng dự báo không vượt quá 150%
    )


@dataclass
class PromotionCriteria:
    """Cấu hình tiêu chí tương thích ngược tích hợp cả 2 cổng kiểm soát."""

    min_baseline_improvement: float = 0.10
    max_validation_wape: float = 0.30
    min_calibration_coverage: float = 0.75


def evaluate_development_gate(
    champion_val_mae: float,
    naive_val_mae: float,
    val_wape: float,
    criteria: DevelopmentGateCriteria | None = None,
) -> dict[str, Any]:
    """Đánh giá Development Gate trên tập Validation để quyết định đóng băng Champion."""
    if criteria is None:
        criteria = DevelopmentGateCriteria()

    improvement = (naive_val_mae - champion_val_mae) / max(naive_val_mae, 1.0)
    beats_baseline = bool(improvement >= criteria.min_baseline_improvement)

    # Hỗ trợ cả hai dạng biểu diễn: tỷ lệ (0.28) hoặc phần trăm (28.0%)
    wape_ratio = val_wape / 100.0 if val_wape > 1.0 else val_wape
    wape_acceptable = bool(wape_ratio <= criteria.max_validation_wape)
    champion_approved = beats_baseline and wape_acceptable

    reason = (
        f"Development Gate: Cải thiện so với baseline={improvement * 100:.2f}% "
        f"({'Đạt' if beats_baseline else 'Chưa đạt'}>={criteria.min_baseline_improvement * 100:.0f}%), "
        f"Val WAPE={wape_ratio * 100:.2f}% ({'Đạt' if wape_acceptable else 'Chưa đạt'})."
    )

    return {
        "gate_name": "development_gate",
        "evaluation_split": "validation",
        "champion_approved": champion_approved,
        "beats_baseline": beats_baseline,
        "baseline_improvement_percent": round(improvement * 100, 2),
        "val_wape_acceptable": wape_acceptable,
        "gate_reason": reason,
    }


def evaluate_release_gate(
    test_wape: float,
    test_coverage: float,
    relative_interval_width: float = 1.0,
    criteria: ReleaseGateCriteria | None = None,
) -> dict[str, Any]:
    """Đánh giá Release Gate trên tập Locked Test để xác định trạng thái phát hành (Production Readiness)."""
    if criteria is None:
        criteria = ReleaseGateCriteria()

    wape_ratio = test_wape / 100.0 if test_wape > 1.0 else test_wape
    wape_passed = bool(wape_ratio <= criteria.max_test_wape)
    coverage_passed = bool(test_coverage >= criteria.min_test_coverage)
    width_passed = bool(relative_interval_width <= criteria.max_relative_interval_width)

    production_ready = wape_passed and coverage_passed and width_passed
    readiness_status = "production_ready" if production_ready else "research_only"

    reason = (
        f"Release Gate: Trạng thái {readiness_status.upper()}. "
        f"Test WAPE={wape_ratio * 100:.1f}% ({'Đạt' if wape_passed else 'Chưa đạt'}<={criteria.max_test_wape * 100:.0f}%), "
        f"Coverage={test_coverage * 100:.1f}% ({'Đạt' if coverage_passed else 'Chưa đạt'}>={criteria.min_test_coverage * 100:.0f}%)."
    )

    governance_notice = (
        "QUY TẮC PHÁT HÀNH: Nếu Release Gate trả về 'research_only', nghiêm cấm tinh chỉnh mô hình "
        "dựa trên số liệu của tập Test. Bắt buộc giữ nguyên báo cáo kiểm thử và cải thiện thông qua "
        "bổ sung dữ liệu hoặc mở chu kỳ thử nghiệm độc lập mới."
    )

    return {
        "gate_name": "release_gate",
        "evaluation_split": "test",
        "production_ready": production_ready,
        "readiness_status": readiness_status,
        "test_wape_acceptable": wape_passed,
        "interval_coverage_acceptable": coverage_passed,
        "interval_width_acceptable": width_passed,
        "gate_reason": reason,
        "governance_notice": governance_notice,
    }


def evaluate_promotion(
    champion_val_mae: float,
    naive_val_mae: float,
    val_wape: float,
    actual_coverage: float,
    criteria: PromotionCriteria | None = None,
    test_wape: float | None = None,
    relative_interval_width: float = 1.0,
) -> dict[str, Any]:
    """Điều phối gate cho caller legacy mà không dùng nhầm Validation làm Test.

    ``test_wape`` là optional để giữ tương thích chữ ký cũ. Nếu caller cũ không
    cung cấp metric Locked Test, Release Gate sẽ fail-closed thay vì lấy lại
    ``val_wape`` như một đại diện sai lệch.
    """
    if criteria is None:
        criteria = PromotionCriteria()

    dev_criteria = DevelopmentGateCriteria(
        min_baseline_improvement=criteria.min_baseline_improvement,
        max_validation_wape=criteria.max_validation_wape,
    )
    rel_criteria = ReleaseGateCriteria(
        max_test_wape=criteria.max_validation_wape,
        min_test_coverage=criteria.min_calibration_coverage,
    )

    dev_result = evaluate_development_gate(
        champion_val_mae=champion_val_mae,
        naive_val_mae=naive_val_mae,
        val_wape=val_wape,
        criteria=dev_criteria,
    )

    rel_result = evaluate_release_gate(
        test_wape=float("inf") if test_wape is None else test_wape,
        test_coverage=actual_coverage,
        relative_interval_width=relative_interval_width,
        criteria=rel_criteria,
    )

    improvement = dev_result["baseline_improvement_percent"]
    beats_baseline = dev_result["beats_baseline"]
    wape_acceptable = rel_result["test_wape_acceptable"]
    interval_calibrated = rel_result["interval_coverage_acceptable"]
    is_production_ready = rel_result["production_ready"]

    promotion_status = {
        "model_selected": True,
        "beats_baseline": beats_baseline,
        "baseline_improvement_percent": improvement,
        "wape_acceptable": wape_acceptable,
        "interval_calibrated": interval_calibrated,
        "production_ready": is_production_ready,
    }

    return {
        "selection_status": "champion",
        "production_readiness": rel_result["readiness_status"],
        "promotion_status": promotion_status,
        "deployment_approved": is_production_ready,
        "promotion_reason": rel_result["gate_reason"],
        "development_gate": dev_result,
        "release_gate": rel_result,
    }
