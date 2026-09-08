"""Mô-đun hiển thị báo cáo đánh giá mô hình định giá bất động sản.

Tệp này đọc các artifact metrics đã lưu trong quá trình huấn luyện và hiển thị
báo cáo chuẩn hóa qua màn hình console.
"""

import json
import sys

from src.config import (
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    logger,
)
from src.evaluation.report import format_evaluation_summary


def main() -> None:
    """Đọc và in báo cáo kết quả đánh giá đã lưu trong lượt huấn luyện gần nhất."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        logger.error("Chưa tìm thấy file mô hình hoặc file metrics.")
        raise SystemExit("Chưa có kết quả đánh giá. Hãy chạy huấn luyện trước: python -m src.pipeline train")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    comparison = json.loads(MODEL_COMPARISON_PATH.read_text(encoding="utf-8")) if MODEL_COMPARISON_PATH.exists() else {}
    test_report = comparison.get("test_report_only", {})

    summary_text = format_evaluation_summary(
        {
            "test_sample_count": metrics.get("test_rows", "N/A"),
            "champion_metrics": {
                "mae_million": metrics.get("mae_million", 0),
                "wape_percent": metrics.get("wape_percent", 0),
                "r2": metrics.get("r2", 0),
            },
            "interval_metrics": {
                "target_coverage": metrics.get("prediction_interval_target_coverage", 0.8),
                "actual_coverage": metrics.get("prediction_interval_test_coverage", 0),
                "coverage_gap": metrics.get("coverage_gap", 0),
                "mean_interval_width_million": metrics.get("mean_interval_width_million", 0),
                "relative_interval_width": metrics.get("relative_interval_width", 0),
            },
            "baselines": {
                "naive_median": test_report.get("naive_median", {}),
                "district_property_segment_median": test_report.get("district_property_segment_median", {}),
            },
        }
    )
    try:
        print(summary_text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(summary_text.encode("utf-8"))
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
