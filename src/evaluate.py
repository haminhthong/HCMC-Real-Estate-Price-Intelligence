"""In báo cáo test đã lưu trong ``reports/metrics.json``."""

import json
import sys

from src.config import METRICS_PATH, logger
from src.evaluation.report import format_evaluation_summary


def main() -> None:
    """Đọc metrics hiện hành và in tóm tắt generalization performance."""
    if not METRICS_PATH.exists():
        raise SystemExit("Chưa có báo cáo. Hãy chạy: python -m src.pipeline train")

    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("reports/metrics.json không hợp lệ: %s", exc)
        raise SystemExit("Không đọc được reports/metrics.json") from exc

    summary_text = format_evaluation_summary(
        {
            "test_sample_count": metrics.get("test_rows", "N/A"),
            "champion_metrics": metrics.get("test", {}),
            "interval_metrics": metrics.get("interval", {}),
            "baselines": metrics.get("baselines", {}),
        }
    )
    try:
        print(summary_text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(summary_text.encode("utf-8"))
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
