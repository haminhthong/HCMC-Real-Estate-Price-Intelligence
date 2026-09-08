"""[DEPRECATED LEGACY WRAPPER] Mô-đun tương thích ngược cho huấn luyện mô hình.

LƯU Ý KIẾN TRÚC:
Toàn bộ quy trình huấn luyện và điều phối pipeline chính thức được quản lý bởi `src.pipeline`.
Tệp này chỉ được duy trì làm proxy chuyển tiếp cho các lệnh `python -m src.train` cũ.
"""

from pathlib import Path
from typing import Any

from src.calibration.conformal import conformal_quantile
from src.config import DATA_PATH, MODEL_VERSION
from src.data.split import split_group_indices
from src.evaluation.metrics import regression_metrics
from src.modeling.pipelines import build_pipeline
from src.pipeline import run_pipeline


def train(data_path: Path | str = DATA_PATH) -> dict[str, Any]:
    """Chạy quy trình huấn luyện và trả về từ điển metrics."""
    result = run_pipeline(data_path=data_path, model_version=MODEL_VERSION)
    return result["champion_metrics"]


__all__ = [
    "build_pipeline",
    "conformal_quantile",
    "regression_metrics",
    "split_group_indices",
    "train",
]


if __name__ == "__main__":
    train()
