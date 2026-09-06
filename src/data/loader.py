"""Mô-đun nạp dữ liệu thô (Data Ingestion) cho dự án."""

from pathlib import Path
import pandas as pd

from src.config import logger


def load_raw_dataset(path: Path | str) -> pd.DataFrame:
    """Nạp tệp dữ liệu thô từ đường dẫn đã chỉ định.

    Args:
        path: Đường dẫn tệp CSV hoặc Parquet.

    Returns:
        pd.DataFrame chứa toàn bộ dữ liệu thô chưa qua xử lý.

    Raises:
        FileNotFoundError: Nếu tệp không tồn tại.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.error("Không tìm thấy tệp dữ liệu thô tại: %s", file_path)
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {file_path}")

    logger.info("Đang nạp dữ liệu từ: %s", file_path)
    if file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    logger.info("Đã nạp thành công %d bản ghi và %d cột.", len(df), len(df.columns))
    return df
