"""Mô-đun tương thích ngược cho kiểm tra cảnh báo độ tin cậy và rào chắn miền.

Ủy nhiệm toàn bộ chức năng kiểm soát sang gói chuẩn hóa `src.reliability.guards`.
"""

from typing import Any
import pandas as pd

from src.reliability.guards import check_reliability_guards


def check_ood_guards(
    feature_frame: pd.DataFrame,
    values: dict[str, Any],
    model_package: dict[str, Any],
    predicted_price: float,
    lower_bound: float,
    upper_bound: float,
    data_quality_score: float,
    as_of_date: Any = None,
) -> tuple[list[str], str, str, str]:
    """Hàm ủy nhiệm tương thích ngược kiểm tra các rào chắn độ tin cậy."""
    return check_reliability_guards(
        feature_frame=feature_frame,
        values=values,
        model_package=model_package,
        predicted_price=predicted_price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        input_completeness_score=data_quality_score,
        as_of_date=as_of_date,
    )


__all__ = ["check_ood_guards", "check_reliability_guards"]
