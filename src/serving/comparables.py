"""Mô-đun tương thích ngược cho tra cứu bất động sản tương đồng.

Ủy nhiệm toàn bộ chức năng sang gói chuyên biệt `src.comparables`.
"""

from typing import Any
from src.comparables import ComparableContext, find_comparables as _find_comparables


def find_comparables(
    model_package: dict[str, Any],
    values: dict[str, Any],
    n_matches: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """Hàm ủy nhiệm tương thích ngược tìm kiếm bất động sản tương đồng."""
    return _find_comparables(
        model_package=model_package,
        values=values,
        n_matches=n_matches,
    )


__all__ = ["ComparableContext", "find_comparables"]
