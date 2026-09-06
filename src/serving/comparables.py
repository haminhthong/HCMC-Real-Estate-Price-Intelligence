"""Mô-đun tra cứu bất động sản tương đồng (Comparable Properties Engine)."""

from typing import Any
import numpy as np
import pandas as pd


def find_comparables(
    model_package: dict[str, Any],
    values: dict[str, Any],
    n_matches: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """Tìm 3-5 bất động sản tương đồng nhất từ tập dữ liệu tham chiếu lịch sử.

    Args:
        model_package: Gói mô hình (chứa danh sách `reference_listings`).
        values: Thuộc tính của bất động sản cần tra cứu.
        n_matches: Số lượng bất động sản tương đồng cần trích xuất.

    Returns:
        Tuple gồm danh sách các bất động sản tương đồng và thống kê trung vị (giá, đơn giá).
    """
    references = model_package.get("reference_listings", [])
    if not references:
        return [], {"median_price_million": None, "median_unit_price_million_m2": None}

    target_type = values.get("Property Type")
    target_area_name = values.get("location_area")
    target_area = (
        float(values.get("Area", 80.0))
        if pd.notna(values.get("Area")) and float(values.get("Area")) > 0
        else 80.0
    )
    target_beds = (
        float(values.get("Bedrooms", 3))
        if pd.notna(values.get("Bedrooms"))
        else 3.0
    )

    # 1. Lọc ứng viên: Ưu tiên cùng loại hình & cùng khu vực
    candidates = [
        r
        for r in references
        if r.get("property_type") == target_type
        and r.get("location_area") == target_area_name
    ]
    # Nếu không đủ ứng viên, mở rộng sang cùng loại hình trên toàn TP.HCM
    if len(candidates) < n_matches:
        candidates = [r for r in references if r.get("property_type") == target_type]
    if not candidates:
        candidates = references

    scored = []
    for c in candidates:
        c_area = float(c["area"]) if c.get("area") else target_area
        c_beds = float(c["bedrooms"]) if c.get("bedrooms") else target_beds
        # Khoảng cách hình học chuẩn hóa
        dist = abs(c_area - target_area) / max(target_area, 10.0) + 0.3 * abs(c_beds - target_beds)
        similarity = float(round(1.0 / (1.0 + dist), 2))
        scored.append((dist, similarity, c))

    scored.sort(key=lambda item: item[0])
    selected = scored[:n_matches]

    comparable_list = []
    prices = []
    unit_prices = []
    for _, sim, item in selected:
        record = dict(item)
        record["similarity_score"] = sim
        for key in ("bedrooms", "bathrooms", "floors", "area", "price_million", "unit_price_million_m2", "distance_to_cbd_km"):
            val = record.get(key)
            if val is not None:
                if pd.isna(val):
                    record[key] = None
                elif key in ("bedrooms", "bathrooms", "floors"):
                    record[key] = int(val)
                else:
                    record[key] = float(val)

        comparable_list.append(record)
        if record.get("price_million") is not None:
            prices.append(record["price_million"])
        if record.get("unit_price_million_m2") is not None:
            unit_prices.append(record["unit_price_million_m2"])

    summary = {
        "median_price_million": round(float(np.median(prices)), 1) if prices else None,
        "median_unit_price_million_m2": round(float(np.median(unit_prices)), 1) if unit_prices else None,
    }
    return comparable_list, summary
