"""Động cơ tra cứu bất động sản tương đồng đa chiều (Comparable Properties Engine).

Tra cứu bằng quy tắc thủ công các tin đăng có đặc điểm gần với đầu vào:
1. Lọc theo loại hình, khu vực, diện tích và thời điểm định giá.
2. Xếp hạng bằng khoảng cách diện tích, GPS, kết cấu, CBD và recency.
3. Trả về các tin gần nhất làm thông tin tham chiếu cho dự báo.

Các trọng số trong prototype được cấu hình thủ công; chúng không phải hệ số
thẩm định được học hoặc được chứng minh bởi một chuẩn định giá bên ngoài.
"""

from typing import Any

import numpy as np
import pandas as pd

from .context import ComparableContext


def _finite_float(value: Any, fallback: float | None = None) -> float | None:
    """Đọc số thực hữu hạn; coi NaN/Inf và giá trị lỗi là dữ liệu thiếu."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return fallback
    return numeric_value if np.isfinite(numeric_value) else fallback


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Tính khoảng cách Haversine (km) giữa 2 tọa độ GPS."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(r * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def find_comparables(
    model_package: dict[str, Any],
    values: dict[str, Any],
    n_matches: int = 4,
    context: ComparableContext | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """Tìm 3-5 bất động sản tương đồng nhất từ tập dữ liệu tham chiếu lịch sử.

    Args:
        model_package: Gói mô hình (chứa danh sách `reference_listings`).
        values: Thuộc tính của bất động sản cần tra cứu.
        n_matches: Số lượng bất động sản tương đồng cần trích xuất.
        context: Tham số chuẩn hóa (nếu None sẽ lấy từ model_package).

    Returns:
        Tuple gồm danh sách các bất động sản tương đồng và thống kê trung vị (giá, đơn giá).
    """
    references = model_package.get("reference_listings", [])
    if not references:
        return [], {"median_price_million": None, "median_unit_price_million_m2": None}

    if context is None:
        comp_ctx_data = model_package.get("comparable_context")
        if comp_ctx_data:
            context = ComparableContext.from_dict(comp_ctx_data)
        else:
            context = ComparableContext()

    target_type = values.get("Property Type")
    target_area_name = values.get("location_area")
    target_group_id = values.get("property_group_id")
    valuation_date = pd.to_datetime(
        values.get(
            "as_of_date", values.get("valuation_date", values.get("listing_date"))
        ),
        errors="coerce",
        utc=True,
    )

    target_area = _finite_float(values.get("Area"), context.median_area)
    target_area = (
        target_area if target_area and target_area > 0 else context.median_area
    )
    target_beds = _finite_float(values.get("Bedrooms"), 3.0) or 3.0
    target_baths = _finite_float(values.get("Bathrooms"), 2.0) or 2.0
    target_lat = _finite_float(values.get("Latitude"))
    target_lon = _finite_float(values.get("Longitude"))
    target_cbd = _finite_float(values.get("distance_to_cbd_km"))

    def is_dated_past_listing(reference: dict[str, Any]) -> bool:
        """Chặn future listing khi query có mốc thời gian định giá.

        Luồng serving luôn tự bổ sung ``as_of_date``. Chỉ các caller legacy
        không truyền mốc thời gian mới được giữ fixture cũ không có ngày.
        """
        reference_date = pd.to_datetime(
            reference.get("listing_date"), errors="coerce", utc=True
        )
        if pd.isna(valuation_date):
            return True
        return bool(pd.notna(reference_date) and reference_date <= valuation_date)

    # Lọc tin tương lai và tin thiếu ngày khi truy vấn có mốc định giá.
    dated_candidates = [
        r
        for r in references
        if r.get("property_type") == target_type
        and (target_group_id is None or r.get("property_group_id") != target_group_id)
        and is_dated_past_listing(r)
    ]

    if pd.notna(valuation_date):
        dated_candidates = [
            r
            for r in dated_candidates
            if (
                valuation_date
                - pd.to_datetime(r.get("listing_date"), errors="coerce", utc=True)
            ).days
            <= 365
        ]

    area_candidates = [
        r
        for r in dated_candidates
        if _finite_float(r.get("area")) is None
        or abs(_finite_float(r.get("area")) - target_area) / max(target_area, 1.0)
        <= 0.25
    ]
    local_candidates = [
        r for r in area_candidates if r.get("location_area") == target_area_name
    ]
    candidates = (
        local_candidates if len(local_candidates) >= n_matches else area_candidates
    )

    # Nếu không đủ ứng viên cùng quận, mở rộng sang cùng loại hình trên toàn TP.HCM
    if len(candidates) < n_matches:
        candidates = [
            r
            for r in dated_candidates
            if r.get("property_type") == target_type
            and (
                target_group_id is None or r.get("property_group_id") != target_group_id
            )
        ]

    # Không lấy tin tương lai hoặc chính bất động sản đang định giá để bù số lượng.
    if not candidates:
        return [], {"median_price_million": None, "median_unit_price_million_m2": None}

    scored = []
    for c in candidates:
        c_area_value = _finite_float(c.get("area"))
        c_beds_value = _finite_float(c.get("bedrooms"))
        c_baths_value = _finite_float(c.get("bathrooms"))
        c_area = c_area_value if c_area_value and c_area_value > 0 else target_area
        c_beds = c_beds_value if c_beds_value is not None else target_beds
        c_baths = c_baths_value if c_baths_value is not None else target_baths
        c_lat = _finite_float(c.get("latitude"))
        c_lon = _finite_float(c.get("longitude"))
        c_cbd = _finite_float(c.get("distance_to_cbd_km"))

        # a) Khoảng cách diện tích chuẩn hóa
        area_dist = min(abs(c_area - target_area) / max(target_area, 10.0), 2.0)

        # b) Khoảng cách địa lý thực địa (Haversine GPS khi cả 2 có tọa độ)
        if (
            target_lat is not None
            and target_lon is not None
            and c_lat is not None
            and c_lon is not None
        ):
            raw_km = haversine_km(target_lat, target_lon, c_lat, c_lon)
            geo_dist = min(raw_km / 5.0, 2.0)
        else:
            # Khi thiếu GPS, dùng mức phạt 0.15 cho cùng quận và 0.8 cho khác quận.
            geo_dist = 0.15 if c.get("location_area") == target_area_name else 0.8

        # c) Khoảng cách kết cấu phòng ngủ & phòng vệ sinh
        bed_dist = min(abs(c_beds - target_beds) / 3.0, 2.0)
        bath_dist = min(abs(c_baths - target_baths) / 3.0, 2.0)

        # d) Khoảng cách tới CBD
        if target_cbd is not None and c_cbd is not None:
            cbd_dist = min(abs(c_cbd - target_cbd) / max(context.cbd_scale, 5.0), 2.0)
        else:
            cbd_dist = 0.2

        # e) Khoảng cách độ mới tin đăng (Recency)
        target_date_val = valuation_date
        c_date_val = c.get("listing_date")
        if (
            target_date_val is not None
            and c_date_val is not None
            and pd.notna(target_date_val)
            and pd.notna(c_date_val)
            and str(c_date_val).strip() != ""
        ):
            try:
                t_dt = pd.to_datetime(target_date_val, utc=True)
                c_dt = pd.to_datetime(c_date_val, utc=True)
                days_diff = max((t_dt - c_dt).days, 0)
                recency_dist = min(days_diff / 180.0, 2.0)
            except (TypeError, ValueError):
                recency_dist = 0.2
        else:
            recency_dist = 0.2

        # Trọng số thủ công của bản thử nghiệm, chưa được kiểm định cho thẩm định giá.
        dist = (
            0.30 * area_dist
            + 0.25 * geo_dist
            + 0.15 * bed_dist
            + 0.10 * bath_dist
            + 0.10 * cbd_dist
            + 0.10 * recency_dist
        )

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
        for key in (
            "bedrooms",
            "bathrooms",
            "floors",
            "area",
            "price_million",
            "unit_price_million_m2",
            "distance_to_cbd_km",
        ):
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
        "median_unit_price_million_m2": round(float(np.median(unit_prices)), 1)
        if unit_prices
        else None,
    }
    return comparable_list, summary
