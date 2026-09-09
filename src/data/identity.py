"""Mô-đun định danh bất động sản vật lý (Property Identity Resolution) và nhóm tin đăng.

Ứng dụng kiến trúc phân tầng đa cấp (Multi-level Identity Resolution) kết hợp cấu trúc
dữ liệu Union-Find (Disjoint-Set) để nhận diện chính xác các tin đăng cùng một căn nhà:
- Level 1 (Strong Match): Cùng địa chỉ chuẩn hóa, cùng loại hình, diện tích khớp chuẩn xác và GPS/kết cấu tương thích.
- Level 2 (Medium Match): Cùng khu vực hành chính (quận/phường/đường), cùng loại hình, diện tích lệch <= 3%, số phòng/tầng tương thích.
- Level 3 (Weak Match): Cùng phân đoạn khu vực, tương đồng kết cấu cao.

Gán mã định danh duy nhất (property_group_id) và mức độ tin cậy định danh (identity_confidence),
phục vụ cơ chế chia tập Group-isolated Temporal Split chống rò rỉ dữ liệu (Data Leakage).
"""

from typing import Any

import numpy as np
import pandas as pd

from .geo import _normalize_text, extract_area


class UnionFind:
    """Cấu trúc dữ liệu Disjoint-Set Union với Path Compression và Union by Rank."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return False
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            self.parent[root_j] = root_i
            self.rank[root_i] += 1
        return True


def make_property_signature(df: pd.DataFrame) -> pd.Series:
    """Tạo chữ ký bất động sản vật lý độc lập với tin đăng, giá, ngày đăng và môi giới.

    Duy trì chữ ký băm 64-bit chuẩn hóa cho mục đích kiểm thử tính bất biến (invariance).
    """
    location = (
        df["Location"].map(_normalize_text)
        if "Location" in df
        else pd.Series("", index=df.index)
    )
    property_type = (
        df["Property Type"].map(_normalize_text)
        if "Property Type" in df
        else pd.Series("", index=df.index)
    )

    area = pd.to_numeric(df.get("Area"), errors="coerce").round(0)
    width = pd.to_numeric(df.get("Width"), errors="coerce").round(1)
    length = pd.to_numeric(df.get("Length"), errors="coerce").round(1)
    latitude = pd.to_numeric(df.get("Latitude"), errors="coerce").round(4)
    longitude = pd.to_numeric(df.get("Longitude"), errors="coerce").round(4)
    bedrooms = pd.to_numeric(df.get("Bedrooms"), errors="coerce")
    bathrooms = pd.to_numeric(df.get("Bathrooms"), errors="coerce")

    signature = pd.DataFrame(
        {
            "location": location,
            "property_type": property_type,
            "area": area,
            "width": width,
            "length": length,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "latitude": latitude,
            "longitude": longitude,
        }
    )

    return pd.util.hash_pandas_object(
        signature.fillna("missing"),
        index=False,
    ).astype(str)


def _identity_value(row: pd.Series, column: str, default: Any = np.nan) -> Any:
    """Đọc một thuộc tính identity mà không lỗi khi cột bị thiếu."""
    return row.get(column, default)


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def _compatible_structure(left: pd.Series, right: pd.Series) -> bool:
    """Chỉ từ chối khi hai tin có cấu trúc đã biết mâu thuẫn."""
    for column in ("Bedrooms", "Bathrooms", "Floors"):
        left_value = pd.to_numeric(_identity_value(left, column), errors="coerce")
        right_value = pd.to_numeric(_identity_value(right, column), errors="coerce")
        if (
            pd.notna(left_value)
            and pd.notna(right_value)
            and abs(float(left_value) - float(right_value)) > 1
        ):
            return False

    for column, tolerance in (("Width", 0.08), ("Length", 0.08)):
        left_value = pd.to_numeric(_identity_value(left, column), errors="coerce")
        right_value = pd.to_numeric(_identity_value(right, column), errors="coerce")
        if (
            pd.notna(left_value)
            and pd.notna(right_value)
            and _relative_difference(float(left_value), float(right_value)) > tolerance
        ):
            return False
    return True


def _gps_distance_m(left: pd.Series, right: pd.Series) -> float | None:
    """Tính khoảng cách GPS để chặn merge nhầm hai địa chỉ trùng tên."""
    values = [
        _identity_value(left, "Latitude"),
        _identity_value(left, "Longitude"),
        _identity_value(right, "Latitude"),
        _identity_value(right, "Longitude"),
    ]
    if any(pd.isna(value) for value in values):
        return None
    radius_m = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(float, values)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lam = np.radians(lon2 - lon1)
    haversine_a = (
        np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2) ** 2
    )
    return float(radius_m * 2 * np.arcsin(np.sqrt(np.clip(haversine_a, 0.0, 1.0))))


def _listing_key(row: pd.Series) -> tuple[str, str] | None:
    """Khóa listing cấp nguồn; cùng khóa là cùng listing event."""
    source = row.get("source_id", row.get("Source ID", "default"))
    listing_id = row.get("source_listing_id", row.get("Listing ID"))
    if pd.isna(listing_id) or str(listing_id).strip() == "":
        return None
    return _normalize_text(source) or "default", str(listing_id).strip()


def _location_components(row: pd.Series) -> tuple[str, str, str]:
    """Lấy area/ward/street bảo thủ từ địa chỉ thô, không tự bịa tọa độ."""
    location = _normalize_text(row.get("Location", ""))
    area = _normalize_text(row.get("location_area", "")) or _normalize_text(
        extract_area(location)
    )
    chunks = [chunk.strip() for chunk in location.split(",") if chunk.strip()]
    ward = next((chunk for chunk in chunks if "phường" in chunk or "xã" in chunk), "")
    street = next((chunk for chunk in chunks if "đường" in chunk or "hẻm" in chunk), "")
    return area, ward, street


def _match_level(left: pd.Series, right: pd.Series) -> str | None:
    """Phân loại cặp ứng viên theo Strong/Medium/Weak."""
    left_type = _normalize_text(left.get("Property Type", ""))
    right_type = _normalize_text(right.get("Property Type", ""))
    if not left_type or left_type != right_type:
        return None

    left_location = _normalize_text(left.get("Location", ""))
    right_location = _normalize_text(right.get("Location", ""))
    left_area = pd.to_numeric(left.get("Area"), errors="coerce")
    right_area = pd.to_numeric(right.get("Area"), errors="coerce")
    if (
        not left_location
        or not right_location
        or pd.isna(left_area)
        or pd.isna(right_area)
    ):
        return None

    area_delta = _relative_difference(float(left_area), float(right_area))
    gps_distance = _gps_distance_m(left, right)
    structure_ok = _compatible_structure(left, right)

    # Strong: địa chỉ chuẩn hóa + loại hình + diện tích gần, không mâu thuẫn.
    if (
        left_location == right_location
        and area_delta <= 0.03
        and structure_ok
        and (gps_distance is None or gps_distance <= 50.0)
    ):
        return "strong"

    left_area_key, left_ward, left_street = _location_components(left)
    right_area_key, right_ward, right_street = _location_components(right)
    same_locality = (
        left_area_key
        and left_area_key == right_area_key
        and left_ward
        and left_ward == right_ward
        and (not left_street or not right_street or left_street == right_street)
    )
    # Medium chỉ union khi không có GPS và có thêm ward/street tương thích.
    if gps_distance is None and same_locality and area_delta <= 0.05 and structure_ok:
        return "medium"

    # Weak chỉ được audit, tuyệt đối không union.
    if left_area_key and left_area_key == right_area_key and area_delta <= 0.10:
        return "weak"
    return None


def resolve_property_identities(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve identity bảo thủ và ghi nhận weak duplicate để audit.

    ``property_group_id`` là identity của property vật lý; ngày/giá vẫn được
    giữ ở từng listing event và không bị gộp thành một quan sát duy nhất.
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    level_counts = {"strong": 0, "medium": 0, "weak": 0}
    audit_base = {
        "unique_property_groups": 0,
        "multi_listing_groups_count": 0,
        "largest_group_size": 0,
        "level_counts": level_counts,
        "possible_duplicate_pairs": 0,
    }
    if n == 0:
        out["property_group_id"] = pd.Series(dtype=str)
        out["identity_confidence"] = pd.Series(dtype=str)
        out["possible_duplicate"] = pd.Series(dtype=bool)
        return out, audit_base

    union_find = UnionFind(n)
    union_levels: dict[tuple[int, int], str] = {}
    weak_pairs: set[tuple[int, int]] = set()

    # Block theo loại hình, khu vực và bucket diện tích; mở rộng thêm bucket kế bên.
    blocks: dict[tuple[str, str, int], list[int]] = {}
    for index, row in out.iterrows():
        property_type = _normalize_text(row.get("Property Type", ""))
        area_name = _normalize_text(row.get("location_area", "")) or _normalize_text(
            extract_area(row.get("Location", ""))
        )
        area = pd.to_numeric(row.get("Area"), errors="coerce")
        bucket = int(float(area) // 10) if pd.notna(area) else -1
        blocks.setdefault((property_type, area_name, bucket), []).append(index)

    expanded_blocks: dict[tuple[str, str, int], list[int]] = {}
    for (property_type, area_name, bucket), indices in blocks.items():
        for neighbor_bucket in (bucket - 1, bucket, bucket + 1):
            expanded_blocks.setdefault(
                (property_type, area_name, neighbor_bucket), []
            ).extend(indices)

    # Level A: cùng source + source listing id.
    listing_keys: dict[tuple[str, str], int] = {}
    for index, row in out.iterrows():
        key = _listing_key(row)
        if key is None:
            continue
        if key in listing_keys:
            pair = (listing_keys[key], index)
            if union_find.union(*pair):
                level_counts["strong"] += 1
                union_levels[pair] = "strong"
        else:
            listing_keys[key] = index

    checked_pairs: set[tuple[int, int]] = set()
    for indices in expanded_blocks.values():
        for position, left_index in enumerate(indices):
            for right_index in indices[position + 1 :]:
                pair = (min(left_index, right_index), max(left_index, right_index))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                level = _match_level(out.iloc[pair[0]], out.iloc[pair[1]])
                if level is None:
                    continue
                level_counts[level] += 1
                if level == "weak":
                    weak_pairs.add(pair)
                    continue
                if union_find.union(*pair):
                    union_levels[pair] = level

    root_members: dict[int, list[int]] = {}
    for index in range(n):
        root_members.setdefault(union_find.find(index), []).append(index)

    confidence_rank = {"singleton": 0, "weak": 1, "medium": 2, "strong": 3}
    root_to_id: dict[int, str] = {}
    root_confidence: dict[int, str] = {}
    weak_rows = {index for pair in weak_pairs for index in pair}
    for root, members in root_members.items():
        root_to_id[root] = make_property_signature(out.iloc[[members[0]]]).iloc[0]
        confidence = "strong" if len(members) > 1 else "singleton"
        for pair, level in union_levels.items():
            if (
                pair[0] in members
                and pair[1] in members
                and confidence_rank[level] > confidence_rank[confidence]
            ):
                confidence = level
        root_confidence[root] = confidence

    out["property_group_id"] = [
        root_to_id[union_find.find(index)] for index in range(n)
    ]
    out["identity_confidence"] = [
        "weak"
        if index in weak_rows and root_confidence[union_find.find(index)] == "singleton"
        else root_confidence[union_find.find(index)]
        for index in range(n)
    ]
    out["possible_duplicate"] = [index in weak_rows for index in range(n)]

    group_sizes = [len(members) for members in root_members.values()]
    audit_base.update(
        {
            "unique_property_groups": len(root_members),
            "multi_listing_groups_count": sum(size > 1 for size in group_sizes),
            "largest_group_size": max(group_sizes, default=0),
            "possible_duplicate_pairs": len(weak_pairs),
        }
    )
    return out, audit_base


def assign_property_group(df: pd.DataFrame) -> pd.DataFrame:
    """Gán cột `property_group_id` và `identity_confidence` cho DataFrame."""
    resolved_df, _ = resolve_property_identities(df)
    return resolved_df
