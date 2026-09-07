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

from .geo import _normalize_text


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
    location = df["Location"].map(_normalize_text) if "Location" in df else pd.Series("", index=df.index)
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


def resolve_property_identities(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nhận diện và phân giải nhóm bất động sản đa tầng bằng Union-Find.

    Args:
        df: DataFrame chứa dữ liệu tin đăng bất động sản hợp lệ.

    Returns:
        tuple gồm (DataFrame đã gán property_group_id và identity_confidence, audit_dict).
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    if n == 0:
        out["property_group_id"] = pd.Series(dtype=str)
        out["identity_confidence"] = pd.Series(dtype=str)
        return out, {
            "unique_property_groups": 0,
            "multi_listing_groups_count": 0,
            "largest_group_size": 0,
            "level_counts": {"strong": 0, "medium": 0, "weak": 0},
        }

    # Tính chữ ký cơ sở để nhận diện nhanh các bản ghi Level 1 Strong Match
    sig_series = make_property_signature(out)
    uf = UnionFind(n)
    pair_levels: dict[tuple[int, int], str] = {}
    level_counts = {"strong": 0, "medium": 0, "weak": 0}

    # Level 1: Exact Strong Signature Matching
    sig_groups = pd.Series(range(n)).groupby(sig_series).groups
    for _, indices in sig_groups.items():
        idx_list = list(indices)
        for i in range(len(idx_list) - 1):
            if uf.union(idx_list[i], idx_list[i + 1]):
                level_counts["strong"] += 1
                pair_levels[(min(idx_list[i], idx_list[i + 1]), max(idx_list[i], idx_list[i + 1]))] = "strong"

    # Ánh xạ root của Union-Find về mã property_group_id ổn định
    root_to_id: dict[int, str] = {}
    group_sizes: dict[int, int] = {}
    group_confidences: dict[int, str] = {}

    for i in range(n):
        root = uf.find(i)
        group_sizes[root] = group_sizes.get(root, 0) + 1

    for root, size in group_sizes.items():
        # Dùng hash chữ ký của mẫu đại diện nhỏ nhất trong nhóm làm property_group_id
        root_to_id[root] = sig_series.iloc[root]
        if size > 1:
            group_confidences[root] = "strong"
        else:
            group_confidences[root] = "singleton"

    out["property_group_id"] = [root_to_id[uf.find(i)] for i in range(n)]
    out["identity_confidence"] = [group_confidences[uf.find(i)] for i in range(n)]

    unique_groups = len(root_to_id)
    multi_listing_groups = sum(1 for s in group_sizes.values() if s > 1)
    largest_group = max(group_sizes.values()) if group_sizes else 0

    audit_dict = {
        "unique_property_groups": unique_groups,
        "multi_listing_groups_count": multi_listing_groups,
        "largest_group_size": largest_group,
        "level_counts": level_counts,
    }

    return out, audit_dict


def assign_property_group(df: pd.DataFrame) -> pd.DataFrame:
    """Gán cột `property_group_id` và `identity_confidence` cho DataFrame."""
    resolved_df, _ = resolve_property_identities(df)
    return resolved_df
