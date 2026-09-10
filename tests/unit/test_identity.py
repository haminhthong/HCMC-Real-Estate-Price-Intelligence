"""Kiểm thử đơn vị cho mô-đun Identity và Chữ ký bất động sản."""

import pandas as pd

from src.data.identity import assign_property_group, make_property_signature


def test_property_signature_invariance():
    """Chữ ký bất động sản không thay đổi khi giá hoặc ngày đăng thay đổi."""
    df1 = pd.DataFrame(
        [
            {
                "Location": "Quận 1, TP.HCM",
                "Property Type": "Nhà riêng",
                "Area": 80.0,
                "Width": 4.0,
                "Length": 20.0,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "Latitude": 10.7769,
                "Longitude": 106.7009,
                "Price": 8500.0,
                "listing_date": "2024-01-01",
            }
        ]
    )
    df2 = df1.copy()
    df2["Price"] = 9200.0
    df2["listing_date"] = "2025-06-01"

    sig1 = make_property_signature(df1).iloc[0]
    sig2 = make_property_signature(df2).iloc[0]
    assert sig1 == sig2, "Chữ ký phải độc lập với biến thiên giá và ngày đăng!"


def test_assign_property_group_adds_column():
    df = pd.DataFrame(
        [
            {
                "Location": "Quận 7, TP.HCM",
                "Property Type": "Căn hộ chung cư",
                "Area": 70.0,
            }
        ]
    )
    out = assign_property_group(df)
    assert "property_group_id" in out.columns
    assert "identity_confidence" in out.columns
    assert isinstance(out["property_group_id"].iloc[0], str)
    assert out["identity_confidence"].iloc[0] == "singleton"


def test_resolve_property_identities_multi_level():
    from src.data.identity import resolve_property_identities

    df = pd.DataFrame(
        [
            {
                "Location": "12 Nguyễn Huệ, Quận 1, TP.HCM",
                "Property Type": "Nhà riêng",
                "Area": 80.0,
                "Width": 4.0,
                "Length": 20.0,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "Price": 8500.0,
                "listing_date": "2025-01-01",
            },
            {
                "Location": "12 Nguyễn Huệ, Quận 1, TP.HCM",
                "Property Type": "Nhà riêng",
                "Area": 80.0,
                "Width": 4.0,
                "Length": 20.0,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "Price": 9200.0,
                "listing_date": "2025-06-01",
            },
            {
                "Location": "Quận 3, TP.HCM",
                "Property Type": "Căn hộ chung cư",
                "Area": 65.0,
                "Price": 4500.0,
                "listing_date": "2025-03-01",
            },
        ]
    )
    resolved, audit = resolve_property_identities(df)
    assert (
        resolved["property_group_id"].iloc[0] == resolved["property_group_id"].iloc[1]
    )
    assert (
        resolved["property_group_id"].iloc[0] != resolved["property_group_id"].iloc[2]
    )
    assert resolved["identity_confidence"].iloc[0] == "strong"
    assert resolved["identity_confidence"].iloc[2] == "singleton"
    assert audit["unique_property_groups"] == 2
    assert audit["multi_listing_groups_count"] == 1
    assert audit["largest_group_size"] == 2


def test_separate_components_never_share_group_id():
    """Địa chỉ cấp quận và chữ ký giống nhau không đủ để gộp hai căn nhà."""
    from src.data.identity import resolve_property_identities

    row = {
        "Location": "Quận 7, TP.HCM",
        "Property Type": "Nhà riêng",
        "Area": 80,
        "Bedrooms": 3,
        "Bathrooms": 2,
    }
    resolved, audit = resolve_property_identities(pd.DataFrame([row, row]))
    assert resolved["property_group_id"].nunique() == 2
    assert audit["unique_property_groups"] == 2
    assert resolved["possible_duplicate"].all()


def test_medium_matches_are_review_only():
    """Cùng phường nhưng khác số nhà không tự động tạo một nhóm."""
    from src.data.identity import resolve_property_identities

    rows = [
        {
            "Location": f"{number} Nguyễn Huệ, Phường Bến Nghé, Quận 1",
            "Property Type": "Nhà riêng",
            "Area": 80,
        }
        for number in (12, 14, 16)
    ]
    resolved, audit = resolve_property_identities(pd.DataFrame(rows))
    assert audit["level_counts"]["medium"] == 3
    assert resolved["property_group_id"].nunique() == 3
    assert resolved["possible_duplicate"].all()


def test_source_listing_id_is_scoped_to_source():
    """Mã tin trùng chỉ được gộp nếu thuộc cùng một nguồn."""
    from src.data.identity import resolve_property_identities

    rows = [
        {"source_id": source, "source_listing_id": "42", "Area": 80}
        for source in ("a", "a", "b")
    ]
    resolved, _ = resolve_property_identities(pd.DataFrame(rows))
    assert (
        resolved["property_group_id"].iloc[0] == resolved["property_group_id"].iloc[1]
    )
    assert resolved["property_group_id"].nunique() == 2
