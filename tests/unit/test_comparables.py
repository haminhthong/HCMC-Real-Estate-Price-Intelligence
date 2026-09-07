"""Kiểm thử cho Comparable Context, Comparable Engine và Offline Evaluation."""

import pandas as pd
import pytest
from src.comparables import ComparableContext, evaluate_comparables_on_validation, find_comparables


def test_comparable_context_fit_and_serialization():
    df = pd.DataFrame(
        [
            {"Area": 50.0, "distance_to_cbd_km": 5.0, "Property Type": "Nhà riêng", "location_area": "Quận 1"},
            {"Area": 100.0, "distance_to_cbd_km": 15.0, "Property Type": "Căn hộ chung cư", "location_area": "Quận 7"},
        ]
    )
    ctx = ComparableContext.fit(df)
    assert ctx.median_area == 75.0
    assert "Nhà riêng | Quận 1" in ctx.segment_counts
    d = ctx.to_dict()
    restored = ComparableContext.from_dict(d)
    assert restored.median_area == ctx.median_area


def test_find_comparables_multi_dimensional():
    ref_listings = [
        {
            "property_type": "Nhà riêng",
            "location_area": "Quận 1",
            "area": 82.0,
            "bedrooms": 3,
            "bathrooms": 2,
            "floors": 2,
            "latitude": 10.7769,
            "longitude": 106.7009,
            "distance_to_cbd_km": 0.5,
            "price_million": 8500.0,
            "unit_price_million_m2": 103.66,
            "property_group_id": "group-ref-1",
        },
        {
            "property_type": "Nhà riêng",
            "location_area": "Quận 1",
            "area": 78.0,
            "bedrooms": 3,
            "bathrooms": 2,
            "floors": 2,
            "latitude": 10.7770,
            "longitude": 106.7010,
            "distance_to_cbd_km": 0.6,
            "price_million": 8300.0,
            "unit_price_million_m2": 106.41,
            "property_group_id": "group-ref-2",
        },
        {
            "property_type": "Nhà riêng",
            "location_area": "Quận 1",
            "area": 250.0,
            "bedrooms": 6,
            "bathrooms": 6,
            "floors": 5,
            "latitude": 10.7780,
            "longitude": 106.7020,
            "distance_to_cbd_km": 1.2,
            "price_million": 45000.0,
            "unit_price_million_m2": 180.0,
            "property_group_id": "group-ref-3",
        },
    ]
    model_pkg = {"reference_listings": ref_listings}
    query = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 80.0,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Latitude": 10.7769,
        "Longitude": 106.7009,
        "distance_to_cbd_km": 0.5,
        "property_group_id": "group-target",
    }
    comps, summary = find_comparables(model_pkg, query, n_matches=2)
    assert len(comps) == 2
    # Top match should be 82m2 or 78m2, not the 250m2 villa
    assert comps[0]["area"] in (82.0, 78.0)
    assert comps[0]["similarity_score"] > comps[1]["similarity_score"] or comps[0]["similarity_score"] >= 0.8
    assert summary["median_price_million"] is not None


def test_evaluate_comparables_on_validation():
    df_train = pd.DataFrame(
        [
            {"Area": 80.0, "Price": 8000.0, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2, "Property Type": "Nhà riêng", "location_area": "Quận 1", "property_group_id": "g1"},
            {"Area": 85.0, "Price": 8500.0, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2, "Property Type": "Nhà riêng", "location_area": "Quận 1", "property_group_id": "g2"},
            {"Area": 90.0, "Price": 9000.0, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2, "Property Type": "Nhà riêng", "location_area": "Quận 1", "property_group_id": "g3"},
        ]
    )
    df_val = pd.DataFrame(
        [
            {"Area": 82.0, "Price": 8200.0, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2, "Property Type": "Nhà riêng", "location_area": "Quận 1", "property_group_id": "g4"},
        ]
    )
    res = evaluate_comparables_on_validation(df_train, df_val, n_matches=2)
    assert "comparable_metrics" in res
    assert "naive_baseline_metrics" in res
    assert res["comparable_metrics"]["mae_million"] < 500.0
