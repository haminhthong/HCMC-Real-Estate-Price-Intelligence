"""Mô-đun kiểm định độc lập hiệu năng của động cơ Comparable Properties (Comparable Evaluation).

Thực hiện benchmark trên tập Validation:
Với mỗi bất động sản trong Validation, động cơ CHỈ tra cứu từ tập TRAIN (loại trừ cùng property_group_id).
Đo lường sai số định giá trung vị của các căn tương đồng (Comparable Median) đối chiếu với
Naive Median, Segment Median và mô hình Machine Learning.
"""

from typing import Any

import numpy as np
import pandas as pd

from .context import ComparableContext
from .engine import find_comparables


def evaluate_comparables_on_validation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    context: ComparableContext | None = None,
    n_matches: int = 4,
) -> dict[str, Any]:
    """Đánh giá ngoại suy định giá của Comparable Engine trên tập Validation độc lập.

    Args:
        df_train: DataFrame tập huấn luyện (chỉ được tra cứu comps từ đây).
        df_val: DataFrame tập kiểm chuẩn.
        context: Gói ComparableContext đã fit.
        n_matches: Số lượng căn tương đồng trích xuất (Top-K).

    Returns:
        dict chứa các chỉ số MAE, Median AE, WAPE và bảng so sánh đối chuẩn baselines.
    """
    if context is None:
        context = ComparableContext.fit(df_train)

    # Chuẩn bị reference listings từ tập Train
    references = []
    for row in df_train.to_dict(orient="records"):
        area_raw = row.get("Area")
        price_raw = row.get("Price")
        area = float(area_raw) if pd.notna(area_raw) else None
        price = float(price_raw) if pd.notna(price_raw) else None
        unit_p = (
            round(price / area, 2)
            if price is not None and area is not None and area > 0
            else None
        )
        beds_raw = row.get("Bedrooms")
        baths_raw = row.get("Bathrooms")
        floors_raw = row.get("Floors")
        lat_raw = row.get("Latitude")
        lon_raw = row.get("Longitude")
        cbd_raw = row.get("distance_to_cbd_km")
        date_raw = row.get("listing_date")
        references.append(
            {
                "listing_id": row.get("Listing ID"),
                "property_group_id": row.get("property_group_id"),
                "property_type": row.get("Property Type"),
                "location_area": row.get("location_area"),
                "area": area,
                "bedrooms": float(beds_raw) if pd.notna(beds_raw) else None,
                "bathrooms": float(baths_raw) if pd.notna(baths_raw) else None,
                "floors": float(floors_raw) if pd.notna(floors_raw) else None,
                "price_million": price,
                "unit_price_million_m2": unit_p,
                "latitude": float(lat_raw) if pd.notna(lat_raw) else None,
                "longitude": float(lon_raw) if pd.notna(lon_raw) else None,
                "distance_to_cbd_km": float(cbd_raw) if pd.notna(cbd_raw) else None,
                "listing_date": str(date_raw) if pd.notna(date_raw) else None,
            }
        )

    model_package = {
        "reference_listings": references,
        "comparable_context": context.to_dict(),
    }

    train_median_price = float(df_train["Price"].median())

    # Bảng tra cứu segment median đơn giá từ Train
    unit_prices = df_train["Price"] / df_train["Area"].replace(0, np.nan)
    segment_unit_lookup = (
        df_train.assign(_unit_price=unit_prices)
        .groupby(["Property Type", "location_area"])["_unit_price"]
        .median()
        .to_dict()
    )

    actuals = []
    comp_preds = []
    naive_preds = []
    segment_preds = []

    for val_row in df_val.to_dict(orient="records"):
        actual_price = float(val_row["Price"])
        actuals.append(actual_price)
        naive_preds.append(train_median_price)

        # Dự báo cơ sở bằng trung vị của phân khúc.
        p_type = val_row.get("Property Type")
        loc_area = val_row.get("location_area")
        area = float(val_row.get("Area", 80.0))
        seg_unit = segment_unit_lookup.get((p_type, loc_area))
        if seg_unit is not None and not np.isnan(seg_unit):
            seg_pred = seg_unit * area
        else:
            seg_pred = train_median_price
        segment_preds.append(seg_pred)

        # Ước lượng bằng trung vị giá các tin tương đồng để đối chiếu.
        query_val = dict(val_row)
        # Định giá tại ngày của validation listing để engine chỉ nhìn lịch sử.
        query_val["as_of_date"] = val_row.get("listing_date")
        _, summary = find_comparables(
            model_package=model_package,
            values=query_val,
            n_matches=n_matches,
            context=context,
        )
        comp_price = summary.get("median_price_million")
        if comp_price is None or np.isnan(comp_price):
            comp_price = seg_pred
        comp_preds.append(comp_price)

    y_true = np.array(actuals, dtype=float)
    y_comp = np.array(comp_preds, dtype=float)
    y_naive = np.array(naive_preds, dtype=float)
    y_seg = np.array(segment_preds, dtype=float)

    total_actual = float(np.sum(y_true))

    def calc_stats(y_p: np.ndarray) -> dict[str, float]:
        ae = np.abs(y_p - y_true)
        mae = float(np.mean(ae))
        med_ae = float(np.median(ae))
        wape = float(np.sum(ae) / max(total_actual, 1.0) * 100.0)
        return {
            "mae_million": round(mae, 2),
            "median_ae_million": round(med_ae, 2),
            "wape_percent": round(wape, 2),
        }

    comp_stats = calc_stats(y_comp)
    naive_stats = calc_stats(y_naive)
    seg_stats = calc_stats(y_seg)

    improvement_vs_naive = round(
        (naive_stats["mae_million"] - comp_stats["mae_million"])
        / max(naive_stats["mae_million"], 1.0)
        * 100.0,
        2,
    )

    return {
        "evaluation_samples": len(df_val),
        "comparable_metrics": comp_stats,
        "naive_baseline_metrics": naive_stats,
        "segment_baseline_metrics": seg_stats,
        "improvement_over_naive_percent": improvement_vs_naive,
        "summary": (
            f"Comparable Median đạt Val MAE={comp_stats['mae_million']:,.1f}M "
            f"(WAPE={comp_stats['wape_percent']}%), so với Naive MAE={naive_stats['mae_million']:,.1f}M "
            f"và Segment MAE={seg_stats['mae_million']:,.1f}M."
        ),
    }
