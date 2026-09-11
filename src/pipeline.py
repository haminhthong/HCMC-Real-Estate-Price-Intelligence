"""Pipeline train và đánh giá giá niêm yết bất động sản TP.HCM.

Luồng duy nhất: làm sạch và resolve identity → split theo thời gian và property
group → chọn Baseline/Ridge/ExtraTrees → refit → conformal calibration → future
test → ghi model, báo cáo và comparables.
"""

import argparse
from pathlib import Path
from typing import Any

from src.artifacts.loader import clear_model_cache
from src.artifacts.writer import save_model_artifacts
from src.calibration.conformal import calibrate_conformal
from src.comparables import ComparableContext, evaluate_comparables_on_validation
from src.config import CANONICAL_SPLIT_PROTOCOL, DATA_PATH, MODEL_VERSION, logger
from src.data.cleaning import clean_data
from src.data.loader import load_raw_dataset
from src.data.split import split_group_indices
from src.evaluation.evaluator import evaluate_champion_on_test
from src.evaluation.report import format_evaluation_summary
from src.features.builder import build_features
from src.modeling.selector import select_champion_model
from src.modeling.trainer import refit_champion_model


def run_pipeline(
    data_path: Path | str = DATA_PATH,
    model_version: str = MODEL_VERSION,
) -> dict[str, Any]:
    """Chạy pipeline từ dữ liệu thô đến model, báo cáo và artifact hiện hành."""
    logger.info("Bắt đầu pipeline giá bất động sản TP.HCM")

    # Nạp dữ liệu thô.
    raw_df = load_raw_dataset(data_path)

    # Kiểm tra dữ liệu, nhận diện bất động sản và loại tin đăng trùng.
    clean_df = clean_data(raw_df)
    audit_stats = getattr(clean_df, "attrs", {}).get("data_audit", {})

    if len(clean_df) < 50:
        raise ValueError(
            "Số lượng mẫu hợp lệ sau làm sạch quá nhỏ (< 50) để chia 4 tập."
        )

    # Chia nhóm bất động sản theo thời gian với tỷ lệ 60 / 15 / 10 / 15.
    train_idx, val_idx, calib_idx, test_idx = split_group_indices(clean_df)
    logger.info(
        "Grouped Temporal Split hoàn tất: Train=%d, Validation=%d, Calibration=%d, Test=%d.",
        len(train_idx),
        len(val_idx),
        len(calib_idx),
        len(test_idx),
    )

    df_train = clean_df.iloc[train_idx].copy()
    df_val = clean_df.iloc[val_idx].copy()
    df_calib = clean_df.iloc[calib_idx].copy()
    df_test = clean_df.iloc[test_idx].copy()

    # Chọn mô hình và biến mục tiêu bằng tập xác thực.
    selection_result = select_champion_model(df_train=df_train, df_val=df_val)
    selected_model_name = selection_result["selected_model_name"]

    # Huấn luyện lại mô hình được chọn trên tập huấn luyện và xác thực gộp.
    refit_result = refit_champion_model(
        df_train=df_train,
        df_val=df_val,
        selected_model_name=selected_model_name,
    )
    champion_pipeline = refit_result["champion_pipeline"]
    final_feature_context = refit_result["final_feature_context"]

    # Hiệu chuẩn khoảng dự báo trên tập riêng chiếm khoảng 10% số nhóm.
    features_calib = build_features(df_calib, context=final_feature_context)
    calibration_result = calibrate_conformal(
        model_pipeline=champion_pipeline,
        features_calib=features_calib,
        df_calib=df_calib,
        target_coverage=0.8,
    )

    # Đánh giá mô hình được chọn và các mô hình cơ sở trên tập kiểm tra.
    features_test = build_features(df_test, context=final_feature_context)
    evaluation_result = evaluate_champion_on_test(
        champion_pipeline=champion_pipeline,
        df_test=df_test,
        features_test=features_test,
        residual_log_quantile=calibration_result["residual_log_quantile"],
        naive_baseline=refit_result["naive_baseline"],
        segment_baseline=refit_result["segment_baseline"],
        target_coverage=0.8,
    )

    champion_metrics = evaluation_result["champion_metrics"]
    int_metrics = evaluation_result["interval_metrics"]

    # Chuẩn bị dữ liệu tham chiếu và đánh giá tin đăng tương đồng.
    ref_df = refit_result["df_train_dev"].copy()
    ref_df["distance_to_cbd_km"] = refit_result["features_train_dev"][
        "distance_to_cbd_km"
    ].to_numpy()
    # Benchmark dùng context Train-only. Serving sau khi development hoàn tất
    # được phép dùng reference Train+Validation, nhưng không dùng context đó
    # để benchmark ngược lại.
    benchmark_context = ComparableContext.fit(df_train)
    comparable_context = ComparableContext.fit(ref_df)

    # Đánh giá trên tập xác thực, chỉ lấy tin tham chiếu từ tập huấn luyện.
    comp_eval_result = evaluate_comparables_on_validation(
        df_train=df_train,
        df_val=df_val,
        context=benchmark_context,
        n_matches=4,
    )
    logger.info(
        "Comparable Engine Validation Benchmark: %s", comp_eval_result["summary"]
    )

    # Tổng hợp dữ liệu, cách chia tập và nguồn dữ liệu của lần chạy.
    def date_range(frame: Any) -> tuple[str, str]:
        if frame["listing_date"].notna().any():
            return (
                frame["listing_date"].min().isoformat(),
                frame["listing_date"].max().isoformat(),
            )
        return ("unknown", "unknown")

    split_summary = {
        "protocol": CANONICAL_SPLIT_PROTOCOL,
        "train_groups_count": int(df_train["property_group_id"].nunique()),
        "validation_groups_count": int(df_val["property_group_id"].nunique()),
        "calibration_groups_count": int(df_calib["property_group_id"].nunique()),
        "test_groups_count": int(df_test["property_group_id"].nunique()),
        "train_rows": len(df_train),
        "validation_rows": len(df_val),
        "calibration_rows": len(df_calib),
        "test_rows": len(df_test),
        "train_date_range": date_range(df_train),
        "validation_date_range": date_range(df_val),
        "calibration_date_range": date_range(df_calib),
        "test_date_range": date_range(df_test),
    }
    date_min, date_max = date_range(clean_df)
    data_summary = {
        "source_file": Path(data_path).name,
        "model_version": model_version,
        "rows_raw": audit_stats.get("rows_raw", len(raw_df)),
        "rows_valid": audit_stats.get("rows_valid", len(clean_df)),
        "rows_clean": audit_stats.get("rows_clean", len(clean_df)),
        "unique_property_groups": audit_stats.get(
            "unique_property_groups", clean_df["property_group_id"].nunique()
        ),
        "multi_listing_groups_count": audit_stats.get("multi_listing_groups_count", 0),
        "largest_group_size": audit_stats.get("largest_group_size", 1),
        "rows_removed_by_reason": audit_stats.get("rows_removed_by_reason", {}),
        "missing_rate_by_column": {
            col: round(float(clean_df[col].isna().mean() * 100), 2)
            for col in clean_df.columns
        },
        "target_percentiles": {
            f"p{p}": round(float(clean_df["Price"].quantile(p / 100)), 1)
            for p in [10, 25, 50, 75, 90]
        },
        "area_percentiles": {
            f"p{p}": round(float(clean_df["Area"].quantile(p / 100)), 1)
            for p in [10, 25, 50, 75, 90]
        },
        "date_min": date_min,
        "date_max": date_max,
        "reference_date_final": final_feature_context.reference_date,
        "split_protocol": CANONICAL_SPLIT_PROTOCOL,
        "known_date_rows": int(clean_df["listing_date"].notna().sum()),
        "unknown_date_rows": int(clean_df["listing_date"].isna().sum()),
        "district_coverage": sorted(
            clean_df["location_area"].dropna().unique().tolist()
        ),
        "data_quality_funnel": {
            "raw_listings": int(audit_stats.get("rows_raw", len(raw_df))),
            "market_scope_valid": int(audit_stats.get("rows_valid", len(clean_df))),
            "identity_resolved": int(
                audit_stats.get(
                    "rows_identity_resolved",
                    clean_df["property_group_id"].notna().sum(),
                )
            ),
            "duplicate_listing_events_removed": int(
                audit_stats.get("rows_removed_by_reason", {}).get(
                    "exact_duplicate_listings", 0
                )
            ),
            "clean_listing_events": len(clean_df),
            "temporal_eligible": int(clean_df["listing_date"].notna().sum()),
        },
        "target": "Price (triệu VND, giá đăng rao)",
        "comparable_validation_benchmark": comp_eval_result,
    }

    data_summary.update(
        {
            "model_version": model_version,
            "model_type": selected_model_name,
            "target_formulation": "total_price",
            "property_types": sorted(
                ref_df["Property Type"].dropna().unique().tolist()
            ),
            "supported_areas": sorted(
                ref_df["location_area"].dropna().unique().tolist()
            ),
            "split_summary": split_summary,
        }
    )

    # Lưu mô hình, dữ liệu phục vụ dự báo và báo cáo hiện hành.
    artifact_paths = save_model_artifacts(
        model_version=model_version,
        pipeline=champion_pipeline,
        feature_context=final_feature_context,
        calibration_result=calibration_result,
        evaluation_result=evaluation_result,
        selection_result=selection_result,
        data_summary=data_summary,
        reference_df=ref_df,
        training_ranges=refit_result["training_ranges"],
        segment_unit_prices=refit_result["segment_unit_prices"],
        comparable_context=comparable_context,
    )

    # Xóa LRU cache để serving nhận ngay artifact mới
    clear_model_cache()

    logger.info(
        "Pipeline hoàn tất: model=%s, test_rows=%d", selected_model_name, len(df_test)
    )
    summary_text = format_evaluation_summary(evaluation_result)
    try:
        print(summary_text)
    except UnicodeEncodeError:
        import sys

        sys.stdout.buffer.write(summary_text.encode("utf-8"))
        sys.stdout.write("\n")

    return {
        "model_version": model_version,
        "champion_model": selected_model_name,
        "target_formulation": "total_price",
        "champion_metrics": champion_metrics,
        "interval_metrics": int_metrics,
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
    }


def main() -> None:
    """CLI entrypoint cho pipeline."""
    parser = argparse.ArgumentParser(
        description="HCMC Real Estate Price Intelligence ML Pipeline"
    )
    parser.add_argument(
        "action",
        choices=["train", "evaluate"],
        default="train",
        nargs="?",
        help="Hành động cần thực thi: 'train' (toàn bộ pipeline) hoặc 'evaluate' (xem báo cáo)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_PATH),
        help="Đường dẫn file dữ liệu đầu vào",
    )

    args = parser.parse_args()

    if args.action == "train":
        run_pipeline(data_path=args.data_path)
    elif args.action == "evaluate":
        from src.evaluate import main as eval_main

        eval_main()


if __name__ == "__main__":
    main()
