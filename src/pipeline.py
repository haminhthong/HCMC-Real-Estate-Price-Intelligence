"""Mô-đun điều phối Master ML Lifecycle Pipeline cho HCMC Real Estate Price Intelligence.

Quy trình chuẩn hóa 12 bước:
1. DATA INGESTION: Nạp dữ liệu thô
2. DATA CLEANING & IDENTITY: Chuẩn hóa, lọc ngoại lệ, định danh bất động sản, deduplicate cấp độ tin đăng
3. DATA MANIFEST: Khởi tạo DatasetManifest
4. GROUPED TEMPORAL SPLIT: Phân chia 60/15/10/15 cô lập nhóm bất động sản theo ngày muộn nhất
5. SPLIT MANIFEST: Khởi tạo SplitManifest
6. PHASE A - MODEL SELECTION: Train (60%) -> Candidate models -> Validation (15%) -> Chọn Champion
7. PHASE B - FINAL REFIT: Train + Validation (75%) -> Cập nhật reference_date_final -> Refit Champion
8. UNCERTAINTY CALIBRATION: Calibration (10%) -> Tính phân vị residual conformal quantile
9. INDEPENDENT EVALUATION: Test (15%) -> Đánh giá duy nhất Champion vs Naive & Segment Baselines
10. PROMOTION GATE: Kiểm tra tiêu chuẩn triển khai sẵn sàng thực tế
11. DATA CARD & AUDIT: Tổng hợp thẻ dữ liệu
12. ARTIFACT PERSISTENCE: Lưu trữ versioned folder, decoupled reference dataset, và cập nhật production pointer
"""

import argparse
from pathlib import Path
from typing import Any

from src.artifacts.loader import clear_model_cache
from src.artifacts.schema import (
    evaluate_development_gate,
    evaluate_release_gate,
)
from src.artifacts.writer import save_model_artifacts
from src.calibration.conformal import calibrate_conformal
from src.config import DATA_PATH, MODEL_VERSION, logger
from src.data.cleaning import clean_data
from src.data.loader import load_raw_dataset
from src.data.manifest import create_dataset_manifest, create_split_manifest
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
    """Chạy toàn bộ quy trình Master ML Lifecycle từ dữ liệu thô đến khi lưu trữ artifacts."""
    logger.info("==========================================================================")
    logger.info("   BẮT ĐẦU MASTER ML LIFECYCLE PIPELINE (HCMC REAL ESTATE INTELLIGENCE)  ")
    logger.info("==========================================================================")

    # 1. DATA INGESTION
    raw_df = load_raw_dataset(data_path)

    # 2. DATA QUALITY & PROPERTY IDENTITY & LISTING DEDUPLICATION
    clean_df = clean_data(raw_df)
    audit_stats = getattr(clean_df, "attrs", {}).get("data_audit", {})

    if len(clean_df) < 50:
        raise ValueError("Số lượng mẫu hợp lệ sau làm sạch quá nhỏ (< 50) để chia 4 tập.")

    # 3. DATASET MANIFEST
    dataset_manifest = create_dataset_manifest(
        clean_df,
        source_path=data_path,
        snapshot_id=Path(data_path).stem,
    )

    # 4. GROUPED TEMPORAL SPLIT (60 / 15 / 10 / 15)
    train_idx, val_idx, calib_idx, test_idx = split_group_indices(clean_df)
    logger.info(
        "Grouped Temporal Split hoàn tất: Train=%d, Validation=%d, Calibration=%d, Test=%d.",
        len(train_idx),
        len(val_idx),
        len(calib_idx),
        len(test_idx),
    )

    # 5. SPLIT MANIFEST
    split_manifest = create_split_manifest(
        clean_df,
        train_idx=train_idx,
        validation_idx=val_idx,
        calibration_idx=calib_idx,
        test_idx=test_idx,
    )

    df_train = clean_df.iloc[train_idx].copy()
    df_val = clean_df.iloc[val_idx].copy()
    df_calib = clean_df.iloc[calib_idx].copy()
    df_test = clean_df.iloc[test_idx].copy()

    # 6. PHASE A: MODEL & TARGET FORMULATION SELECTION (Train 60% -> Val 15%)
    selection_result = select_champion_model(df_train=df_train, df_val=df_val)
    selected_model_name = selection_result["selected_model_name"]
    selected_target_fmt = selection_result["selected_target_fmt"]

    # 7. PHASE B: FINAL REFIT CHAMPION MODEL (Train + Validation = 75%)
    refit_result = refit_champion_model(
        df_train=df_train,
        df_val=df_val,
        selected_model_name=selected_model_name,
        selected_target_fmt=selected_target_fmt,
    )
    champion_pipeline = refit_result["champion_pipeline"]
    final_feature_context = refit_result["final_feature_context"]

    # 8. CONFORMAL CALIBRATION (Calibration 10% với model đã refit)
    features_calib = build_features(df_calib, context=final_feature_context)
    calibration_result = calibrate_conformal(
        model_pipeline=champion_pipeline,
        features_calib=features_calib,
        df_calib=df_calib,
        target_formulation=selected_target_fmt,
        target_coverage=0.8,
    )

    # 9. INDEPENDENT EVALUATION (Test 15% - Champion vs Baselines ONLY)
    features_test = build_features(df_test, context=final_feature_context)
    evaluation_result = evaluate_champion_on_test(
        champion_pipeline=champion_pipeline,
        df_test=df_test,
        features_test=features_test,
        selected_target_fmt=selected_target_fmt,
        residual_log_quantile=calibration_result["residual_log_quantile"],
        naive_baseline=refit_result["naive_baseline"],
        segment_baseline=refit_result["segment_baseline"],
        target_coverage=0.8,
    )

    # 10. GOVERNANCE & PROMOTION GATES (Development Gate on Val, Release Gate on Test)
    champion_metrics = evaluation_result["champion_metrics"]
    int_metrics = evaluation_result["interval_metrics"]

    # Development Gate chỉ được đọc metrics của champion trên Validation.
    # Không dùng fallback số cứng vì nó có thể che giấu lỗi selection.
    dev_gate = evaluate_development_gate(
        champion_val_mae=selection_result["best_val_mae"],
        naive_val_mae=selection_result["naive_val_mae"],
        val_wape=float(selection_result["selected_validation_metrics"]["wape_percent"]),
    )
    rel_gate = evaluate_release_gate(
        test_wape=float(champion_metrics["wape_percent"]),
        test_coverage=float(int_metrics["actual_coverage"]),
        relative_interval_width=float(int_metrics["relative_interval_width"]),
    )
    # Development và Release là hai quyết định độc lập.
    # Test chỉ đi vào Release Gate, không bị truyền ngược thành Validation WAPE.
    promotion_result = {
        "selection_status": "champion" if dev_gate["champion_approved"] else "research_candidate",
        "production_readiness": rel_gate["readiness_status"],
        "deployment_approved": bool(rel_gate["production_ready"]),
        "promotion_status": {
            "model_selected": True,
            "beats_baseline": dev_gate["beats_baseline"],
            "baseline_improvement_percent": dev_gate["baseline_improvement_percent"],
            "wape_acceptable": dev_gate["val_wape_acceptable"],
            "interval_calibrated": rel_gate["interval_coverage_acceptable"],
            "production_ready": rel_gate["production_ready"],
        },
        "development_gate": dev_gate,
        "release_gate": rel_gate,
        "promotion_reason": rel_gate["gate_reason"],
    }
    logger.info("%s", dev_gate["gate_reason"])
    logger.info("%s", rel_gate["gate_reason"])

    # 11. COMPARABLE ENGINE CONTEXT & OFFLINE VALIDATION BENCHMARK
    from src.comparables import ComparableContext, evaluate_comparables_on_validation
    ref_df = refit_result["df_train_dev"].copy()
    ref_df["distance_to_cbd_km"] = refit_result["features_train_dev"]["distance_to_cbd_km"].to_numpy()
    # Benchmark dùng context Train-only. Serving sau khi development hoàn tất
    # được phép dùng reference Train+Validation, nhưng không dùng context đó
    # để benchmark ngược lại.
    benchmark_context = ComparableContext.fit(df_train)
    comparable_context = ComparableContext.fit(ref_df)

    # Đánh giá ngoại suy định giá của Comparable Engine trên Validation set (Train comps ONLY)
    comp_eval_result = evaluate_comparables_on_validation(
        df_train=df_train,
        df_val=df_val,
        context=benchmark_context,
        n_matches=4,
    )
    logger.info("Comparable Engine Validation Benchmark: %s", comp_eval_result["summary"])

    # 12. DATA CARD & METADATA
    data_card = {
        "source_file": Path(data_path).name,
        "model_version": model_version,
        "rows_raw": audit_stats.get("rows_raw", len(raw_df)),
        "rows_valid": audit_stats.get("rows_valid", len(clean_df)),
        "rows_clean": audit_stats.get("rows_clean", len(clean_df)),
        "unique_property_groups": audit_stats.get("unique_property_groups", clean_df["property_group_id"].nunique()),
        "multi_listing_groups_count": audit_stats.get("multi_listing_groups_count", 0),
        "largest_group_size": audit_stats.get("largest_group_size", 1),
        "rows_removed_by_reason": audit_stats.get("rows_removed_by_reason", {}),
        "missing_rate_by_column": {
            col: round(float(clean_df[col].isna().mean() * 100), 2) for col in clean_df.columns
        },
        "target_percentiles": {
            f"p{p}": round(float(clean_df["Price"].quantile(p / 100)), 1)
            for p in [10, 25, 50, 75, 90]
        },
        "area_percentiles": {
            f"p{p}": round(float(clean_df["Area"].quantile(p / 100)), 1)
            for p in [10, 25, 50, 75, 90]
        },
        "date_min": dataset_manifest.date_min,
        "date_max": dataset_manifest.date_max,
        "reference_date_final": final_feature_context.reference_date,
        "split_protocol": split_manifest.protocol,
        "target": "Price (triệu VND, giá đăng rao)",
        "development_gate": dev_gate,
        "release_gate": rel_gate,
        "promotion_status": promotion_result["promotion_status"],
        "readiness_status": promotion_result["production_readiness"],
        "comparable_validation_benchmark": comp_eval_result,
    }

    # 13. ARTIFACT PERSISTENCE
    artifact_paths = save_model_artifacts(
        version=model_version,
        pipeline=champion_pipeline,
        feature_context=final_feature_context,
        calibration_result=calibration_result,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
        evaluation_result=evaluation_result,
        selection_result=selection_result,
        promotion_result=promotion_result,
        data_card=data_card,
        reference_df=ref_df,
        training_ranges=refit_result["training_ranges"],
        training_quantiles=refit_result["training_quantiles"],
        segment_unit_prices=refit_result["segment_unit_prices"],
        comparable_context=comparable_context,
    )

    # Xóa LRU cache để serving nhận ngay artifact mới
    clear_model_cache()

    logger.info("==========================================================================")
    logger.info("   PIPELINE HOÀN TẤT THÀNH CÔNG!                                         ")
    logger.info("==========================================================================")
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
        "target_formulation": selected_target_fmt,
        "champion_metrics": champion_metrics,
        "interval_metrics": int_metrics,
        "promotion_result": promotion_result,
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
    }


def main() -> None:
    """CLI entrypoint cho master pipeline."""
    parser = argparse.ArgumentParser(description="HCMC Real Estate Price Intelligence ML Pipeline")
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
    parser.add_argument(
        "--version",
        type=str,
        default=MODEL_VERSION,
        help="Phiên bản mô hình (ví dụ: 1.2.0)",
    )

    args = parser.parse_args()

    if args.action == "train":
        run_pipeline(data_path=args.data_path, model_version=args.version)
    elif args.action == "evaluate":
        from src.evaluate import main as eval_main
        eval_main()


if __name__ == "__main__":
    main()
