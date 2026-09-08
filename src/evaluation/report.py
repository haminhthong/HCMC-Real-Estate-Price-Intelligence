"""Mô-đun định dạng và xuất báo cáo đánh giá mô hình (Reporting)."""

from typing import Any


def format_evaluation_summary(eval_result: dict[str, Any]) -> str:
    """Tạo chuỗi văn bản tóm tắt kết quả đánh giá mô hình trên tập Test độc lập."""
    champ = eval_result.get("champion_metrics", {})
    intervals = eval_result.get("interval_metrics", {})
    baselines = eval_result.get("baselines", {})

    naive = baselines.get("naive_median", {})
    segment = baselines.get("district_property_segment_median", {})

    lines = [
        "=" * 70,
        "   BÁO CÁO ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ĐỘC LẬP (REPORT ONLY)",
        "=" * 70,
        f"Số lượng mẫu kiểm thử (Test Samples): {eval_result.get('test_sample_count', 'N/A')}",
        "",
        "1. SO SÁNH HIỆU NĂNG VỚI BASELINES:",
        f"   - Champion Model:        MAE = {champ.get('mae_million', 0):,.1f} tr | WAPE = {champ.get('wape_percent', 0):.1f}% | R² = {champ.get('r2', 0):.3f}",
        f"   - Naive Median:          MAE = {naive.get('mae_million', 0):,.1f} tr | WAPE = {naive.get('wape_percent', 0):.1f}% | R² = {naive.get('r2', 0):.3f}",
        f"   - Segment Median:        MAE = {segment.get('mae_million', 0):,.1f} tr | WAPE = {segment.get('wape_percent', 0):.1f}% | R² = {segment.get('r2', 0):.3f}",
        "",
        "2. CHẤT LƯỢNG KHOẢNG DỰ BÁO (CONFORMAL PREDICTION INTERVAL):",
        f"   - Mục tiêu bao phủ (Target Coverage):  {intervals.get('target_coverage', 0) * 100:.1f}%",
        f"   - Bao phủ thực tế (Actual Coverage):    {intervals.get('actual_coverage', 0) * 100:.1f}%",
        f"   - Chênh lệch bao phủ (Coverage Gap):    {intervals.get('coverage_gap', 0) * 100:+.1f}%",
        f"   - Độ rộng trung bình (Mean Width):     {intervals.get('mean_interval_width_million', 0):,.1f} triệu VND",
        f"   - Độ rộng tương đối (Relative Width):  {intervals.get('relative_interval_width', 0) * 100:.1f}%",
        "=" * 70,
    ]
    return "\n".join(lines)
