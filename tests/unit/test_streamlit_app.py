"""Kiểm thử cấu trúc và chức năng của Streamlit App."""

from unittest.mock import patch

from app.streamlit_app import render_formatted_warning


def test_render_formatted_warning_all_branches():
    """Đảm bảo mọi loại cảnh báo đều được format thành card mà không phát sinh lỗi."""
    warnings = [
        "STALE_MARKET_REFERENCE: Dữ liệu train cũ hơn 200 ngày.",
        "MISSING_GPS: Thiếu GPS nên comparable không dùng được khoảng cách địa lý.",
        "WIDE_PREDICTION_INTERVAL: Khoảng dự báo rộng 85%.",
        "INPUT_OUTSIDE_TRAINING_RANGE: Area=520 nằm ngoài miền train [5, 500].",
        "MISSING_INPUTS: Độ đầy đủ input chỉ đạt 40/100.",
        "UNKNOWN_WARNING: Một cảnh báo khác.",
    ]

    with patch("streamlit.markdown") as mock_markdown:
        for w in warnings:
            render_formatted_warning(w)
        assert mock_markdown.call_count == len(warnings)
