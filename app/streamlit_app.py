"""Property Price Intelligence Console - Streamlit Application.

Decision-support tool for residential listing-price estimation using historical
listing evidence, conformal uncertainty intervals, comparable retrieval, and
model explainability.
"""

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from src.artifacts.loader import load_model
from src.config import RESIDENTIAL_TYPES, SUPPORTED_AREAS
from src.serving.predictor import predict_one

# ---------------------------------------------------------------------------
# Cấu hình nạp gói mô hình
# ---------------------------------------------------------------------------
try:
    _ui_model_package = load_model()
except (FileNotFoundError, KeyError, ValueError, OSError):
    _ui_model_package = {}

SUPPORTED_MODEL_TYPES = (
    _ui_model_package.get("supported_property_types") or RESIDENTIAL_TYPES
)
SUPPORTED_MODEL_AREAS = [
    area
    for area in (_ui_model_package.get("supported_areas") or SUPPORTED_AREAS)
    if area != "Unknown"
]
MODEL_VERSION = _ui_model_package.get("version", "1.2.0")
MODEL_TYPE = _ui_model_package.get("model_type", "extra_trees")
REFERENCE_DATE_STR = _ui_model_package.get("reference_date", "2025-09-30")

# ---------------------------------------------------------------------------
# Cấu hình giao diện và Theme CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HCMC Property Price Intelligence Console",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Tổng thể giao diện hiện đại */
    .reportview-container, .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        color: #334155;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    /* Thẻ chỉ số chính (Metric Cards) */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #CBD5E1;
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.4rem;
    }
    /* Thanh khoảng bất định trực quan (Uncertainty Range) */
    .uncertainty-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1.2rem 0;
    }
    .range-bar-track {
        position: relative;
        height: 10px;
        background: #E2E8F0;
        border-radius: 5px;
        margin: 25px 0 15px 0;
    }
    .range-bar-fill {
        position: absolute;
        height: 10px;
        background: linear-gradient(90deg, #93C5FD 0%, #3B82F6 50%, #1D4ED8 100%);
        border-radius: 5px;
    }
    .point-marker {
        position: absolute;
        top: -6px;
        width: 22px;
        height: 22px;
        background: #1E293B;
        border: 3px solid #FFFFFF;
        border-radius: 50%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.25);
        transform: translateX(-50%);
    }
    /* Thẻ cảnh báo dữ liệu (Warning Cards) */
    .alert-card {
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }
    .alert-card.warning {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
    }
    .alert-card.info {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
    }
    .alert-card.neutral {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        color: #334155;
    }
    .alert-title {
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.03em;
        margin-bottom: 0.15rem;
    }
    /* Comparable listing card */
    .comp-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .comp-badge {
        display: inline-block;
        background: #EFF6FF;
        color: #2563EB;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
    }
    .comp-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1E293B;
        margin-bottom: 0.3rem;
    }
    .comp-metric {
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 0.25rem;
    }
    .comp-price {
        font-size: 1.25rem;
        font-weight: 800;
        color: #059669;
        margin: 0.4rem 0;
    }
    /* Section grouping in form */
    .form-section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #2563EB;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #EFF6FF;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header & Hero
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero-title">🏠 HCMC Real Estate Price Intelligence Console</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">Hệ thống hỗ trợ ra quyết định ước lượng giá niêm yết bất động sản dân dụng TP.HCM từ dữ liệu tin đăng lịch sử</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="disclaimer-banner">
        <strong>📌 Tuyên bố trách nhiệm (Disclaimer):</strong> Giá ước tính phản ánh mức giá niêm yết kỳ vọng dựa trên các tin đăng tham chiếu lịch sử tại TP.HCM. Đây không phải là giá giao dịch thực tế đã công chứng hay chứng thư thẩm định giá pháp lý.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Phạm vi hỗ trợ và thông số mô hình
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Thông Số Hệ Thống")
    st.write(f"- **Mô hình**: `{MODEL_TYPE}`")
    st.write(f"- **Phiên bản**: `v{MODEL_VERSION}`")
    st.write("- **Biến mục tiêu**: Giá niêm yết (`total_price`)")
    st.write("- **Mục tiêu Conformal**: `80.0%`")
    st.write(f"- **Dữ liệu tham chiếu**: `{REFERENCE_DATE_STR}`")

    st.divider()
    st.markdown("### 🎯 Phạm Vi Hỗ Trợ (Scope)")
    st.markdown(
        """
        - **Thị trường**: Nhà ở dân dụng tại TP.HCM
        - **Diện tích đất**: `5 – 500 m²`
        - **Loại BĐS hỗ trợ**:
        """
    )
    for ptype in SUPPORTED_MODEL_TYPES:
        st.markdown(f"  - `{ptype}`")

    with st.expander("📍 23 Quận/Huyện được hỗ trợ"):
        for area in SUPPORTED_MODEL_AREAS:
            st.markdown(f"- {area}")

    st.divider()
    st.markdown("### 🚫 Giới Hạn Nghiêm Ngặt")
    st.caption(
        "Hệ thống không hỗ trợ: định giá tự động pháp lý, BĐS thương mại/dự án lớn, "
        "tính toán lãi suất vay, hay dự báo xu hướng đầu tư tương lai."
    )


# ---------------------------------------------------------------------------
# Hàm phân loại và hiển thị cảnh báo dễ đọc
# ---------------------------------------------------------------------------
def render_formatted_warning(raw_warning: str) -> None:
    """Chuyển cảnh báo thô thành card cảnh báo có phân loại rõ ràng."""
    if raw_warning.startswith("STALE_MARKET_REFERENCE"):
        st.markdown(
            f"""
            <div class="alert-card warning">
                <div>
                    <div class="alert-title">⚠️ Dữ liệu thị trường đã cũ (Market Data Outdated)</div>
                    <div>{raw_warning.replace('STALE_MARKET_REFERENCE: ', '')}. Mức giá ước lượng có thể không phản ánh đầy đủ biến động thị trường gần đây.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif raw_warning.startswith("MISSING_GPS"):
        st.markdown(
            """
            <div class="alert-card info">
                <div>
                    <div class="alert-title">ℹ️ Chưa cung cấp tọa độ GPS (GPS Not Provided)</div>
                    <div>Thuật toán tìm tin đăng tương đồng sẽ dùng mức phạt khoảng cách theo ranh giới quận thay vì cự ly địa lý chính xác (Haversine).</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif raw_warning.startswith("WIDE_PREDICTION_INTERVAL"):
        st.markdown(
            f"""
            <div class="alert-card warning">
                <div>
                    <div class="alert-title">⚠️ Độ bất định cao (High Uncertainty)</div>
                    <div>{raw_warning.replace('WIDE_PREDICTION_INTERVAL: ', '')}. Thị trường có sự phân tán giá lớn đối với phân khúc này.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif raw_warning.startswith("INPUT_OUTSIDE_TRAINING_RANGE"):
        st.markdown(
            f"""
            <div class="alert-card warning">
                <div>
                    <div class="alert-title">⚠️ Giá trị ngoài miền huấn luyện (Out of Training Range)</div>
                    <div>{raw_warning.replace('INPUT_OUTSIDE_TRAINING_RANGE: ', '')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif raw_warning.startswith("MISSING_INPUTS"):
        st.markdown(
            f"""
            <div class="alert-card neutral">
                <div>
                    <div class="alert-title">ℹ️ Thông tin đầu vào chưa đầy đủ (Incomplete Input)</div>
                    <div>{raw_warning.replace('MISSING_INPUTS: ', '')}. Việc bổ sung thêm số phòng, kích thước hoặc vị trí sẽ giúp tăng độ chuẩn xác.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-card neutral">
                <div>
                    <div class="alert-title">⚠️ Cảnh báo dữ liệu</div>
                    <div>{raw_warning}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Cấu trúc 4 Tab chính
# ---------------------------------------------------------------------------
tab_estimate, tab_comparables, tab_why, tab_model = st.tabs(
    [
        "🏠 Price Estimate",
        "🔍 Comparable Evidence",
        "💡 Why This Estimate?",
        "📊 Model & Data",
    ]
)

# ===========================================================================
# TAB 1: PRICE ESTIMATE (Ước Lượng Giá Listing)
# ===========================================================================
with tab_estimate:
    st.subheader("Nhập thông số bất động sản cần ước lượng")

    with st.form("property_estimate_form"):
        col_prop, col_loc, col_feat = st.columns(3)

        with col_prop:
            st.markdown(
                '<div class="form-section-title">1. Thuộc tính BĐS (Property)</div>',
                unsafe_allow_html=True,
            )
            prop_type = st.selectbox(
                "Loại hình bất động sản (*)",
                SUPPORTED_MODEL_TYPES,
                help="Chỉ hỗ trợ các loại hình nhà ở dân dụng chính tại TP.HCM.",
            )
            prop_area = st.number_input(
                "Diện tích sử dụng / đất (m²) (*)",
                min_value=5.0,
                max_value=500.0,
                value=75.0,
                step=5.0,
                help="Phạm vi mô hình hỗ trợ: 5.0 đến 500.0 m².",
            )
            prop_beds = st.number_input(
                "Số phòng ngủ (Bedrooms)",
                min_value=0,
                max_value=10,
                value=3,
                help="Nhập 0 nếu chưa rõ hoặc chưa có thông tin.",
            )
            prop_baths = st.number_input(
                "Số phòng vệ sinh (Bathrooms)",
                min_value=0,
                max_value=20,
                value=2,
            )
            prop_floors = st.number_input(
                "Số tầng (Floors)",
                min_value=0,
                max_value=100,
                value=2,
            )

        with col_loc:
            st.markdown(
                '<div class="form-section-title">2. Vị trí & Tọa độ (Location)</div>',
                unsafe_allow_html=True,
            )
            prop_district = st.selectbox(
                "Quận/Huyện khu vực (*)",
                SUPPORTED_MODEL_AREAS,
                index=0 if "Quận 1" not in SUPPORTED_MODEL_AREAS else 0,
            )
            prop_position = st.selectbox(
                "Vị trí bất động sản",
                ["Trong hẻm", "Đường chính", "Không rõ"],
            )
            prop_alley = st.number_input(
                "Độ rộng hẻm trước nhà (m)",
                min_value=0.0,
                max_value=30.0,
                value=3.5,
                step=0.5,
            )
            col_dim1, col_dim2 = st.columns(2)
            with col_dim1:
                prop_width = st.number_input(
                    "Mặt tiền (m)",
                    min_value=0.0,
                    max_value=100.0,
                    value=4.0,
                    step=0.5,
                    help="Để 0.0 nếu chưa có thông tin.",
                )
            with col_dim2:
                prop_length = st.number_input(
                    "Chiều dài (m)",
                    min_value=0.0,
                    max_value=200.0,
                    value=18.0,
                    step=1.0,
                    help="Để 0.0 nếu chưa có thông tin.",
                )
            prop_direction = st.selectbox(
                "Hướng nhà",
                [
                    "Không rõ",
                    "Đông",
                    "Tây",
                    "Nam",
                    "Bắc",
                    "Đông Nam",
                    "Đông Bắc",
                    "Tây Nam",
                    "Tây Bắc",
                ],
            )

            # Tọa độ GPS tùy chọn (Location details - optional)
            with st.expander("📍 Tọa độ GPS chính xác (Tùy chọn)"):
                st.caption(
                    "Nếu có GPS, engine sẽ tính khoảng cách địa lý thực (Haversine). "
                    "Nếu để trống, hệ thống sẽ tự động fallback sang khoảng cách theo quận."
                )
                col_gps1, col_gps2 = st.columns(2)
                with col_gps1:
                    prop_lat = st.number_input(
                        "Vĩ độ (Latitude)",
                        min_value=0.0,
                        max_value=12.0,
                        value=0.0,
                        format="%.5f",
                        help="TP.HCM nằm trong khoảng 10.38 đến 11.16.",
                    )
                with col_gps2:
                    prop_lon = st.number_input(
                        "Kinh độ (Longitude)",
                        min_value=0.0,
                        max_value=108.0,
                        value=0.0,
                        format="%.5f",
                        help="TP.HCM nằm trong khoảng 106.35 đến 107.03.",
                    )

        with col_feat:
            st.markdown(
                '<div class="form-section-title">3. Tiện ích & Thời điểm định giá</div>',
                unsafe_allow_html=True,
            )
            selected_amenities = st.multiselect(
                "Đặc điểm tiện ích nổi bật:",
                ["Có nội thất", "Hẻm ô tô", "Gần chợ", "Gần trường", "Bán gấp"],
                default=["Hẻm ô tô"],
            )

            # As-of date (Thời điểm định giá)
            st.markdown("**Thời điểm định giá (Valuation Date / As-of Date)**")
            valuation_date_input = st.date_input(
                "Chọn ngày định giá",
                value=date(2025, 9, 30),
                help=(
                    "Dữ liệu listing tương đồng chỉ lấy các tin có listing_date <= ngày này "
                    "và trong vòng 365 ngày để bảo đảm tính point-in-time trung thực."
                ),
            )
            st.caption(
                "📌 *Ý nghĩa:* Tin đăng tương đồng được trích xuất tại thời điểm này "
                "để ngăn ngừa look-ahead leakage (dùng tin đăng của tương lai)."
            )

        estimate_submitted = st.form_submit_button(
            "🔍 Thực Hiện Ước Lượng Giá Niêm Yết",
            type="primary",
            use_container_width=True,
        )

    if estimate_submitted:
        payload: dict[str, Any] = {
            "Property Type": prop_type,
            "location_area": prop_district,
            "Area": float(prop_area),
            "Bedrooms": int(prop_beds) if prop_beds > 0 else None,
            "Bathrooms": int(prop_baths) if prop_baths > 0 else None,
            "Floors": int(prop_floors) if prop_floors > 0 else None,
            "Width": float(prop_width) if prop_width > 0 else None,
            "Length": float(prop_length) if prop_length > 0 else None,
            "Alley Width": float(prop_alley) if prop_alley > 0 else None,
            "Direction": prop_direction,
            "Position": prop_position,
            "Latitude": float(prop_lat) if prop_lat > 0 else None,
            "Longitude": float(prop_lon) if prop_lon > 0 else None,
            "has_furniture": "Có nội thất" in selected_amenities,
            "car_alley": "Hẻm ô tô" in selected_amenities,
            "near_market": "Gần chợ" in selected_amenities,
            "near_school": "Gần trường" in selected_amenities,
            "is_urgent_sale": "Bán gấp" in selected_amenities,
            "as_of_date": valuation_date_input.isoformat(),
        }

        try:
            prediction_output = predict_one(payload, include_explanation=True)
            st.session_state["prediction_output"] = prediction_output
            st.session_state["submitted_payload"] = payload
        except (ValueError, FileNotFoundError, RuntimeError) as err:
            st.error(f"Không thể thực hiện ước lượng: {err}")

    # Hiển thị kết quả ước lượng nếu đã có trong session
    if "prediction_output" in st.session_state:
        res = st.session_state["prediction_output"]
        inp = st.session_state["submitted_payload"]

        price_million = float(res["predicted_price_million"])
        price_billion = price_million / 1000.0
        interval_data = res["prediction_interval"]
        lower_billion = float(interval_data["lower_million"]) / 1000.0
        upper_billion = float(interval_data["upper_million"]) / 1000.0
        unit_price_million = price_million / max(float(inp["Area"]), 1.0)
        market_age = res.get("market_age_days")

        st.divider()
        st.markdown("### 🎯 Kết Quả Ước Lượng Giá & Price Intelligence")

        # 4 KPI Cards
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-label">Estimated Listing Price</div>
                    <div class="kpi-value">{price_billion:,.2f} tỷ VND</div>
                    <div class="kpi-subtext">{price_million:,.1f} triệu VND (Point Estimate)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">80% Prediction Interval</div>
                    <div class="kpi-value">{lower_billion:,.2f} – {upper_billion:,.2f} tỷ</div>
                    <div class="kpi-subtext">Độ rộng: {upper_billion - lower_billion:,.2f} tỷ VND (Conformal)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Estimated Unit Price</div>
                    <div class="kpi-value">{unit_price_million:,.1f} tr/m²</div>
                    <div class="kpi-subtext">Tính trên diện tích {inp['Area']:.0f} m²</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Market Reference Age</div>
                    <div class="kpi-value">{market_age if market_age is not None else 'N/A'} ngày</div>
                    <div class="kpi-subtext">Mốc tham chiếu: {res.get('market_reference_date', 'N/A')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Trực quan hóa khoảng bất định Conformal (Uncertainty Range Bar)
        total_span = max(upper_billion - lower_billion, 0.01)
        point_offset_pct = max(
            min(((price_billion - lower_billion) / total_span) * 100.0, 100.0), 0.0
        )

        st.markdown(
            f"""
            <div class="uncertainty-box">
                <div style="display: flex; justify-content: space-between; font-weight: 700; color: #1E293B; font-size: 0.95rem;">
                    <span>Khoảng Bất Định Dự Kiến (Conformal Prediction Interval)</span>
                    <span style="color: #2563EB;">Mục tiêu bao phủ: 80% (Thực tế test: 72.7%)</span>
                </div>
                <div class="range-bar-track">
                    <div class="range-bar-fill" style="left: 0%; width: 100%;"></div>
                    <div class="point-marker" style="left: {point_offset_pct:.1f}%;" title="Ước lượng điểm: {price_billion:.2f} tỷ"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #64748B;">
                    <div>Cận dưới: <strong>{lower_billion:,.2f} tỷ VND</strong></div>
                    <div style="color: #0F172A; font-weight: 700;">● Ước lượng điểm: {price_billion:,.2f} tỷ VND</div>
                    <div>Cận trên: <strong>{upper_billion:,.2f} tỷ VND</strong></div>
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.6rem;">
                    * Lưu ý: Khoảng dự báo bất đối xứng do được hiệu chuẩn conformal trên log-scale và nghịch đảo qua hàm expm1.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Input Quality & Completeness Score
        input_quality = res.get("input_quality", {})
        comp_score = float(input_quality.get("completeness_score", 100.0))
        missing_fields = input_quality.get("missing_fields", [])

        st.markdown("#### 📋 Độ Đầy Đủ Thông Tin Đầu Vào (Input Completeness)")
        col_bar, col_desc = st.columns([3, 2])
        with col_bar:
            st.progress(
                min(max(comp_score / 100.0, 0.0), 1.0),
                text=f"Điểm hoàn thiện thông tin: {comp_score:.0f} / 100",
            )
        with col_desc:
            if missing_fields:
                missing_str = ", ".join(missing_fields)
                st.caption(f"Trường còn khuyết: `{missing_str}`")
            else:
                st.caption("✅ Đã cung cấp đầy đủ các trường đo lường chính.")

        # Warning Panel (Chức năng cốt lõi)
        warnings = res.get("warnings", [])
        if warnings:
            st.markdown("#### ⚠️ Cảnh Báo Dữ Liệu & Chất Lượng Phục Vụ (Warning Panel)")
            for w in warnings:
                render_formatted_warning(w)

        # Gợi ý sang tab Comparables
        comp_count = len(res.get("comparables", []))
        st.info(
            f"💡 Hệ thống đã truy vấn **{comp_count} tin đăng tương đồng** trong lịch sử để làm bằng chứng định giá. "
            "Chuyển sang tab **🔍 Comparable Evidence** để xem so sánh chi tiết."
        )


# ===========================================================================
# TAB 2: COMPARABLE EVIDENCE (Bằng Chứng Listing Tương Đồng)
# ===========================================================================
with tab_comparables:
    st.subheader("Bằng Chứng Thị Trường Từ Tin Đăng Lịch Sử (Comparable Evidence)")

    if "prediction_output" not in st.session_state:
        st.info(
            "Vui lòng thực hiện một lần ước lượng tại tab **🏠 Price Estimate** để xem các listing tương đồng."
        )
    else:
        res = st.session_state["prediction_output"]
        inp = st.session_state["submitted_payload"]
        comparables = res.get("comparables", [])
        comp_summary = res.get("comparable_summary") or {}

        model_price_billion = float(res["predicted_price_million"]) / 1000.0
        median_comp_price = comp_summary.get("median_price_million")
        median_comp_unit = comp_summary.get("median_unit_price_million_m2")

        # Quy trình truy vấn (Retrieval Pipeline)
        with st.expander("ℹ️ Nguyên lý hoạt động của Comparable Engine", expanded=False):
            st.markdown(
                """
                ```
                Thông tin BĐS cần định giá
                       ↓
                Lọc cùng loại hình bất động sản
                       ↓
                Chỉ lấy tin đăng trong quá khứ (listing_date <= as_of_date) & trong vòng 365 ngày
                       ↓
                Ưu tiên ứng viên có diện tích chênh lệch không quá ±25%
                       ↓
                Ưu tiên ứng viên trong cùng quận/huyện
                       ↓
                Xếp hạng đa chiều: Diện tích + GPS (Haversine) + Phòng ngủ + WC + Cự ly CBD + Độ mới (Recency)
                       ↓
                Trích xuất Top 4 tin đăng tương đồng nhất làm bằng chứng tham chiếu
                ```
                *Comparable evidence là bằng chứng dữ liệu tham khảo, **không phải** mô hình ước lượng thứ hai và không tự động thay đổi kết quả của Machine Learning.*
                """
            )

        # So sánh Model Estimate với Comparable Median
        st.markdown("#### ⚖️ Đối So sánh Mô hình ML vs Trung vị Tin Tương Đồng")
        c_comp1, c_comp2, c_comp3, c_comp4 = st.columns(4)

        with c_comp1:
            st.metric(
                label="Model Estimate",
                value=f"{model_price_billion:,.2f} tỷ VND",
                help="Giá trị trung tâm do mô hình ExtraTrees đưa ra.",
            )

        with c_comp2:
            val_str = (
                f"{median_comp_price / 1000.0:,.2f} tỷ VND"
                if median_comp_price
                else "N/A"
            )
            st.metric(
                label="Comparable Median",
                value=val_str,
                help="Trung vị giá của các tin đăng tương đồng tìm được.",
            )

        with c_comp3:
            if median_comp_price and model_price_billion > 0:
                diff_pct = (
                    (model_price_billion - (median_comp_price / 1000.0))
                    / (median_comp_price / 1000.0)
                ) * 100.0
                st.metric(
                    label="Độ lệch (Model vs Median)",
                    value=f"{diff_pct:+.1f}%",
                    help="Độ chênh lệch tương đối giữa mô hình và mức trung vị tin đăng tương đồng.",
                )
            else:
                st.metric(label="Độ lệch (Model vs Median)", value="N/A")

        with c_comp4:
            st.metric(
                label="Comparable Median Đơn Giá",
                value=f"{median_comp_unit:,.1f} tr/m²" if median_comp_unit else "N/A",
                help="Đơn vị tính trên m² của các tin đăng tham chiếu.",
            )

        st.divider()

        # Hiển thị 4 thẻ Comparable Cards
        if comparables:
            st.markdown("#### 📑 Danh Sách Các Tin Đăng Tương Đồng (Top 4 Matches)")
            card_cols = st.columns(len(comparables))

            for idx, (col, comp) in enumerate(
                zip(card_cols, comparables, strict=False)
            ):
                sim_score = comp.get("similarity_score", 1.0)
                sim_pct = int(round(sim_score * 100))
                p_million = comp.get("price_million")
                p_str = f"{p_million / 1000.0:,.2f} tỷ" if p_million else "Chưa có"
                u_str = (
                    f"{comp.get('unit_price_million_m2', 0):,.1f} tr/m²"
                    if comp.get("unit_price_million_m2")
                    else "N/A"
                )
                cbd_str = (
                    f"{comp.get('distance_to_cbd_km', 0):.1f} km"
                    if comp.get("distance_to_cbd_km") is not None
                    else "N/A"
                )

                with col:
                    st.markdown(
                        f"""
                        <div class="comp-card">
                            <span class="comp-badge">Match #{idx + 1} · Độ tương đồng {sim_pct}%</span>
                            <div class="comp-title">{comp.get('location_area', 'TP.HCM')}</div>
                            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.4rem;">
                                {comp.get('property_type', 'Nhà riêng')}
                            </div>
                            <div class="comp-price">{p_str}</div>
                            <div class="comp-metric">📐 Diện tích: <strong>{comp.get('area', 'N/A')} m²</strong></div>
                            <div class="comp-metric">🏢 Đơn giá: <strong>{u_str}</strong></div>
                            <div class="comp-metric">🛏️ Kết cấu: {comp.get('bedrooms', '-')} PN · {comp.get('bathrooms', '-')} WC</div>
                            <div class="comp-metric">🚗 Cự ly CBD: {cbd_str}</div>
                            <div class="comp-metric" style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.4rem;">
                                📅 Ngày đăng: {comp.get('listing_date', 'N/A')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Bảng chi tiết toàn bộ các cột
            with st.expander("📊 Xem bảng dữ liệu chi tiết của các tin đăng"):
                comp_df = pd.DataFrame(comparables)
                col_rename = {
                    "property_type": "Loại BĐS",
                    "location_area": "Khu vực",
                    "area": "Diện tích (m²)",
                    "price_million": "Giá (triệu VND)",
                    "unit_price_million_m2": "Đơn giá (tr/m²)",
                    "bedrooms": "PN",
                    "bathrooms": "WC",
                    "floors": "Số tầng",
                    "distance_to_cbd_km": "Cách CBD (km)",
                    "similarity_score": "Độ tương đồng",
                    "listing_date": "Ngày đăng tin",
                }
                valid = [c for c in col_rename if c in comp_df.columns]
                st.dataframe(
                    comp_df[valid].rename(columns=col_rename),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("Không tìm thấy tin đăng tương đồng nào phù hợp trong quá khứ.")


# ===========================================================================
# TAB 3: WHY THIS ESTIMATE? (Giải Thích Mô Hình)
# ===========================================================================
with tab_why:
    st.subheader("Tại Sao Mô Hình Đưa Ra Mức Giá Này? (Why This Estimate?)")

    if "prediction_output" not in st.session_state:
        st.info(
            "Vui lòng thực hiện ước lượng giá tại tab **🏠 Price Estimate** để xem các yếu tố đóng góp của mô hình."
        )
    else:
        res = st.session_state["prediction_output"]
        contributions = res.get("top_contributions", [])

        if contributions:
            st.markdown(
                """
                Bảng dưới đây hiển thị 5 đặc trưng có trọng số đóng góp lớn nhất vào dự báo hiện tại của mô hình 
                (được tính bằng thuật toán SHAP TreeExplainer trên thang log-target).
                """
            )

            contrib_df = pd.DataFrame(contributions)
            # Tạo nhãn trực quan với mũi tên tăng/giảm
            labels = []
            signs = []
            for _, row in contrib_df.iterrows():
                val = float(row["shap_value"])
                name = row.get("friendly_name") or row["feature"]
                if val >= 0:
                    labels.append(f"↑ {name}")
                    signs.append("Tăng giá (+)")
                else:
                    labels.append(f"↓ {name}")
                    signs.append("Giảm giá (-)")

            contrib_df["Yếu tố"] = labels
            contrib_df["Chiều hướng"] = signs
            contrib_df["Đóng góp SHAP (Log-Scale)"] = contrib_df["shap_value"]

            c_chart, c_table = st.columns([3, 2])

            with c_chart:
                chart_data = contrib_df.set_index("Yếu tố")["Đóng góp SHAP (Log-Scale)"]
                st.bar_chart(chart_data, color="#2563EB")

            with c_table:
                display_cols = [
                    "Yếu tố",
                    "Chiều hướng",
                    "Đóng góp SHAP (Log-Scale)",
                ]
                st.dataframe(
                    contrib_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            # Disclaimer quan trọng: SHAP không phải quan hệ nhân quả
            st.markdown(
                """
                <div class="alert-card warning" style="margin-top: 1.5rem;">
                    <div>
                        <div class="alert-title">⚠️ Quan Trọng: SHAP KHÔNG PHẢI QUAN HỆ NHÂN QUẢ (Causality)</div>
                        <div>
                            Giá trị đóng góp SHAP chỉ giải thích cách thuật toán tổng hợp các nhánh cây quyết định thống kê để ra con số dự báo cho trường hợp này.<br>
                            <strong>Hệ thống không khẳng định quan hệ nhân quả thực tế:</strong> Việc thêm 1 phòng ngủ không đồng nghĩa với việc giá nhà ngoài đời thực sẽ tăng cố định một khoản tiền; mở rộng hẻm lên 5m không bảo đảm giá thực tăng theo tỷ lệ tương ứng.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Mô hình hiện tại không trả về giá trị giải thích đặc trưng cho lần dự báo này."
            )


# ===========================================================================
# TAB 4: MODEL & DATA (Minh Bạch Kỹ Thuật & Giới Hạn)
# ===========================================================================
with tab_model:
    st.subheader("Minh Bạch Kỹ Thuật, Hiệu Năng Mô Hình & Giới Hạn Nghiên Cứu")

    st.markdown(
        """
        Dự án tuân thủ nguyên tắc minh bạch tối đa: không quảng cáo quá mức về độ chính xác,
        công khai sai số trên tập kiểm tra độc lập và thừa nhận khoảng bất định rộng của dữ liệu niêm yết.
        """
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### ⚙️ Cấu Hình Mô Hình & Gói Artifacts")
        st.write(f"- **Thuật toán chính**: `{MODEL_TYPE}` (ExtraTreesRegressor)")
        st.write(f"- **Phiên bản mô hình**: `v{MODEL_VERSION}`")
        st.write(
            f"- **Giao thức chia mẫu**: `{_ui_model_package.get('split_protocol', 'group_isolated_temporal_split')}`"
        )
        st.write(
            f"- **Mục tiêu Conformal**: `{_ui_model_package.get('target_coverage', 0.8) * 100:.0f}%`"
        )
        st.write(
            f"- **Phân vị sai số Residual Quantile**: `{_ui_model_package.get('residual_log_quantile', 0.0):.4f}` (log-scale)"
        )

    with col_m2:
        st.markdown("#### 🛡️ Nguyên Tắc Chống Rò Rỉ Dữ Liệu (Anti-Leakage)")
        st.markdown(
            """
            1. **Nhận diện BĐS trước chia tập (Identity Isolation)**: Cùng một `property_group_id` (kể cả tin đăng lại) không bao giờ nằm ở hai phía của split.
            2. **FeatureContext Train-only**: Các tham số thống kê và chuẩn hóa chỉ được tính từ tập Huấn luyện.
            3. **Calibration & Test riêng biệt**: Conformal Calibration chạy trên tập riêng chiếm 10% số nhóm; Final Test chỉ dùng để báo cáo kiểm thử độc lập.
            """
        )

    st.divider()

    # So sánh hiệu năng thực tế trên Final Test với các mô hình cơ sở (Baselines)
    st.markdown(
        "#### 📊 Hiệu Năng Trên Tập Kiểm Tra Độc Lập (Final Group-Temporal Test)"
    )
    st.caption(
        "Tập kiểm tra bao gồm 110 mẫu độc lập diễn ra sau mốc thời gian huấn luyện."
    )

    metric_table = pd.DataFrame(
        [
            {
                "Mô hình": "Champion (ExtraTrees)",
                "MAE (Triệu VND)": "4,928.7",
                "Median AE (Triệu)": "2,478.7",
                "RMSE (Triệu)": "8,168.7",
                "R² Score": "0.081",
                "WAPE": "45.5%",
                "MAPE": "42.5%",
            },
            {
                "Mô hình": "District Segment Median",
                "MAE (Triệu VND)": "6,068.2",
                "Median AE (Triệu)": "2,850.0",
                "RMSE (Triệu)": "9,939.6",
                "R² Score": "-0.360",
                "WAPE": "56.0%",
                "MAPE": "50.4%",
            },
            {
                "Mô hình": "Naive Median",
                "MAE (Triệu VND)": "6,332.0",
                "Median AE (Triệu)": "3,150.0",
                "RMSE (Triệu)": "10,142.4",
                "R² Score": "-0.416",
                "WAPE": "58.5%",
                "MAPE": "53.7%",
            },
        ]
    )
    st.table(metric_table.set_index("Mô hình"))

    st.markdown(
        """
        > **Nhận xét khách quan:** Mô hình Machine Learning có sự cải thiện rõ rệt so với các mô hình cơ sở 
        (giảm MAE hơn 1.4 tỷ VND so với Naive Median, WAPE giảm từ 58.5% xuống 45.5%). Tuy nhiên, sai số tuyệt đối vẫn còn đáng kể 
        (MAE ~4.93 tỷ, sai số trung vị ~2.48 tỷ) do bản chất dữ liệu tin đăng cá nhân chứa nhiều yếu tố nhiễu và kỳ vọng chủ quan của người bán.
        """
    )

    st.divider()

    # Chất lượng khoảng dự báo Conformal
    st.markdown("#### 🎯 Chất Lượng Khoảng Dự Báo Conformal Prediction")
    c_int1, c_int2, c_int3 = st.columns(3)
    with c_int1:
        st.metric(label="Mục tiêu bao phủ (Target Coverage)", value="80.0%")
    with c_int2:
        st.metric(
            label="Bao phủ thực tế quan sát (Observed)",
            value="72.7%",
            delta="-7.3% gap",
            delta_color="inverse",
        )
    with c_int3:
        st.metric(label="Độ rộng trung bình khoảng", value="10.49 tỷ VND")

    st.caption(
        "Khoảng dự báo Conformal không phải là khoảng cam kết 80% tuyệt đối. "
        "Mức độ bao phủ thực tế trên tập test đạt 72.7%; độ rộng khoảng tương đối lớn (~96.8%) "
        "phản ánh đúng mức độ bất định cố hữu của phân khúc nhà ở tại TP.HCM."
    )

    st.divider()

    # Visual sơ đồ Temporal Split Protocol
    st.markdown("#### ⏳ Sơ Đồ Chia Tập Dữ Liệu Theo Thời Gian (Temporal Split Flow)")
    st.markdown(
        """
        ```
        Quá khứ --------------------------------------------------------> Tương lai

        TẬP HUẤN LUYỆN (TRAIN - 60%, 436 mẫu)
        ████████████████████████████████████████████

        TẬP XÁC THỰC (VALIDATION - 15%, 109 mẫu)
                                                    ███████████

        TẬP HIỆU CHUẨN (CALIBRATION - 10%, 72 mẫu)
                                                               ███████

        TẬP KIỂM TRA ĐỘC LẬP (FINAL TEST - 15%, 110 mẫu)
                                                                      ███████████
        ```
        *Các tin đăng thuộc cùng một căn nhà (cùng `property_group_id`) luôn được gom trọn vẹn vào một tập duy nhất theo ngày đăng muộn nhất.*
        """
    )
