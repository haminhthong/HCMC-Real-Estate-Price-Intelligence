"""Dashboard Streamlit cho dự báo giá và kiểm tra listing tương đồng."""

import pandas as pd
import streamlit as st

from src.artifacts.loader import load_model
from src.config import RESIDENTIAL_TYPES, SUPPORTED_AREAS
from src.serving.predictor import predict_one

try:
    _ui_model_package = load_model()
except (FileNotFoundError, KeyError, ValueError):
    _ui_model_package = {}

SUPPORTED_MODEL_TYPES = (
    _ui_model_package.get("supported_property_types") or RESIDENTIAL_TYPES
)
SUPPORTED_MODEL_AREAS = _ui_model_package.get("supported_areas") or SUPPORTED_AREAS

# ---------------------------------------------------------------------------
# Cấu hình trang Streamlit và kiểu hiển thị
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HCMC Real Estate Price Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Kiểu CSS riêng cho giao diện
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tiêu đề chính
st.markdown(
    '<div class="main-header">🏠 HCMC Real Estate Price Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Hệ thống ước lượng giá đăng bất động sản dân dụng tại TP.HCM dựa trên Machine Learning & Conformal Prediction</div>',
    unsafe_allow_html=True,
)

# Tạo các thẻ chức năng
tab_predict, tab_shap, tab_info = st.tabs(
    [
        "🏠 Dự Báo Giá",
        "📊 Phân Tích SHAP & Thị Trường",
        "ℹ️ Thông Tin Mô Hình & Dữ Liệu",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1: DỰ BÁO GIÁ
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Nhập thông tin bất động sản cần định giá")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Thông tin cơ bản**")
            property_type = st.selectbox("Loại bất động sản (*)", SUPPORTED_MODEL_TYPES)
            area_name = st.selectbox(
                "Quận/huyện khu vực (*)",
                [area for area in SUPPORTED_MODEL_AREAS if area != "Unknown"],
            )
            area = st.number_input(
                "Diện tích đất/sử dụng (m²) (*)", 5.0, 500.0, 80.0, step=5.0
            )
            bedrooms = st.number_input("Số phòng ngủ (0 = chưa cung cấp)", 0, 10, 3)

        with col2:
            st.markdown("**Thông số kích thước & kết cấu**")
            bathrooms = st.number_input("Số phòng vệ sinh", 0, 20, 2)
            floors = st.number_input("Số tầng", 0, 100, 2)
            width = st.number_input(
                "Chiều rộng mặt tiền (m)", 0.1, 100.0, 4.0, step=0.5
            )
            length = st.number_input(
                "Chiều dài / chiều sâu (m)", 0.1, 200.0, 20.0, step=1.0
            )

        with col3:
            st.markdown("**Vị trí & Tiện ích**")
            alley = st.number_input(
                "Độ rộng hẻm trước nhà (m)", 0.0, 30.0, 3.0, step=0.5
            )
            direction = st.selectbox(
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
            position = st.selectbox(
                "Vị trí",
                ["Không rõ", "Trong hẻm", "Đường chính"],
            )

        st.markdown("**Đặc điểm nổi bật / Cờ tiện ích**")
        amenities = st.multiselect(
            "Chọn các tiện ích đi kèm:",
            ["Có nội thất", "Hẻm ô tô", "Gần chợ", "Gần trường", "Bán gấp"],
            default=["Hẻm ô tô"],
        )

        submitted = st.form_submit_button(
            "🔍 Thực Hiện Định Giá", type="primary", use_container_width=True
        )

    if submitted:
        payload = {
            "Property Type": property_type,
            "location_area": area_name,
            "Area": area,
            "Bedrooms": bedrooms if bedrooms > 0 else None,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "Width": width,
            "Length": length,
            "Alley Width": alley,
            "Direction": direction,
            "Position": position,
            "has_furniture": "Có nội thất" in amenities,
            "car_alley": "Hẻm ô tô" in amenities,
            "near_market": "Gần chợ" in amenities,
            "near_school": "Gần trường" in amenities,
            "is_urgent_sale": "Bán gấp" in amenities,
        }

        try:
            result = predict_one(payload, include_explanation=True)
            st.session_state["last_result"] = result

            st.divider()
            st.subheader("🎯 Kết Quả Ước Lượng Giá & Price Intelligence")

            res_c1, res_c2, res_c3 = st.columns(3)

            price_billion = result["predicted_price_million"] / 1000
            interval = result["prediction_interval"]
            lower_billion = interval["lower_million"] / 1000
            upper_billion = interval["upper_million"] / 1000

            res_c1.metric(
                label="Giá Ước Tính (Point Estimate)",
                value=f"{price_billion:,.2f} tỷ VND",
                help="Giá trị trung tâm được dự báo từ mô hình Machine Learning.",
            )

            res_c2.metric(
                label="Khoảng Dự Báo (Prediction Interval 80%)",
                value=f"{lower_billion:,.2f} – {upper_billion:,.2f} tỷ",
                help="Mức bao phủ mục tiêu là 80%; mức thực tế được đo trên tập kiểm tra.",
            )

            res_c3.metric(
                label="Số cảnh báo dữ liệu",
                value=str(len(result.get("warnings", []))),
                help="Cảnh báo được tạo từ input, tuổi dữ liệu và độ rộng khoảng dự báo.",
            )

            # Cảnh báo nếu có
            if result["warnings"]:
                with st.expander("⚠️ Cảnh báo dữ liệu và khoảng dự báo", expanded=True):
                    for warning in result["warnings"]:
                        st.warning(warning)

            # Bảng so sánh bất động sản tương đồng
            comparables = result.get("comparables", [])
            if comparables:
                comp_df = pd.DataFrame(comparables)
                display_cols = {
                    "property_type": "Loại hình",
                    "location_area": "Khu vực",
                    "area": "Diện tích (m²)",
                    "price_million": "Giá (triệu VND)",
                    "unit_price_million_m2": "Đơn giá (tr/m²)",
                    "bedrooms": "PN",
                    "bathrooms": "WC",
                    "distance_to_cbd_km": "Cách CBD (km)",
                    "similarity_score": "Độ tương đồng",
                }
                valid_cols = [c for c in display_cols if c in comp_df.columns]
                comp_display = comp_df[valid_cols].rename(columns=display_cols)
                st.dataframe(comp_display, use_container_width=True, hide_index=True)

            st.caption(f"📌 *{result['disclaimer']}*")

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(f"Đã xảy ra lỗi trong quá trình dự báo: {exc}")

# ---------------------------------------------------------------------------
# TAB 2: PHÂN TÍCH SHAP & THỊ TRƯỜNG
# ---------------------------------------------------------------------------
with tab_shap:
    st.subheader("Giải Thích Mô Hình & Mức Độ Đóng Góp Đặc Trưng (SHAP Values)")
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        contributions = result.get("top_contributions", [])

        if contributions:
            chart_df = pd.DataFrame(contributions)
            # Ưu tiên sử dụng nhãn tiếng Việt thân thiện
            if "friendly_name" in chart_df.columns:
                chart_df["Đặc Trưng"] = chart_df["friendly_name"]
            else:
                chart_df["Đặc Trưng"] = chart_df["feature"]

            chart_df["Tác Động SHAP (Log-Scale)"] = chart_df["shap_value"]

            st.markdown(
                "Đồ thị dưới đây hiển thị 5 yếu tố có mức độ đóng góp cao nhất vào ước lượng của mô hình (trong không gian log-target). "
                "Lưu ý: Giá trị SHAP biểu thị đóng góp thống kê của mô hình, không phải quan hệ nhân quả tuyệt đối."
            )

            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                st.bar_chart(
                    chart_df.set_index("Đặc Trưng")["Tác Động SHAP (Log-Scale)"],
                    color="#2563EB",
                )

            with col_table:
                display_shap = chart_df[["Đặc Trưng", "Tác Động SHAP (Log-Scale)"]]
                st.dataframe(display_shap, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có dữ liệu SHAP.")
    else:
        st.info(
            "Vui lòng thực hiện một lần định giá tại Tab '🏠 Dự Báo Giá' để xem phân tích SHAP chi tiết."
        )

# ---------------------------------------------------------------------------
# TAB 3: THÔNG TIN MÔ HÌNH & DỮ LIỆU
# ---------------------------------------------------------------------------
with tab_info:
    st.subheader("Thông Tin Mô Hình & Luồng Dữ Liệu")
    try:
        model_package = load_model()
        c_info1, c_info2 = st.columns(2)

        with c_info1:
            st.markdown("### 🛠️ Cấu hình mô hình")
            st.write(f"- **Phiên bản mô hình**: `{model_package['version']}`")
            st.write(f"- **Thuật toán chính**: `{model_package['model_type']}`")
            st.write(f"- **Split**: `{model_package['split_protocol']}`")
            st.write(
                f"- **Mục tiêu bao phủ Conformal**: `{model_package['target_coverage'] * 100:.0f}%`"
            )

        with c_info2:
            st.markdown("### 🛡️ Nguyên Lý Chống Data Leakage")
            st.markdown(
                """
                1. **Identity trước split**: cùng `property_group_id` không xuyên qua các tập.
                2. **FeatureContext train-only**: preprocessing được fit trước trên Train.
                3. **Conformal interval**: Calibration riêng, Final Group-Temporal Test chỉ dùng để báo cáo.
                """
            )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.error(f"Chưa thể tải thông tin mô hình: {exc}")
