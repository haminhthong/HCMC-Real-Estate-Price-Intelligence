# HCMC Residential Listing Price Intelligence Platform 🏠

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-F7931E?logo=scikit-learn)
![Pytest](https://img.shields.io/badge/Pytest-Passing-0A9EDC?logo=pytest)
![Production Readiness](https://img.shields.io/badge/Status-Research--Only-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Platform Positioning & Scope Definition**:  
> **HCMC Residential Listing Price Intelligence Platform** là hệ thống ước lượng **giá niêm yết tham khảo (asking/listing price)** cho bất động sản nhà ở tại TP.HCM. Thay vì chỉ trả về một con số giá đơn độc ("AI định giá chính xác căn nhà"), nền tảng cung cấp một bộ giải pháp định giá đa chiều: **Ước lượng giá (Valuation) + Khoảng tin cậy Conformal (Uncertainty) + Bối cảnh thị trường (Market Context) + Bất động sản tham chiếu (Comparables) + Rào chắn độ tin cậy (Domain & Reliability Guardrails)**.

---

## 1. Problem Framing & Real Estate Boundaries

Hệ thống được thiết kế để giải quyết bài toán định giá tham khảo cho bất động sản nhà ở dân dụng tại TP.HCM (Nhà riêng, Nhà mặt tiền, Căn hộ chung cư, Biệt thự liền kề) dựa trên nguồn tin rao trực tuyến:

- **Bản chất dữ liệu tin rao**: Nguồn thu thập từ các sàn bất động sản là **giá niêm yết / giá chào bán (asking/listing price)**, **không phải giá giao dịch chốt thực tế (transacted price)**. Dữ liệu thực tế mang tính phân tán cao, nhiều tin trùng lặp từ nhiều môi giới, phân phối giá lệch phải nặng (long-tail skew) và phương sai lớn ở phân khúc cao cấp.
- **Tính minh bạch về sai số (No Masking Metrics)**:  
  Mô hình hiện ở trạng thái **`research_only`**:
  - Test MAE: **4.62 tỷ VND**
  - Test WAPE: **42.92%**
  - Test $R^2$: **0.134**
  - Conformal Interval (80% target): Đạt coverage thực tế **84.4%**, nhưng bề rộng trung bình khoảng dự báo lên tới **12.62 tỷ VND** (độ rộng tương đối $\approx 117\%$).  
  *Chính mức độ bất định cố hữu này là lý do cốt lõi hệ thống bắt buộc phải có Reliability Guards + Comparables Context + Conformal Interval thay vì chỉ trả về một con số dự báo điểm đơn thuần.*

### 📌 Phân Định Biên Giới Sử Dụng (In-Scope vs Out-of-Scope)

| Phù Hợp Sử Dụng (In-Scope) | KHÔNG Dùng Cho (Out-of-Scope) |
| :--- | :--- |
| ✅ **Người mua nhà**: Tham khảo khoảng dao động giá hợp lý và bối cảnh các căn tương đồng trước khi đàm phán. | ❌ Thẩm định giá pháp lý để cấp tín dụng / thế chấp ngân hàng |
| ✅ **Người bán / Môi giới**: Đối chiếu giá niêm yết dự kiến với mặt bằng trung vị phân khúc và các căn tương đồng lịch sử. | ❌ Giám định tranh chấp tài sản, phân chia thừa kế trước tòa án |
| ✅ **Nhà phân tích dữ liệu**: Theo dõi biến động đơn giá trung vị (triệu VND/m²) theo quận/huyện và cự ly tới CBD. | ❌ Thuật toán tự động hóa đặt lệnh đầu tư tài chính phái sinh |

---

## 2. Canonical Pipelines

### A. Offline Development Pipeline (11 Steps)

```text
                 RAW PROPERTY LISTINGS (2,500 listings)
                         ↓
1. DATA INGESTION & SNAPSHOT
   ├── source: data_public_sample.csv
   ├── schema validation & types
   └── snapshot manifest
                         ↓
2. DATA QUALITY & AUDIT
   ├── predefined target-aware market scope rules:
   │   • Price ∈ [100M, 50B VND] (phạm vi thị trường hỗ trợ)
   │   • Area ∈ [5, 500 m²], Đơn giá >= 10M/m²
   └── supported residential types (727 valid listings)
                         ↓
3. PROPERTY IDENTITY RESOLUTION
   ├── Multi-level matching:
   │   • Level 1 Strong: Exact normalized address + GPS within tolerance
   │   • Level 2 Medium: Location + Property Type + Area (±5%) + Compatible Beds/Baths
   │   • Level 3 Weak: Area block + Structural similarity
   └── Union-Find clustering → property_group_id (707 unique groups)
                         ↓
4. LISTING DEDUPLICATION
   ├── Drop exact duplicate listings (subset: property_group_id, date, Price): -4 rows
   └── Preserve legitimate relistings (723 clean listings across 707 groups)
                         ↓
5. GROUP-ISOLATED TEMPORAL ORDERING SPLIT
   Property groups ordered by latest listing date (60% / 15% / 10% / 15%)
      ↓
   Train (436 rows, 424 groups)
   Validation (106 rows, 106 groups)
   Calibration (72 rows, 70 groups)
   Locked Test (109 rows, 107 groups)
                         ↓
6. SHARED FEATURE CONTRACT
   ├── Structural: Area, Width, Length, Bedrooms, Bathrooms, Floors, Alley Width
   ├── Geospatial: Lat, Lon, distance_to_cbd_km
   ├── Text Signals: Regex heuristics with Vietnamese negation handling
   ├── Missingness Indicators: *_missing boolean features
   └── Temporal Contract: as_of_date / market_time_offset_days (no clip lower=0)
                         ↓
7. MODEL DEVELOPMENT & CANDIDATE BENCHMARK
   ├── Naive Global Median
   ├── District x Type Segment Median
   ├── Ridge Regression
   ├── Random Forest
   ├── HistGradientBoosting
   └── ExtraTrees Regressor
                         ↓
   Validation-Only Selection (Development Gate: ExtraTrees on Total Price)
                         ↓
8. FINAL REFIT
   Refit Champion on Train + Validation (542 listings)
                         ↓
9. UNCERTAINTY CALIBRATION
   Calibration Set (72 listings) → Split Conformal in Log-Residual Space
                         ↓
10. RELEASE EVALUATION
   Locked Test Set (109 listings)
   ├── Metrics: MAE, Median AE, RMSE, R², MAPE, WAPE, sMAPE
   ├── Conformal: Actual Coverage (84.4%), Mean/Median Width
   ├── Granular Slices: Price Tier, Completeness, District, CBD Distance
   └── Release Gate Evaluation (Decision: research_only)
                         ↓
11. VERSIONED ARTIFACT GENERATION
   Model (.joblib) + Feature Context + Calibration Residuals
   + Comparable Context + Metrics + Manifests
```

### B. Online Serving Architecture (5-Pillar Structured Intelligence)

```text
Property Request (JSON) with optional as_of_date
      ↓
Input Contract Validation (Pydantic)
      ↓
Shared Feature Builder (as_of_date → market_time_offset_days, completeness score)
      ↓
Domain & Reliability Guardrails
(P01–P99 Quantiles, Low-Support Segment, P90 Luxury Tier, Temporal Extrapolation)
      ↓
Champion Model (ExtraTrees)
      ↓
Point Listing-Price Estimate (Triệu VND)
      ↓
Asymmetric Conformal Prediction Interval (80% target coverage)
      ↓
Multi-dimensional Comparable Property Retrieval (Top-K comps from Train Index)
      ↓
Market Context (Segment Median, Comparable Median)
      ↓
SHAP TreeExplainer (Feature contributions on log-target)
      ↓
Structured Intelligence Response
┌──────────────────────────────────────────────────────────┐
│ 1. VALUATION       : Point estimate (Triệu VND)          │
│ 2. UNCERTAINTY     : Asymmetric 80% prediction interval  │
│ 3. MARKET CONTEXT  : Segment & comp median benchmarks    │
│ 4. COMPARABLES     : Top-5 nearest historical listings   │
│ 5. RELIABILITY     : Multi-factor risk & guard warnings  │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Data Audit, Target-Aware Scope & Multi-Level Identity

### A. Bảng Kiểm Toán Truy Nguyên Dữ Liệu (Data Audit Trail)

Toàn bộ chỉ số được trích xuất trực tiếp từ artifact kiểm định `artifacts/data_card.json`:

```text
2,500 Raw Listings (data_public_sample.csv)
   │
   ├─► Loại bỏ loại hình không hỗ trợ (Đất nền, kho xưởng, mặt bằng...): -1,587 rows
   │
   ├─► Predefined Target-Aware Validity Rules (Phạm vi thị trường): -186 rows
   │   • Price ∉ [100 triệu, 50 tỷ VND]
   │   • Area ∉ [5 m², 500 m²]
   │   • Đơn giá < 10 triệu/m²
   │   • Kích thước / phòng ngủ / số tầng phi thực tế
   │   • Tọa độ ngoài ranh giới TP.HCM (Lat: 10.3–11.2, Lon: 106.3–107.0)
   │   (Ngưỡng được chốt cố định theo nghiệp vụ trước thử nghiệm, không tinh chỉnh trên Val/Test)
   │
   ├─► Valid Listings: 727 bản ghi
   │
   ├─► Listing Deduplication: -4 bản ghi trùng lặp tuyệt đối (cùng property_group, ngày đăng, giá)
   │
   └─► 723 Clean Listings
       │
       ▼
   707 Inferred Unique Property Groups (15 nhóm có nhiều tin đăng, lớn nhất 3 tin/nhóm, 692 singletons)
```

### B. Giải Quyết Danh Tính Tài Sản (Multi-Level Property Identity Resolution)

Thay vì băm chuỗi cứng (exact hash heuristic) dễ gây **false merge** (khi thuộc tính khuyết thiếu cao: Width thiếu 53%, Length thiếu 65%, Bathrooms thiếu 49%, GPS thiếu 73%), hệ thống ứng dụng mô hình phân giải danh tính 3 tầng kết hợp giải thuật **Union-Find**:

1. **Level 1 — Strong Match**:
   - Cùng địa chỉ chuẩn hóa (normalized address) và tọa độ GPS trong bán kính dung sai ($\approx 20\,\text{m}$).
2. **Level 2 — Medium Match**:
   - Cùng khu vực (location_area / quận), cùng loại hình, diện tích chênh lệch $\le 5\%$, số phòng ngủ/vệ sinh tương thích.
3. **Level 3 — Weak Match**:
   - Cùng phân khu địa lý, tương đồng kết cấu cao và độ tương đồng văn bản/mô tả cao.

Mỗi nhóm tài sản được gắn nhãn `identity_confidence` (`strong`, `medium`, `weak`, `singleton`), cho phép kiểm toán minh bạch:
- Số nhóm bất động sản: **707**
- Nhóm có $>1$ tin đăng: **15 nhóm** (14 nhóm 2 tin, 1 nhóm 3 tin)
- Nhóm lớn nhất: **3 tin đăng**

---

## 4. Group-Isolated Temporal Ordering Split Semantics

Dữ liệu được phân chia theo quy tắc **Group-Isolated Temporal Ordering Split**:
1. Nhóm toàn bộ 723 tin đăng theo 707 `property_group_id`.
2. Lấy ngày đăng mới nhất của mỗi nhóm (`latest listing_date per group`).
3. Sắp xếp 707 nhóm theo thứ tự thời gian tăng dần và phân bổ theo tỷ lệ **60% / 15% / 10% / 15%**.
4. Toàn bộ các tin đăng của cùng một nhóm tài sản bắt buộc nằm trọn vẹn trong cùng một tập dữ liệu.

### Bảng Phân Bổ Mẫu và Nhóm Tài Sản

| Tập Dữ Liệu | Tỷ Lệ Nhóm | Số Nhóm Tài Sản (`property_group_id`) | Số Lượng Bản Ghi (`rows`) | Vai Trò Kỹ Thuật |
| :--- | :---: | :---: | :---: | :--- |
| **Train Set** | 60% | **424** | **436** | Fit tham số mô hình, feature encoders, reference index |
| **Validation Set** | 15% | **106** | **106** | Đánh giá & lựa chọn Champion Model (Development Gate) |
| **Calibration Set** | 10% | **70** | **72** | Hiệu chuẩn khoảng bất định (Split Conformal Residuals) |
| **Locked Test Set** | 15% | **107** | **109** | Đánh giá độc lập một lần duy nhất (Release Gate) |
| **Tổng Cộng** | 100% | **707** | **723** | Bảo đảm 100% không rò rỉ nhóm giữa các tập |

> ⚠️ **Làm Rõ Ngữ Nghĩa (Important Semantic Note)**:  
> Đây là **Group-isolated temporal ordering split**, không phải *strict row-level temporal split*. Nếu một căn nhà đăng tin lần đầu vào tháng 5 và cập nhật lần cuối vào tháng 9, toàn bộ lịch sử đăng của căn nhà đó sẽ đi theo mốc tháng 9 vào tập sau. Thiết kế này ưu tiên tối thượng việc **chống rò rỉ thông tin cùng tài sản giữa Train và Test**.

### Thử Nghiệm Kiểm Soát Trượt Thời Gian (Protocol B: Strict Temporal Holdout)

Để đo lường tác động của **Market Temporal Drift**, hệ thống cung cấp thêm giao thức `split_strict_temporal_purged()`:
- Đặt điểm cắt thời gian tuyệt đối ($T_{\text{cutoff}}$).
- Train $\le T_{\text{cutoff}}$, Test $> T_{\text{cutoff}}$.
- **Purging**: Loại bỏ triệt để mọi property group xuất hiện ở cả 2 khoảng thời gian khỏi tập Test.

---

## 5. Feature Contract & Missingness Indicators

Hệ thống tuân thủ hợp đồng đặc trưng nghiêm ngặt trong `src/features/builder.py`:

1. **Chỉ Báo Khuyết Thiếu Có Chủ Đích (Missingness Indicators)**:
   - Thay vì chỉ điền trung vị đơn thuần (làm mất thông tin về việc môi giới không khai báo), hệ thống tạo thêm 8 cờ nhị phân: `gps_missing`, `width_missing`, `length_missing`, `bedrooms_missing`, `bathrooms_missing`, `floors_missing`, `road_type_missing`, `alley_width_missing`.
2. **Chuẩn Hóa Temporal Contract (`as_of_date` / `market_time_offset_days`)**:
   - Tách biệt rõ giữa `listing_date` (ngày đăng lịch sử trong train data) và `as_of_date` / `valuation_date` (ngày người dùng yêu cầu định giá).
   - `market_time_offset_days = (as_of_date - train_reference_date).days`.
   - Không dùng hàm `.clip(lower=0)`. Nếu `as_of_date` cách xa dữ liệu huấn luyện (> 180 ngày), hệ thống kích hoạt cảnh báo `TEMPORAL_EXTRAPOLATION`.
3. **Độ Hoàn Thiện Dữ Liệu (`input_completeness_score`)**:
   - Sử dụng chuẩn mực `input_completeness_score` (đã loại bỏ hoàn toàn alias cũ `data_quality_score`).
4. **NLP Text Signals Có Xử Lý Từ Phủ Định (Negation Handling)**:
   - Nhận diện phủ định tiếng Việt (`không|chưa|chẳng|ko` + `có|nội thất|ô tô`).

---

## 6. Model Development & Validation Benchmark

Hệ thống so sánh 6 thuật toán cùng 2 cách đặt biến mục tiêu **chỉ trên tập Validation** (`artifacts/model_comparison.json`):

| Thuật Toán / Đường Cơ Sở | Biến Mục Tiêu | Val MAE (Triệu VND) | Val Median AE | Val RMSE | Val $R^2$ | Val MAPE | Val WAPE | Trạng Thái Lựa Chọn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Global Median** | Total Price | 3,801.1 | 2,200.0 | 6,814.6 | -0.135 | 53.25% | 49.69% | Baseline |
| **District x Type Segment Median** | Segment Unit | 3,759.4 | 2,212.5 | 6,415.2 | -0.006 | 57.52% | 49.14% | Baseline |
| **Ridge Linear Regression** | Total Price | 5,901.9 | 1,623.0 | 21,287.5 | -10.076 | 66.51% | 77.15% | Candidate |
| **Random Forest** | Total Price | 4,230.4 | 2,061.8 | 6,630.7 | -0.075 | 87.31% | 55.30% | Candidate |
| **HistGradientBoosting** | Total Price | 5,605.9 | 4,181.7 | 7,519.1 | -0.382 | 119.98% | 73.28% | Candidate |
| **ExtraTrees Regressor 🏆** | **Total Price** | **3,438.6** | **2,099.2** | **5,295.2** | **0.315** | **63.64%** | **44.95%** | **Selected Champion** |
| *ExtraTrees (Price / m²)* | Price / m² | 5,115.7 | 2,050.2 | 8,535.2 | -0.781 | 84.37% | 66.87% | Formulation B |

> **Quyết Định Lựa Chọn (Champion Selection)**:  
> **ExtraTrees Regressor trên Total Price** được lựa chọn làm Champion vì đạt Val MAE thấp nhất ($3,438.6$ triệu VND), vượt cả Naive Baseline lẫn Segment Baseline, và duy trì $R^2 = 0.315$.

---

## 7. Two-Gate Governance & Release Evaluation

Hệ thống phân tách rành mạch 2 cấp cổng quản trị để bảo vệ tính toàn vẹn của tập Test:

```text
       Validation Split Benchmark
                  ↓
       [ DEVELOPMENT GATE ]
       • Champion Val MAE < Baseline MAE (Cải thiện >= 10%)
       • Val WAPE <= 35%
                  ↓
         Pass? ──► Freeze Champion Model & Refit on Train+Val
                  ↓
       [ RELEASE GATE ] (Locked Test Split - Evaluated ONCE)
       • Test WAPE <= 30%
       • Conformal Coverage >= 75%
       • Interval Mean Width <= 10,000M
                  ↓
         Result: RESEARCH_ONLY (Không đủ điều kiện Autonomous Production)
```

> 🛡️ **Nguyên Tắc Bất Di Bất Dịch (Release Governance Rule)**:  
> Khi Release Gate trả về `research_only`, **nghiêm cấm tinh chỉnh siêu tham số mô hình dựa trên kết quả của tập Test**. Tập Test chỉ có chức năng phản ánh trung thực năng lực tổng quát hóa.

### Kết Quả Đánh Giá Trên Tập Locked Test (109 Listings)

Số liệu từ `artifacts/metrics.json` và `artifacts/model_comparison.json`:

| Chỉ Số Đánh Giá | Naive Median Baseline | Segment Median Baseline | ExtraTrees Champion | Ý Nghĩa Thực Tế |
| :--- | :---: | :---: | :---: | :--- |
| **MAE (Triệu VND)** | 6,255.1 | 5,963.1 | **4,621.7** | Champion giảm sai số **26.1%** so với Naive |
| **Median AE (Triệu VND)** | 3,325.0 | 2,975.0 | **2,157.4** | Sai số trung vị thực tế $\approx 2.16$ tỷ |
| **RMSE (Triệu VND)** | 10,033.9 | 9,846.0 | **7,874.0** | Độ lệch phương sai giảm rõ rệt |
| **$R^2$ Score** | -0.407 | -0.354 | **0.134** | Giải thích được 13.4% biến thiên giá rao |
| **MAPE (%)** | 54.03% | 49.49% | **35.96%** | Tỷ lệ sai số phần trăm trung bình |
| **WAPE (%)** | 58.08% | 55.37% | **42.92%** | Sai số phần trăm trọng số theo quy mô giá |
| **sMAPE (%)** | 60.33% | 55.97% | **40.54%** | Sai số phần trăm đối xứng |
| **Target Conformal Coverage** | N/A | N/A | **80.0%** | Mức độ bao phủ danh nghĩa kỳ vọng |
| **Actual Test Coverage** | N/A | N/A | **84.40%** | **Đạt và vượt kỳ vọng bao phủ** (92/109 mẫu) |
| **Mean Interval Width** | N/A | N/A | **12,621.6M** | Khoảng mở rộng trung bình 12.62 tỷ VND |
| **Relative Interval Width** | N/A | N/A | **1.172** | Bề rộng khoảng gấp 1.17 lần giá trị ước lượng |
| **Production Readiness** | — | — | **`research_only`** | Bắt buộc kèm rào chắn và tài sản so sánh |

---

## 8. Granular Slice & Uncertainty Analysis

Phân tích chi tiết tập Test theo các lát cắt từ `artifacts/error_analysis.json`:

### A. Theo Tầm Giá (Price Range Slices)

| Lát Cắt Giá | Số Mẫu | Mean MAE (Triệu VND) | Median MAE | WAPE (%) | Conformal Coverage (Target 80%) | Mean Interval Width | Nhận Định Vận Hành |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **< 5 tỷ VND** | 23 | **1,061.3** | **717.8** | **31.79%** | **86.96%** | 7,164.4 triệu | Rất ổn định, độ tin cậy cao |
| **5 – 10 tỷ VND** | 47 | **1,848.4** | **1,293.1** | **25.11%** | **100.0%** | 11,854.6 triệu | **Phân khúc tối ưu nhất (WAPE 25%)** |
| **10 – 15 tỷ VND** | 18 | 4,487.4 | 4,333.4 | 35.92% | **88.89%** | 15,387.3 triệu | Bao phủ tốt, phương sai bắt đầu tăng |
| **> 15 tỷ VND** | 21 | 14,843.5 | 13,112.8 | 59.24% | 42.86% | 17,944.5 triệu | Phương sai cực lớn $\rightarrow$ Kích hoạt rào chắn |

### B. Theo Độ Hoàn Thiện Dữ Liệu (`input_completeness_score`)

| Điểm Hoàn Thiện | Số Mẫu | Mean MAE (Triệu VND) | Median MAE | WAPE (%) | Coverage (Target 80%) | Ý Nghĩa Thực Nghiệm |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **>= 80% (Đầy đủ)** | 16 | **1,446.6** | **586.8** | **18.87%** | **100.0%** | **Đầy đủ dữ liệu cho sai số thấp nhất (WAPE < 19%)** |
| **60 – 80% (Khá)** | 55 | 4,687.2 | 3,268.7 | 41.84% | 87.27% | Sai số ở mức trung bình |
| **< 60% (Thiếu nhiều)** | 37 | 5,980.8 | 3,276.5 | 50.97% | 75.68% | Sai số tăng vọt, kích hoạt cảnh báo độ tin cậy thấp |

---

## 9. Multi-Dimensional Comparable Properties Engine

Hệ thống tách biệt động cơ tìm kiếm tài sản tương đồng thành module nghiệp vụ độc lập `src/comparables/`:

### A. Công Thức Khoảng Cách Tương Đồng Đa Chiều

$$d(x, c) = 0.30 \cdot d_{\text{area}} + 0.25 \cdot d_{\text{geo}} + 0.15 \cdot d_{\text{beds}} + 0.10 \cdot d_{\text{baths}} + 0.10 \cdot d_{\text{cbd}} + 0.10 \cdot d_{\text{recency}}$$

- **Bộ Lọc Ứng Viên (Candidate Filters)**: Cùng loại hình bất động sản, cùng quận hoặc quận lân cận, **loại trừ chính nhóm tài sản đang xét (`exclude same property_group_id`)**.
- **Chuẩn Hóa Thông Qua `ComparableContext`**: Lưu trữ các tham số tỷ lệ (median area, standard scale, CBD distance scale) được fit trên tập dữ liệu tham chiếu Train.
- **Không Trộn Tùy Tiện (Strict Decoupling)**: Không tự ý phối hợp tỷ lệ 70% model + 30% comparables khi chưa được benchmark. Hệ thống trả song song cả 2 thông tin để người dùng đối chiếu.

### B. Kiểm Chuẩn Độc Lập Offline (Validation Benchmark)

Để chứng minh Comparable Engine thực sự tạo bối cảnh thị trường có ý nghĩa chứ không chỉ mang tính trang trí giao diện, hệ thống thực hiện kiểm chuẩn truy xuất Top-K trên tập Validation (các căn tương đồng chỉ được lấy từ Train):

| Phương Pháp Định Vị | Val MAE (Triệu VND) | Val Median AE | Val WAPE (%) | Nhận Xét Kỹ Thuật |
| :--- | :---: | :---: | :---: | :--- |
| **Naive Global Median** | 3,801.1 | 2,200.0 | 49.69% | Baseline thô |
| **Comparable Median (Top-K Comps)** | **3,760.5** | **1,997.5** | **49.15%** | **Cạnh tranh vượt trội Naive Median** |
| **District x Type Segment Median** | 3,699.8 | 2,117.8 | 48.36% | Tham chiếu phân khúc khu vực |
| **ExtraTrees ML Champion** | **3,438.6** | **2,099.2** | **44.95%** | Tận dụng phi tuyến tính vượt trội |

---

## 10. Domain & Reliability Guardrails

Thay vì chỉ kiểm tra ngoại lai toán học đơn thuần, `src/reliability/guards.py` áp dụng hệ thống rào chắn thực địa:

1. **Phân Vị Robust P01 – P99**: Đánh giá các thuộc tính số so với phân vị 1% và 99% của tập Train.
2. **Ngưỡng Hạng Sang Dựa Trên Phân Vị Mục Tiêu (`target_p90`)**: Thay vì hard-code con số 15 tỷ, hệ thống so sánh với phân vị P90 của dữ liệu huấn luyện (**22.74 tỷ VND**).
3. **Phân Khúc Hỗ Trợ Kém (`LOW_SUPPORT_SEGMENT`)**: Tự động kích hoạt cảnh báo khi số lượng mẫu cùng loại hình và cùng quận trong tập huấn luyện nhỏ hơn 5 mẫu.
4. **Ngoại Suy Thời Gian (`TEMPORAL_EXTRAPOLATION`)**: Kích hoạt cảnh báo khi `as_of_date` lệch quá 180 ngày so với mốc tin đăng tham chiếu.
5. **Cấu Trúc Phân Rã Độ Tin Cậy (Decomposed Reliability)**:
   - `overall`: `high` | `medium` | `low`
   - `domain_support`: `in_domain` | `warning_ood`
   - `interval_risk`: `tight` | `moderate` | `wide_interval`
   - `input_completeness_score`: 0 – 100%

---

## 11. API Specification & Structured Intelligence Response

### A. Cấu Trúc 5 Trụ Cột (5-Pillar Response)

```json
{
  "valuation": {
    "point_estimate_million": 7850.0,
    "prediction_interval": {
      "lower_bound_million": 5420.0,
      "upper_bound_million": 11350.0,
      "target_coverage": 0.80
    }
  },
  "uncertainty": {
    "target_coverage": 0.80,
    "lower_bound_million": 5420.0,
    "upper_bound_million": 11350.0,
    "interval_width_million": 5930.0,
    "relative_interval_width": 0.755
  },
  "market_context": {
    "segment_median_unit_price_million_m2": 98.0,
    "comparable_median_price_million": 7600.0,
    "comparable_median_unit_price_million_m2": 95.0
  },
  "comparables": [
    {
      "property_type": "Nhà riêng",
      "location_area": "Quận 1",
      "area": 78.0,
      "price_million": 7500.0,
      "unit_price_million_m2": 96.15,
      "distance_km": 0.35,
      "similarity_score": 0.94
    }
  ],
  "reliability": {
    "overall": "medium",
    "reliability_level": "medium",
    "input_completeness_score": 85.0,
    "domain_support": "in_domain",
    "interval_risk": "moderate",
    "warnings": []
  },
  "model": {
    "version": "1.2.0",
    "model_type": "extra_trees",
    "target_formulation": "total_price",
    "production_readiness": "research_only"
  }
}
```

### B. API Endpoints

- `POST /predict`: Định giá bất động sản với hợp đồng 5 trụ cột đầy đủ (hỗ trợ trường `as_of_date`).
- `GET /market/districts`: Trích xuất danh sách các quận/huyện được hỗ trợ kèm số lượng mẫu tham chiếu.
- `GET /health`: Kiểm tra trạng thái sẵn sàng của dịch vụ và phiên bản mô hình nạp trong RAM.

---

## 12. Cấu Trúc Dự Án (Repository Structure)

```text
hcmc-real-estate-price-intelligence/
├── api/                        # FastAPI Serving Layer
│   ├── main.py                 # REST endpoints (/predict, /market/districts, /health)
│   └── schemas.py              # Pydantic Input/Output contracts
├── app/                        # Streamlit Interactive Web Application
│   └── streamlit_app.py
├── artifacts/                  # Versioned Run Artifacts
│   ├── data_card.json          # Data lineage, group audits & gate decisions
│   ├── metrics.json            # Final Locked Test evaluation metrics
│   ├── model_comparison.json   # Full validation candidate benchmarks
│   ├── error_analysis.json     # Granular slice reports
│   └── comparable_context.json # Fitted scaler and segment context
├── configs/                    # Declarative configuration specs
│   └── data_contract.yaml
├── data/                       # Data snapshots & schemas
│   └── sample/data_public_sample.csv
├── models/                     # Serialized Model Artifacts
│   └── price_model.joblib      # Production bundle (Model + Context + Conformal)
├── src/                        # Core Domain & Engineering Modules
│   ├── artifacts/              # Schema & persistence writers
│   ├── calibration/            # Split Conformal Prediction engine
│   ├── comparables/            # Dedicated Comparable Retrieval Engine
│   │   ├── context.py          # ComparableContext fit & serialization
│   │   ├── engine.py           # Multi-dimensional weighted similarity
│   │   └── evaluator.py        # Offline validation benchmark
│   ├── data/                   # Ingestion, cleaning, identity, split
│   │   ├── cleaning.py         # Target-aware market scope rules
│   │   ├── identity.py         # Multi-level Union-Find identity resolution
│   │   ├── split.py            # Group-isolated temporal ordering split
│   │   └── loader.py
│   ├── evaluation/             # Metrics & granular slice analysis
│   ├── features/               # Feature contracts, indicators & temporal
│   │   ├── builder.py          # Canonical FeatureBuilder with missingness
│   │   ├── context.py          # FeatureContext fitted scalers
│   │   └── temporal.py         # as_of_date & market_time_offset_days
│   ├── modeling/               # Training, candidate baselines & refit
│   ├── reliability/            # Domain & Reliability Guardrails
│   │   └── guards.py           # P01-P99, P90 luxury, temporal extrapolation
│   ├── serving/                # Real-time predictor & backward wrappers
│   └── pipeline.py             # End-to-end master pipeline CLI
└── tests/                      # Pytest Test Suites (Unit, Split, Lifecycle, API)
```

---

## 13. Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### 1. Cài đặt môi trường

```bash
git clone https://github.com/haminhthong/hcmc-real-estate-price-intelligence.git
cd hcmc-real-estate-price-intelligence

# Khởi tạo virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Trên Windows PowerShell
# source .venv/bin/activate     # Trên Linux/macOS

pip install -r requirements-dev.txt
```

### 2. Thực thi toàn bộ vòng đời ML Pipeline

```bash
# Chạy toàn bộ 11 bước offline pipeline và kết xuất artifacts chuẩn hóa
python -m src.pipeline train
```

### 3. Khởi chạy dịch vụ API & Web Dashboard

```bash
# FastAPI Serving
uvicorn api.main:app --reload --port 8000

# Streamlit UI
streamlit run app/streamlit_app.py
```

### 4. Chạy kiểm thử tự động (Test Suite)

```bash
python -m pytest -v
```

---

## 14. CV & GitHub Positioning

### GitHub Tagline
> **A leakage-aware residential listing-price intelligence system for Ho Chi Minh City combining property-identity resolution, grouped temporal validation, geospatial ML, conformal uncertainty, reliability guardrails, comparable-property retrieval and production-oriented serving.**

### CV Bullet Point
> **Built an HCMC residential listing-price intelligence platform with property-level leakage control, grouped temporal validation, geospatial/structural feature engineering, validation-only model selection, split-conformal intervals, reliability guardrails, comparable-property retrieval and FastAPI serving.**

### Pipeline Summary
> **HCMC Listings $\rightarrow$ Property Identity & Data QA $\rightarrow$ Grouped Temporal Split $\rightarrow$ Geospatial ML $\rightarrow$ Conformal Uncertainty $\rightarrow$ Reliability Guard $\rightarrow$ Comparable Properties $\rightarrow$ Price Intelligence API**

---

## 📝 License

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
