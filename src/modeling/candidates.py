"""Mô-đun định nghĩa danh sách các cấu hình và bài toán mục tiêu ứng viên."""

CANDIDATE_MODELS: list[str] = [
    "naive_median",
    "ridge_linear",
    "random_forest",
    "hist_gradient_boosting",
    "extra_trees",
]

TARGET_FORMULATIONS: list[str] = [
    "total_price",
    "price_per_m2",
]
