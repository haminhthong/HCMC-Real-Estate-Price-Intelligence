"""Mô-đun xây dựng Pipeline tiền xử lý và mô hình hồi quy Scikit-learn."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FLAG_FEATURES,
    MISSING_INDICATOR_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_pipeline(model_name: str = "extra_trees") -> Pipeline:
    """Xây dựng pipeline xử lý đặc trưng và thuật toán dự báo theo cấu hình.

    Args:
        model_name: Tên thuật toán ('naive_median', 'ridge_linear', 'random_forest',
            'hist_gradient_boosting', 'extra_trees').

    Returns:
        Pipeline scikit-learn chưa được fit.
    """
    numeric_features = NUMERIC_FEATURES + FLAG_FEATURES + MISSING_INDICATOR_FEATURES
    cat_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    if model_name == "ridge_linear":
        num_pipeline = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        num_pipeline = SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    regressors = {
        "naive_median": DummyRegressor(strategy="median"),
        "ridge_linear": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.8,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }

    if model_name not in regressors:
        raise ValueError(
            f"Tên mô hình không hợp lệ: {model_name}. Chọn từ {list(regressors.keys())}"
        )

    return Pipeline([("preprocessor", preprocessor), ("model", regressors[model_name])])
