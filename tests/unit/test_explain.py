"""Kiểm tra giải thích không bị thay bằng độ quan trọng toàn cục."""

import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.serving.explain import explain_top_features


def test_missing_shap_returns_no_local_explanation(monkeypatch):
    original_import = builtins.__import__

    def import_without_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("SHAP không được cài đặt")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_shap)
    preprocessor = SimpleNamespace(
        transform=lambda frame: frame.to_numpy(),
        get_feature_names_out=lambda: np.array(["num__Area"]),
    )
    package = {
        "pipeline": SimpleNamespace(
            named_steps={
                "preprocessor": preprocessor,
                "model": SimpleNamespace(feature_importances_=np.array([1.0])),
            }
        )
    }
    assert explain_top_features(package, pd.DataFrame({"Area": [75.0]})) == []
