"""Kiểm thử đơn vị cho mô-đun Metrics."""

import numpy as np
import pytest
from src.evaluation.metrics import interval_metrics, regression_metrics


def test_regression_metrics_calculation():
    actual = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 310.0])
    m = regression_metrics(actual, pred)
    assert m["mae_million"] == 10.0
    assert m["median_ae_million"] == 10.0
    assert m["wape_percent"] == 5.0


def test_interval_metrics_calculation():
    actual = np.array([100.0, 200.0, 300.0])
    lower = np.array([90.0, 180.0, 290.0])
    upper = np.array([110.0, 220.0, 310.0])
    im = interval_metrics(actual, lower, upper, target_coverage=0.8)
    assert im["actual_coverage"] == 1.0
    assert im["target_coverage"] == 0.8
    assert im["coverage_gap"] == pytest.approx(0.2)


def test_development_and_release_gates():
    from src.artifacts.schema import evaluate_development_gate, evaluate_release_gate, evaluate_promotion

    # Development gate passes (improvement >= 10%, val_wape <= 35%)
    dev = evaluate_development_gate(champion_val_mae=3500.0, naive_val_mae=4200.0, val_wape=28.0)
    assert dev["champion_approved"] is True
    assert dev["beats_baseline"] is True

    # Release gate on test set: fails if test WAPE is high (> 30%)
    rel = evaluate_release_gate(test_wape=42.9, test_coverage=0.84)
    assert rel["production_ready"] is False
    assert rel["readiness_status"] == "research_only"

    # Hybrid promotion function reflects research_only
    promo = evaluate_promotion(
        champion_val_mae=3500.0,
        naive_val_mae=4200.0,
        val_wape=42.9,
        actual_coverage=0.84,
    )
    assert promo["production_readiness"] == "research_only"
    assert promo["deployment_approved"] is False
    assert "development_gate" in promo
    assert "release_gate" in promo

