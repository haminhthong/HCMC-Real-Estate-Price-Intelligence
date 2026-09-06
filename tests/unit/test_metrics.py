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
