"""Kiểm thử đơn vị cho mô-đun Conformal Prediction."""

import numpy as np
import pytest

from src.calibration.conformal import conformal_quantile


def test_conformal_quantile_values():
    residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    q = conformal_quantile(residuals, coverage=0.8)
    assert q == 0.9, "Phân vị bảo thủ tại coverage 80% với 10 mẫu phải là 0.9"


def test_conformal_quantile_exceptions():
    with pytest.raises(ValueError, match="không được rỗng"):
        conformal_quantile(np.array([]))

    with pytest.raises(ValueError, match="khoảng \\(0, 1\\)"):
        conformal_quantile(np.array([1.0]), coverage=1.2)


@pytest.mark.parametrize(
    "residuals", [[float("nan")], [float("inf")], [-1.0], [[1, 2]]]
)
def test_invalid_residuals_are_rejected(residuals):
    """Không tạo khoảng dự báo từ phần dư sai kiểu hoặc không hữu hạn."""
    with pytest.raises(ValueError, match="một chiều"):
        conformal_quantile(np.asarray(residuals), coverage=0.5)


def test_insufficient_calibration_size_is_rejected():
    """Không hạ thứ hạng quantile để giả vờ đạt mức bao phủ yêu cầu."""
    with pytest.raises(ValueError, match="Không đủ mẫu"):
        conformal_quantile(np.array([1.0]), coverage=0.8)
