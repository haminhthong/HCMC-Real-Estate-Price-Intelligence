"""Fixtures và cấu hình kiểm thử dùng chung cho toàn bộ dự án."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient cho integration test."""
    return TestClient(app)


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """DataFrame dữ liệu thô mẫu chứa đầy đủ các trường bắt buộc."""
    return pd.DataFrame(
        {
            "Price": [5000.0, 6000.0, 8500.0, 12000.0],
            "Area": [50.0, 60.0, 80.0, 120.0],
            "Property Type": [
                "Nhà riêng",
                "Nhà riêng",
                "Nhà mặt tiền",
                "Căn hộ chung cư",
            ],
            "Location": [
                "12 Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP.HCM",
                "45 Lê Lợi, Phường Bến Thành, Quận 1, TP.HCM",
                "78 Hai Bà Trưng, Phường Đa Kao, Quận 1, TP.HCM",
                "100 Nguyễn Thị Thập, Phường Tân Phú, Quận 7, TP.HCM",
            ],
            "Listing ID": ["id-1", "id-2", "id-3", "id-4"],
            "Bedrooms": [2, 3, 3, 4],
            "Bathrooms": [1, 2, 2, 3],
            "Floors": [1, 2, 3, 4],
            "Width": [4.0, 5.0, 4.5, 6.0],
            "Length": [12.0, 12.0, 18.0, 20.0],
            "Alley Width": [2.0, 3.0, 5.0, 6.0],
            "Direction": ["Đông", "Tây", "Nam", "Đông Nam"],
            "Position": ["Trong hẻm", "Đường chính", "Mặt tiền", "Trong hẻm"],
            "Latitude": [10.7769, 10.7770, 10.7780, 10.7300],
            "Longitude": [106.7009, 106.7010, 106.7020, 106.7200],
            "Last Updated Date": [
                "01/01/2025 10:00",
                "15/01/2025 11:00",
                "01/02/2025 09:00",
                "10/02/2025 14:00",
            ],
        }
    )
