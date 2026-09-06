"""Mô-đun tính toán các đặc trưng địa không gian (Geospatial Features)."""

import numpy as np
import pandas as pd


def calculate_distance_to_cbd(
    latitude: pd.Series,
    longitude: pd.Series,
    cbd_lat: float = 10.7769,
    cbd_lon: float = 106.7009,
) -> pd.Series:
    """Tính khoảng cách đường chim bay (km) tới CBD bằng công thức Haversine.

    Args:
        latitude: Chuỗi vĩ độ.
        longitude: Chuỗi kinh độ.
        cbd_lat: Vĩ độ CBD (mặc định Chợ Bến Thành: 10.7769).
        cbd_lon: Kinh độ CBD (mặc định Chợ Bến Thành: 106.7009).

    Returns:
        pd.Series khoảng cách tính theo km.
    """
    lat1 = np.radians(pd.to_numeric(latitude, errors="coerce").astype(float))
    lon1 = np.radians(pd.to_numeric(longitude, errors="coerce").astype(float))
    lat2 = np.radians(float(cbd_lat))
    lon2 = np.radians(float(cbd_lon))

    delta_lat = lat1 - lat2
    delta_lon = lon1 - lon2

    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(haversine))
