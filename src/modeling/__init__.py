"""Gói huấn luyện và phát triển mô hình hồi quy (Modeling).

Bao gồm:
- pipelines: Xây dựng pipeline Scikit-learn
- baselines: Naive Median và Segment Median
- candidates: Danh sách cấu hình ứng viên
- selector: Chọn model trên Validation
- trainer: Refit champion trên Train + Validation
"""

from .baselines import NaiveMedianBaseline, SegmentMedianBaseline
from .candidates import CANDIDATE_MODELS, TARGET_FORMULATIONS
from .pipelines import build_pipeline
from .selector import select_champion_model
from .trainer import refit_champion_model

__all__ = [
    "CANDIDATE_MODELS",
    "TARGET_FORMULATIONS",
    "NaiveMedianBaseline",
    "SegmentMedianBaseline",
    "build_pipeline",
    "refit_champion_model",
    "select_champion_model",
]
