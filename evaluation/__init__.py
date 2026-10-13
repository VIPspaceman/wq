# evaluation/__init__.py
# -*- coding: utf-8 -*-

"""
evaluation：BRAIN Simulation 结果分析层。

子模块：
    metrics          : sharpe / fitness / turnover 提取
    official_checks  : BRAIN 官方 checks 提取
    scoring          : 内部 Score
    classifier       : 内部三层分级
    record           : AlphaRecord + build_record
    ranking          : 排序辅助

⚠️ 本层不依赖 brain/、不调用 BRAIN。
   本层不产生 AST；只处理 result dict 与 AlphaRecord。
"""

from evaluation.classifier import Classification, classify_alpha
from evaluation.metrics import Metrics, extract_metrics
from evaluation.official_checks import (
    OfficialCheck,
    OfficialChecks,
    extract_official_checks,
)
from evaluation.ranking import sort_by_score, top_n_by_score
from evaluation.record import AlphaRecord, build_record
from evaluation.scoring import calculate_score

__all__ = [
    # metrics
    "Metrics", "extract_metrics",
    # official checks
    "OfficialCheck", "OfficialChecks", "extract_official_checks",
    # scoring
    "calculate_score",
    # classification
    "Classification", "classify_alpha",
    # record
    "AlphaRecord", "build_record",
    # ranking
    "sort_by_score", "top_n_by_score",
]