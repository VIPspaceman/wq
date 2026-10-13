# evaluation/metrics.py
# -*- coding: utf-8 -*-

"""
从 BRAIN Simulation result 提取核心指标。

职责边界
--------
对齐 V3 extract_metrics(result)：
    - 从 result["is"] 中取 sharpe / fitness / turnover
    - 缺失或异常 → 0.0
    - 返回 dataclass，而非 dict

不含：
    - 任何网络调用
    - 官方 checks 的解析（在 official_checks.py）
    - Score 计算（在 scoring.py）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Metrics:
    """核心指标。"""
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def extract_metrics(result: Dict[str, Any]) -> Metrics:
    """
    从 result dict 提取 Metrics。

    与 V3 行为一致：
        - 若 result["is"] 不是 dict，视为空
        - 每个字段独立容错，异常 → 0.0
    """
    is_data = result.get("is", {})
    if not isinstance(is_data, dict):
        is_data = {}

    return Metrics(
        sharpe=_as_float(is_data.get("sharpe", 0)),
        fitness=_as_float(is_data.get("fitness", 0)),
        turnover=_as_float(is_data.get("turnover", 0)),
    )