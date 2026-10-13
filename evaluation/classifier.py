# evaluation/classifier.py
# -*- coding: utf-8 -*-

"""
内部三层分级。

【重要声明】
------------------------------------------------------------
CANDIDATE / STRONG / FINAL 是 **本项目内部 heuristic**，
**不是** WorldQuant 官方 Alpha 提交标准。
------------------------------------------------------------

职责边界
--------
对齐 V3 classify_alpha()：
    - FINAL     : Sharpe >= 1.50, Fitness >= 1.20, Turnover <= 0.50
    - STRONG    : Sharpe >= 1.25, Fitness >= 1.00, Turnover <= 0.60
    - CANDIDATE : Sharpe >= 1.00, Fitness >= 0.50, Turnover <= 0.80
    - NEGATIVE  : Sharpe < 0 或 Fitness < 0
    - REJECT    : 其它

阈值从 config.thresholds 读取。
"""

from __future__ import annotations

from enum import Enum

from config.thresholds import CANDIDATE, FINAL, STRONG, AlphaThreshold


class Classification(str, Enum):
    FINAL = "FINAL"
    STRONG = "STRONG"
    CANDIDATE = "CANDIDATE"
    NEGATIVE = "NEGATIVE"
    REJECT = "REJECT"


def _meets(
    threshold: AlphaThreshold,
    sharpe: float,
    fitness: float,
    turnover: float,
) -> bool:
    return (
        sharpe >= threshold.sharpe
        and fitness >= threshold.fitness
        and turnover <= threshold.turnover_max
    )


def classify_alpha(
    sharpe: float,
    fitness: float,
    turnover: float,
) -> Classification:
    if _meets(FINAL, sharpe, fitness, turnover):
        return Classification.FINAL
    if _meets(STRONG, sharpe, fitness, turnover):
        return Classification.STRONG
    if _meets(CANDIDATE, sharpe, fitness, turnover):
        return Classification.CANDIDATE
    if sharpe < 0 or fitness < 0:
        return Classification.NEGATIVE
    return Classification.REJECT