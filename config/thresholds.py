# config/thresholds.py
# -*- coding: utf-8 -*-

"""
Alpha 分层阈值与评分权重。

【重要声明】
------------------------------------------------------------
以下阈值是 **本项目的内部 heuristic**，用于实验筛选与存档，
**不是** WorldQuant 官方 Alpha 提交标准，也**不代表**
任何官方评价体系。

对外沟通、日志、报告、Telegram 通知中，必须显式保留此声明。
------------------------------------------------------------

职责边界
--------
本模块只定义阈值/权重常量，不承担：
    - 分类逻辑（应位于 evaluation/classifier.py）
    - 评分计算（应位于 evaluation/scoring.py）
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# 分层阈值（内部 heuristic）
# ============================================================

@dataclass(frozen=True)
class AlphaThreshold:
    """
    单层 Alpha 判定阈值。

    Attributes
    ----------
    sharpe       : 最低 Sharpe
    fitness      : 最低 Fitness
    turnover_max : 允许的最大 Turnover
    """
    sharpe: float
    fitness: float
    turnover_max: float


# V3 内部 heuristic（请勿对外称为“官方标准”）
CANDIDATE = AlphaThreshold(sharpe=1.00, fitness=0.50, turnover_max=0.80)
STRONG    = AlphaThreshold(sharpe=1.25, fitness=1.00, turnover_max=0.60)
FINAL     = AlphaThreshold(sharpe=1.50, fitness=1.20, turnover_max=0.50)


# ============================================================
# 评分权重
# ============================================================

@dataclass(frozen=True)
class ScoringWeights:
    """
    Score = w_f * fitness
          + w_s * sharpe
          - w_t * turnover
          - complexity_penalty * max(0, complexity - min_complexity)

    注意：
        该 Score 是 **本程序内部** 的排序指标，
        不是 WorldQuant 官方评分。
    """
    fitness_weight: float = 0.50
    sharpe_weight: float = 0.35
    turnover_weight: float = 0.15
    complexity_penalty: float = 0.01
    complexity_floor: int = 5
    # FIX-6a：相关性惩罚系数 λ
    # NewScore = OldScore - correlation_penalty * avg_corr
    correlation_penalty: float = 0.3


DEFAULT_SCORING = ScoringWeights()


# ============================================================
# 多样性约束
# ============================================================

# 同一 Family 在 Elite 中的最大占比（V3 内部约束）
MAX_SAME_FAMILY_ELITE: int = 3


# ============================================================
# 常量声明
# ============================================================

DISCLAIMER_ZH = (
    "以上阈值是本项目内部 heuristic，"
    "不是 WorldQuant 官方 Alpha 提交标准。"
)
