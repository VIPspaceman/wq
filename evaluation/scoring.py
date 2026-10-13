# evaluation/scoring.py
# -*- coding: utf-8 -*-

"""
内部 Score 计算。

【重要声明】
------------------------------------------------------------
Score 是 **本项目内部** 的排序指标，与 WorldQuant 官方无关。
------------------------------------------------------------

职责边界
--------
对齐 V3 calculate_score()：
    Score = w_f * fitness
          + w_s * sharpe
          - w_t * turnover
          - complexity_penalty * max(0, complexity - floor)

权重从 config.thresholds.DEFAULT_SCORING 读取。
"""

from __future__ import annotations

import logging

from config.thresholds import DEFAULT_SCORING, ScoringWeights


logger = logging.getLogger(__name__)


def calculate_score(
    sharpe: float,
    fitness: float,
    turnover: float,
    complexity: int,
    weights: ScoringWeights = DEFAULT_SCORING,
    *,
    avg_corr: float = 0.0,
) -> float:
    """
    计算内部 Score。

    FIX-6a：新增相关性惩罚项。

        NewScore = OldScore - correlation_penalty * avg_corr

    Parameters
    ----------
    avg_corr : float
        与精英池的平均相关性（0.0 ~ 1.0）。
        默认 0.0（向后兼容，旧调用不受影响）。

    TODO: 未来调用方需传入 avg_corr。
          当前建议用"结构签名代理"：
              同 structure_signature → 1.0
              不同 structure_signature → 0.0
          TODO: 未来可改为从 raw_json 提取 PNL 收益序列计算真实相关性。
    """
    old_score = (
        weights.fitness_weight * fitness
        + weights.sharpe_weight * sharpe
        - weights.turnover_weight * turnover
    )
    penalty = weights.complexity_penalty * max(
        0, complexity - weights.complexity_floor
    )
    old_score = old_score - penalty

    corr_penalty = weights.correlation_penalty * avg_corr
    new_score = old_score - corr_penalty

    logger.debug(
        "calculate_score: old=%.4f new=%.4f avg_corr=%.4f "
        "(corr_penalty=%.4f)",
        old_score, new_score, avg_corr, corr_penalty,
    )

    return new_score
