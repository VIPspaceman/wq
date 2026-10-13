# evolution/selection.py
# -*- coding: utf-8 -*-

"""
精英选择。

职责边界
--------
对齐 V3 select_elites()：
    1) 按 Score 降序排序
    2) 应用家族多样性约束
    3) 截断到 elite_size

参数显式传入（不再读取全局 CONFIG）。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

from alpha.parser import parse_expression
from config.thresholds import MAX_SAME_FAMILY_ELITE
from core.hashing import expression_hash
from core.signature import structure_signature
from evaluation.record import AlphaRecord
from evaluation.scoring import calculate_score
from evolution.diversity import filter_by_family_diversity


logger = logging.getLogger(__name__)


def _adjusted_score(
    record: AlphaRecord,
    reference_signatures: Set[str],
) -> tuple:
    """
    FIX-6a-2：计算带相关性惩罚的 adjusted_score。

    结构签名代理：
        avg_corr = 精英池中与候选同结构签名的比例（0.0 ~ 1.0）
        同签名 → 1.0，不同 → 0.0

    Returns
    -------
    (adjusted_score, avg_corr)
    """
    node = parse_expression(record.expression)
    sig = structure_signature(node) if node is not None else ""

    if sig and sig in reference_signatures:
        avg_corr = 1.0
    else:
        avg_corr = 0.0

    adjusted = calculate_score(
        record.sharpe,
        record.fitness,
        record.turnover,
        record.complexity,
        avg_corr=avg_corr,
    )
    return adjusted, avg_corr


def select_elites(
    generation_records: List[AlphaRecord],
    *,
    elite_size: int,
    max_same_family: int = MAX_SAME_FAMILY_ELITE,
    reference_signatures: Optional[Set[str]] = None,
) -> List[AlphaRecord]:
    """
    精英选择。

    FIX-6a-2：若提供 reference_signatures（上一代精英池的结构签名集合），
    则用"带相关性惩罚的 adjusted_score"排序，而非 record.score。
    不修改 record.score 字段本身。
    """
    if reference_signatures:
        scored = []
        for record in generation_records:
            adjusted, avg_corr = _adjusted_score(
                record, reference_signatures,
            )
            scored.append((adjusted, avg_corr, record))

        # 日志：候选数 > 20 时只打 top 20
        scored.sort(key=lambda x: x[0], reverse=True)
        log_items = scored[:20] if len(scored) > 20 else scored
        for adjusted, avg_corr, record in log_items:
            logger.debug(
                "  [select] %s score=%.4f adjusted=%.4f avg_corr=%.4f",
                expression_hash(record.expression)[:8],
                record.score, adjusted, avg_corr,
            )

        ranked = [r for _, _, r in scored]
    else:
        # 第一代：无参考签名，不惩罚，按原 score 排序
        ranked = sorted(
            generation_records,
            key=lambda r: r.score,
            reverse=True,
        )

    diverse = filter_by_family_diversity(ranked, max_same_family)
    return diverse[:elite_size]
