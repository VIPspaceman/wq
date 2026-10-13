# evolution/diversity.py
# -*- coding: utf-8 -*-

"""
家族多样性约束。

职责边界
--------
只提供"同一家族最多保留 N 个"的过滤函数，
供 selection.py 使用。

不含：
    - 排序（由调用方负责）
    - 精英数量截断
"""

from __future__ import annotations

from typing import Dict, List

from evaluation.record import AlphaRecord


def filter_by_family_diversity(
    records: List[AlphaRecord],
    max_per_family: int,
) -> List[AlphaRecord]:
    """
    保持输入顺序，过滤掉超出家族配额的元素。

    与 V3 select_elites 中"每个 family 最多保留 max_same_family_elite"
    的逻辑一致。

    注意：本函数 **不排序**。调用方应先按 Score 排序，再调用本函数。
    """
    if max_per_family <= 0:
        return list(records)

    selected: List[AlphaRecord] = []
    family_count: Dict[str, int] = {}

    for record in records:
        count = family_count.get(record.family, 0)
        if count >= max_per_family:
            continue
        selected.append(record)
        family_count[record.family] = count + 1

    return selected