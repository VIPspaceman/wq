# evaluation/ranking.py
# -*- coding: utf-8 -*-

"""
排序辅助。

职责边界
--------
只做排序 / 取 Top-N；不做评分 / 不做过滤。
"""

from __future__ import annotations

from typing import List, Optional

from evaluation.record import AlphaRecord


def sort_by_score(
    records: List[AlphaRecord],
    reverse: bool = True,
) -> List[AlphaRecord]:
    return sorted(
        records,
        key=lambda x: x.score,
        reverse=reverse,
    )


def top_n_by_score(
    records: List[AlphaRecord],
    n: int,
) -> List[AlphaRecord]:
    if n <= 0:
        return []
    return sort_by_score(records, reverse=True)[:n]