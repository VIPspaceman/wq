# search/archive.py
# -*- coding: utf-8 -*-

"""
全局 Alpha 内存存档。

职责边界
--------
    - 保存所有已评估的 AlphaRecord
    - 提供按分级 / 按 Score 的查询

不含：
    - 磁盘持久化（Step 7 storage/）
    - 通知（Step 7 notification/）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from alpha.ast import Node
from core.signature import structure_signature
from evaluation.classifier import Classification
from evaluation.record import AlphaRecord


class AlphaArchive:
    def __init__(self) -> None:
        self._records: List[AlphaRecord] = []
        # FIX-5：结构签名集合，用于 archive 阶段二次去重
        self._structure_sigs: Set[str] = set()

    # --------------------------------------------------------
    # 写入
    # --------------------------------------------------------

    def add(self, record: AlphaRecord) -> None:
        self._records.append(record)

    # --------------------------------------------------------
    # FIX-5：结构级去重
    # --------------------------------------------------------

    def try_add_structure(self, sig: str) -> bool:
        """
        尝试登记一个结构签名。

        Returns
        -------
        True  : 该签名此前未出现过，已登记
        False : 该签名已存在（结构重复）
        """
        if not sig:
            return True
        if sig in self._structure_sigs:
            return False
        self._structure_sigs.add(sig)
        return True

    def has_structure(self, sig: str) -> bool:
        return bool(sig) and sig in self._structure_sigs

    # --------------------------------------------------------
    # FIX-6a-2：相关性惩罚代理
    # --------------------------------------------------------

    def avg_corr_for(self, node: Optional[Node]) -> float:
        """
        计算候选节点与存档的平均相关性（结构签名代理）。

        规则（与 evaluation/scoring.py 的 TODO 对齐）：
            同 structure_signature → 1.0
            不同 structure_signature → 0.0

        返回存档中与候选同结构签名的比例（0.0 ~ 1.0）。
        存档为空时返回 0.0（无惩罚）。
        """
        if node is None or not self._structure_sigs:
            return 0.0
        sig = structure_signature(node)
        if not sig:
            return 0.0
        return 1.0 if sig in self._structure_sigs else 0.0

    # --------------------------------------------------------
    # 读取
    # --------------------------------------------------------

    def all(self) -> List[AlphaRecord]:
        return list(self._records)

    def by_classification(
        self, cls: Classification,
    ) -> List[AlphaRecord]:
        return [r for r in self._records if r.classification == cls]

    def best_by_score(self, n: int) -> List[AlphaRecord]:
        if n <= 0:
            return []
        return sorted(
            self._records,
            key=lambda x: x.score,
            reverse=True,
        )[:n]

    def counts(self) -> Dict[str, int]:
        counts = {c.value: 0 for c in Classification}
        for r in self._records:
            counts[r.classification.value] += 1
        return counts

    def __len__(self) -> int:
        return len(self._records)
