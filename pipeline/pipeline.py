# -*- coding: utf-8 -*-

"""
Pipeline：按顺序执行 5 个 Filter。

职责边界
--------
    - 按固定顺序调用 Filter
    - 维护副作用（evaluated_hashes / structure_seen / field_counts）
    - 汇总 stats（漏斗日志）

副作用规则（与原 run_generation 完全一致）
------------------------------------------
    拒绝时：evaluated_hashes.add(h)
        - 例外：HashFilter 拒绝时不 add（本来就在集合里）
    通过时：evaluated_hashes.add(h)  ← 原逻辑在收集循环里 add
            structure_seen.add(sig)  ← 若有 sig
            field_counts[main_field] += 1  ← 若有 main_field

不含：
    - 表达式生成
    - simulation 提交
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from alpha.ast import Node, node_to_expr
from alpha.field_quota import get_main_field
from core.hashing import expression_hash
from core.signature import structure_signature
from pipeline.filters import (
    FieldQuotaFilter,
    Filter,
    FilterContext,
    HashFilter,
    SaturationFilter,
    StructureFilter,
    ValidateFilter,
)


logger = logging.getLogger(__name__)


# ============================================================
# Pipeline
# ============================================================

class Pipeline:
    """
    5 层过滤流水线。

    Parameters
    ----------
    filters : List[Filter]
        过滤器列表（顺序即执行顺序）。
        默认构造为 [Hash, Validate, Saturation, Structure, FieldQuota]。
    """

    def __init__(self, filters: Optional[List[Filter]] = None) -> None:
        if filters is None:
            filters = [
                HashFilter(),
                ValidateFilter(),
                SaturationFilter(),
                StructureFilter(),
                FieldQuotaFilter(),
            ]
        self._filters = list(filters)

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def run(
        self,
        nodes: List[Node],
        context: FilterContext,
    ) -> Tuple[List[Node], Dict[str, int]]:
        """
        对 nodes 依次应用所有 Filter。

        Parameters
        ----------
        nodes   : 待过滤的 AST 列表
        context : FilterContext（会被就地修改：evaluated_hashes /
                  structure_seen / field_counts）

        Returns
        -------
        (passed_nodes, stats)
            passed_nodes : 通过所有 Filter 的节点
            stats        : 漏斗统计字典
        """
        passed: List[Node] = []

        stats: Dict[str, int] = {
            "total_input": len(nodes),
            "hash_skipped": 0,
            "validate_skipped": 0,
            "structure_skipped": 0,
            "saturation_skipped": 0,
            "field_quota_skipped": 0,
            "passed": 0,
        }

        for node in nodes:
            expr = node_to_expr(node)
            h = expression_hash(expr)

            rejected = False
            for f in self._filters:
                reason = f.filter(node, context)
                if reason is None:
                    continue

                # ---- 拒绝 ----
                # HashFilter 拒绝时不 add（本来就在集合里）
                if not isinstance(f, HashFilter):
                    context.evaluated_hashes.add(h)

                key = f.stats_key
                if key in stats:
                    stats[key] += 1

                logger.debug(
                    "  [%s] %s | %s",
                    f.__class__.__name__, expr[:80], reason,
                )
                rejected = True
                break

            if rejected:
                continue

            # ---- 通过 ----
            # 原逻辑：通过后 add（避免重复评估）
            context.evaluated_hashes.add(h)

            sig = structure_signature(node)
            if sig:
                context.structure_seen.add(sig)

            main_field = get_main_field(node)
            if main_field:
                context.field_counts[main_field] += 1

            passed.append(node)
            stats["passed"] += 1

        return passed, stats
