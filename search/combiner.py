# -*- coding: utf-8 -*-

"""
Combine 自动搜索。

职责边界
--------
    - 从当前 generation 的 AlphaRecord 里挑选可组合的
    - 生成 K 元组合表达式：rank((a1+a2+...+aK))
    - 去重 + 校验

不含：
    - 提交（由 search/generation.py 通过 executor 做）
    - 诊断（diagnostic/ 已做）

设计
----
    筛选条件：
        - sharpe >= min_sharpe
        - turnover <= max_turnover
        - 不含 "COMBINED" tag（避免组合的组合）
    排序：按 score 降序
    家族多样性：每 family 最多选 max_per_family 个
    组合大小：combine_size（默认 3，即从 top-N 里取 K 个）
    组合数：max_combines

与 Sign-Flip 的区别
-------------------
    - Sign-Flip：对 NEGATIVE 做取反
    - Combine  ：对 CANDIDATE+ 做 K 元组合，探索交互项

两者都通过 evaluated_hashes 去重，不会重复消耗 sim。
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Set

from alpha.parser import parse_expression
from alpha.validator import validate_node
from core.hashing import expression_hash
from core.signature import structure_signature
from evaluation.record import AlphaRecord
from execution.task import Task


logger = logging.getLogger(__name__)


# ============================================================
# 主函数
# ============================================================

def build_combine_tasks(
    records: List[AlphaRecord],
    evaluated_hashes: Set[str],
    *,
    max_combines: int = 8,
    combine_size: int = 3,
    min_sharpe: float = 0.5,
    max_turnover: float = 0.9,
    max_per_family: int = 1,
    combine_max_depth: int = 10,
    combine_max_nodes: int = 30,
    structure_seen: Optional[Set[str]] = None,
) -> List[Task]:
    """
    从满足条件的记录里生成 Combine 任务。

    Parameters
    ----------
    records            : 当前 generation 的所有 record
    evaluated_hashes   : 全局 hash 集合（会就地更新）
    max_combines       : 每代最多生成几个组合
    combine_size       : 参与组合的 top-N 数量
    min_sharpe         : sharpe 下限
    max_turnover       : turnover 上限
    max_per_family     : 每个 family 最多选几个
    combine_max_depth  : 校验深度上限（比默认放宽）
    combine_max_nodes  : 校验节点上限（比默认放宽）

    Returns
    -------
    List[Task]（source="COMBINED"）
    """
    if max_combines <= 0 or combine_size < 2:
        return []

    if structure_seen is None:
        structure_seen = set()

    # -------- 1. 筛选候选 --------
    candidates: List[AlphaRecord] = []
    for r in records:
        if r.has_tag("COMBINED"):
            continue
        if r.sharpe < min_sharpe:
            continue
        if r.turnover > max_turnover:
            continue
        candidates.append(r)

    if len(candidates) < 2:
        return []

    # -------- 2. 按 score 降序 --------
    candidates.sort(key=lambda r: r.score, reverse=True)

    # -------- 3. 家族多样性：每 family 最多 max_per_family 个 --------
    family_count: Dict[str, int] = {}
    diverse: List[AlphaRecord] = []
    for r in candidates:
        cnt = family_count.get(r.family, 0)
        if cnt >= max_per_family:
            continue
        diverse.append(r)
        family_count[r.family] = cnt + 1

    if len(diverse) < 2:
        return []

    # -------- 4. 取 top combine_size --------
    top = diverse[:combine_size]

    # -------- 5. 生成组合 --------
    tasks: List[Task] = []
    for combo in itertools.combinations(top, 2):
        if len(tasks) >= max_combines:
            break

        # 组合表达式：rank((a1+a2))
        inner = "+".join(r.expression for r in combo)
        combined_expr = f"rank(({inner}))"

        # 去重
        h = expression_hash(combined_expr)
        if h in evaluated_hashes:
            continue

        # parse + validate
        node = parse_expression(combined_expr)
        if node is None:
            logger.debug(
                "Combine: 无法解析: %s", combined_expr[:80]
            )
            continue

        ok, reason = validate_node(
            node,
            max_depth=combine_max_depth,
            max_nodes=combine_max_nodes,
        )
        if not ok:
            logger.debug(
                "Combine: 未过校验（%s）: %s",
                reason, combined_expr[:80],
            )
            continue

        # -------- 结构级去重（FIX-1） --------
        sig = structure_signature(node)
        if sig and sig in structure_seen:
            logger.debug(
                "Combine: 结构重复，跳过: %s (sig=%s)",
                combined_expr[:80], sig,
            )
            continue
        if sig:
            structure_seen.add(sig)

        evaluated_hashes.add(h)
        tasks.append(Task(
            expression=combined_expr,
            source="COMBINED",
        ))

    if tasks:
        logger.info(
            "Combine: 生成 %d 个组合变体 "
            "(候选 %d, diverse %d, top %d)",
            len(tasks), len(candidates), len(diverse), len(top),
        )

    return tasks
