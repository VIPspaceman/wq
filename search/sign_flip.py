# search/sign_flip.py
# -*- coding: utf-8 -*-

"""
Sign-Flip 自动搜索。

职责边界
--------
    - 从 NEGATIVE 的 AlphaRecord 里筛选可翻转的
    - 生成 (-1*x) 变体
    - 去重

不含：
    - 提交（由 search/generation.py 通过 executor 做）
    - 诊断（diagnostic/ 已做）

设计
----
    筛选条件：sharpe <= -threshold
    排序：按 |sharpe| 降序
    数量：max_flips

与 AI 修复的区别
----------------
    - AI 修复：对 CANDIDATE（0.8 ≤ sharpe < 1.25 或 sharpe ≤ -0.5）做处理
    - Sign-Flip：对 **NEGATIVE**（sharpe < -0.5）**主动尝试**取反

    两者可能有重叠。refine_loop 和 sign_flip 都通过 evaluated_hashes
    去重，所以不会重复消耗 sim。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

from alpha.ast import Node, binary, constant
from alpha.parser import parse_expression
from alpha.validator import validate_node
from core.hashing import expression_hash
from core.signature import structure_signature
from evaluation.classifier import Classification
from evaluation.record import AlphaRecord
from execution.task import Task


logger = logging.getLogger(__name__)


# ============================================================
# 主函数
# ============================================================

def build_sign_flip_tasks(
    records: List[AlphaRecord],
    evaluated_hashes: Set[str],
    *,
    max_flips: int = 5,
    negative_sharpe_threshold: float = -0.5,
    max_depth: int = 7,
    max_nodes: int = 22,
    structure_seen: Optional[Set[str]] = None,
) -> List[Task]:
    """
    从 NEGATIVE 记录里生成 Sign-Flip 任务。

    Parameters
    ----------
    records                    : 当前 generation 的所有 record
    evaluated_hashes           : 全局 hash 集合（会就地更新）
    max_flips                  : 每代最多生成几个 flip
    negative_sharpe_threshold  : sharpe <= 该值才考虑翻转（默认 -0.5）
    max_depth / max_nodes      : 校验上限（比默认放宽 1）

    Returns
    -------
    List[Task]（source="SIGN_FLIPPED"）
    """
    if structure_seen is None:
        structure_seen = set()

    # -------- 1. 筛选候选 --------
    candidates = []
    for r in records:
        if r.classification != Classification.NEGATIVE:
            continue
        if r.sharpe > negative_sharpe_threshold:
            continue
        # 只对纯负 sharpe 做（fitness 可正可负，只要 sharpe 负）
        candidates.append(r)

    if not candidates:
        return []

    # -------- 2. 按 |sharpe| 降序 --------
    candidates.sort(key=lambda r: abs(r.sharpe), reverse=True)

    # -------- 3. 逐个生成 flip --------
    tasks: List[Task] = []
    for r in candidates:
        if len(tasks) >= max_flips:
            break

        # 若已取反，跳过（避免双取反）
        # 规范化后判断：去掉所有空格，检查是否以 (-1* 开头
        normalized = r.expression.replace(" ", "")
        if normalized.startswith("(-1*"):
            logger.debug(
                "Sign-Flip: 表达式已取反，跳过: %s",
                r.expression[:60],
            )
            continue

        flip_expr = f"(-1*{r.expression})"

        # 去重
        h = expression_hash(flip_expr)
        if h in evaluated_hashes:
            continue

        # parse + validate
        node = parse_expression(flip_expr)
        if node is None:
            logger.debug(
                "Sign-Flip: 无法解析: %s", flip_expr[:80]
            )
            continue

        ok, reason = validate_node(
            node, max_depth=max_depth, max_nodes=max_nodes
        )
        if not ok:
            logger.debug(
                "Sign-Flip: 未过校验（%s）: %s",
                reason, flip_expr[:80],
            )
            continue

        # -------- 结构级去重（FIX-1） --------
        sig = structure_signature(node)
        if sig and sig in structure_seen:
            logger.debug(
                "Sign-Flip: 结构重复，跳过: %s (sig=%s)",
                flip_expr[:80], sig,
            )
            continue
        if sig:
            structure_seen.add(sig)

        evaluated_hashes.add(h)
        tasks.append(Task(
            expression=flip_expr,
            source="SIGN_FLIPPED",
        ))

    if tasks:
        logger.info(
            "Sign-Flip: 生成 %d 个取反变体 "
            "(候选 %d)",
            len(tasks), len(candidates),
        )

    return tasks
