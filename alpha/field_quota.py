# -*- coding: utf-8 -*-

"""
字段配额机制（FIELD-QUOTA）。

背景
----
exp_brain3_test_01（关闭饱和过滤跑 10×30）的结果：
    - 101 条评估中，debt 出现 81 次（80%）
    - Top10 全是 debt 单字段变体
    - CANDIDATE=0

根因：mutation/crossover 围绕 debt 反复进化，没有多样性约束。

职责边界
--------
只负责：
    - 从单个 AST 提取"主字段"（出现次数最多的 field）
    - 从一组表达式统计字段出现次数（分析用）

不含：
    - 配额判定本身（由 search/generation.py 承担）
    - 过滤动作（由 search/generation.py 承担）
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

from alpha.ast import Node, collect_nodes
from alpha.parser import parse_expression


def get_main_field(node: Optional[Node]) -> Optional[str]:
    """
    返回 AST 里出现次数最多的 field 节点的 value。

    平局时返回字母序最小的（保证确定性）。
    node 为 None 或没有 field 节点时返回 None。

    Parameters
    ----------
    node : Node | None

    Returns
    -------
    str | None
        主字段名；无 field 节点时返回 None。
    """
    if node is None:
        return None

    counts: Counter = Counter()
    for n in collect_nodes(node):
        if n.kind == "field" and n.value:
            counts[n.value] += 1

    if not counts:
        return None

    # 先按次数降序，再按字段名字母序升序 → 平局取字母序最小
    best = min(
        counts.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return best[0]


def compute_field_counts(expressions: List[str]) -> Counter:
    """
    输入一组表达式字符串，返回 {field_name: 出现次数}。

    用于分析（例如统计某一代里各字段的分布）。
    无法解析的表达式会被跳过（不抛异常）。

    Parameters
    ----------
    expressions : List[str]

    Returns
    -------
    Counter
        {field_name: 出现次数}
    """
    counts: Counter = Counter()
    for expr in expressions:
        if not expr:
            continue
        try:
            node = parse_expression(expr)
        except Exception:
            continue
        if node is None:
            continue
        for n in collect_nodes(node):
            if n.kind == "field" and n.value:
                counts[n.value] += 1
    return counts
