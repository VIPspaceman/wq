# -*- coding: utf-8 -*-

"""
子树挖掘：从历史表达式里提取高频子树（"基因片段"）。

⚠️ 暂未启用（Step 5b 已回退）
-----------------------------
本模块 **当前不被任何生产代码调用**。

背景：
    Step 5b 曾把 mine_subtrees 接入 alpha/generator.py 的
    generate_grammar_alpha（30% 概率走子树池）。但子树池加载
    失败时（DB 无历史 / 加载异常），该分支返回 None，导致
    生成成功率从 3% 掉到接近 0%。

    因此 Step 5b 已回退：
        - alpha/generator.py 恢复原实现（无子树池）
        - search/miner.py 删除 _load_subtree_pool 调用

    本文件保留作为参考实现，未来若重新启用需：
        1. 修复"子树池为空时不应返回 None"的逻辑
        2. 在 search/miner.py 重新接入加载 + 注入
        3. 加成功率监控，确认不劣化

当前状态：**dead code（保留参考）**。
"""

职责边界
--------
只负责：
    - 从一组表达式字符串里解析 AST
    - 统计所有子树的出现频次
    - 返回频次 >= min_freq 的子树（去重）

不含：
    - 表达式生成（alpha/generator.py）
    - DB 查询（search/miner.py）
    - 子树池的注入（alpha/generator.py:set_subtree_pool）

设计
----
子树用 structure_signature 作为去重 key（同结构不同窗口视为同一子树）。
返回的 Node 是"首次出现"的那个实例的克隆。

不含：
    - 网络请求
    - 持久化
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

from alpha.ast import (
    Node,
    clone_node,
    collect_nodes,
    node_count,
    node_to_expr,
    structure_signature,
)
from alpha.parser import parse_expression


logger = logging.getLogger(__name__)


# 子树大小上限（避免把整个表达式当成一个"子树"）
_MAX_SUBTREE_NODES = 8
# 子树大小下限（太小的子树无信息量）
_MIN_SUBTREE_NODES = 2


def mine_subtrees(
    expressions: List[str],
    min_freq: int = 3,
    *,
    max_subtrees: int = 200,
) -> List[Node]:
    """
    从一组表达式里挖掘高频子树。

    Parameters
    ----------
    expressions : List[str]
        历史表达式字符串列表。
    min_freq : int
        最低出现频次（默认 3）。
    max_subtrees : int
        最多返回多少个子树（按频次降序）。

    Returns
    -------
    List[Node]
        高频子树（已克隆，可安全用于生成）。
        无结果时返回空列表（不抛异常）。
    """
    if not expressions:
        return []

    # sig -> (count, first_node)
    sig_counts: Counter = Counter()
    sig_nodes: Dict[str, Node] = {}

    for expr in expressions:
        if not expr:
            continue
        try:
            root = parse_expression(expr)
        except Exception:
            continue
        if root is None:
            continue

        # 收集所有子树（含 root 本身）
        for sub in _iter_subtrees(root):
            n = node_count(sub)
            if n < _MIN_SUBTREE_NODES or n > _MAX_SUBTREE_NODES:
                continue
            sig = structure_signature(sub)
            if not sig:
                continue
            sig_counts[sig] += 1
            if sig not in sig_nodes:
                sig_nodes[sig] = clone_node(sub)

    # 过滤 + 排序
    candidates = [
        (sig, cnt)
        for sig, cnt in sig_counts.items()
        if cnt >= min_freq
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    result: List[Node] = []
    for sig, _cnt in candidates[:max_subtrees]:
        node = sig_nodes.get(sig)
        if node is not None:
            result.append(node)

    logger.info(
        "mine_subtrees: 输入 %d 条表达式，挖掘出 %d 个高频子树 "
        "(min_freq=%d)",
        len(expressions), len(result), min_freq,
    )
    return result


# ============================================================
# 内部：遍历所有子树
# ============================================================

def _iter_subtrees(root: Node):
    """
    深度优先遍历，产出 root 的所有子树（含 root 本身）。

    注意：collect_nodes 只返回节点列表，不返回"以该节点为根的子树"。
    这里直接递归遍历。
    """
    if root is None:
        return
    yield root
    if root.kind in ("unary", "ts"):
        yield from _iter_subtrees(root.child)
    elif root.kind in ("ts_binary", "binary"):
        yield from _iter_subtrees(root.left)
        yield from _iter_subtrees(root.right)
    # field / constant：无子节点
