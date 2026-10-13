# alpha/ast.py
# -*- coding: utf-8 -*-

"""
Alpha AST。

Step 10-Y 更新
--------------
新增 constant 节点支持：
    - node_to_expr : 直接返回 node.value
    - node_depth   : 1
    - node_count   : 1
    - collect_nodes: 加入结果（无子节点）
    - replace_subtree : 兼容（constant 无 child）
    - 新增 constant() builder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from alpha.fields import default_catalog


# ============================================================
# Node
# ============================================================

@dataclass
class Node:
    kind: str

    value: str = ""

    left: Optional["Node"] = None
    right: Optional["Node"] = None
    child: Optional["Node"] = None

    window: Optional[int] = None

    field_type: Optional[str] = None
    family: Optional[str] = None

    # 说明
    # ----
    # kind 取值：
    #   "field"     : 叶子，value=字段名
    #   "constant"  : 叶子，value=常量字面量
    #   "unary"     : child
    #   "ts"        : child + window（一元时序）
    #   "ts_binary" : left + right + window（二元时序，如 ts_corr/ts_cov）
    #   "binary"    : left + right


# ============================================================
# 克隆
# ============================================================

def clone_node(node: Optional[Node]) -> Optional[Node]:
    if node is None:
        return None
    return Node(
        kind=node.kind,
        value=node.value,
        left=clone_node(node.left),
        right=clone_node(node.right),
        child=clone_node(node.child),
        window=node.window,
        field_type=node.field_type,
        family=node.family,
    )


# ============================================================
# 结构统计
# ============================================================

def node_depth(node: Optional[Node]) -> int:
    if node is None:
        return 0
    if node.kind in ("field", "constant"):     # ★ constant
        return 1
    if node.kind in ("unary", "ts"):
        return 1 + node_depth(node.child)
    if node.kind == "ts_binary":
        return 1 + max(node_depth(node.left), node_depth(node.right))
    if node.kind == "binary":
        return 1 + max(node_depth(node.left), node_depth(node.right))
    return 0


def node_count(node: Optional[Node]) -> int:
    if node is None:
        return 0
    if node.kind in ("field", "constant"):     # ★ constant
        return 1
    if node.kind in ("unary", "ts"):
        return 1 + node_count(node.child)
    if node.kind == "ts_binary":
        return 1 + node_count(node.left) + node_count(node.right)
    if node.kind == "binary":
        return 1 + node_count(node.left) + node_count(node.right)
    return 0


def collect_nodes(node: Optional[Node]) -> List[Node]:
    if node is None:
        return []
    result = [node]
    if node.kind in ("unary", "ts"):
        result.extend(collect_nodes(node.child))
    elif node.kind == "ts_binary":
        result.extend(collect_nodes(node.left))
        result.extend(collect_nodes(node.right))
    elif node.kind == "binary":
        result.extend(collect_nodes(node.left))
        result.extend(collect_nodes(node.right))
    # field / constant：无子节点，跳过
    return result


# ============================================================
# 子树替换
# ============================================================

def replace_subtree(
    root: Node,
    target: Node,
    replacement: Node,
) -> Node:
    if root is target:
        return clone_node(replacement)

    new_root = clone_node(root)

    if new_root.kind in ("unary", "ts"):
        new_root.child = replace_subtree(
            new_root.child, target, replacement
        )
    elif new_root.kind == "ts_binary":
        new_root.left = replace_subtree(
            new_root.left, target, replacement
        )
        new_root.right = replace_subtree(
            new_root.right, target, replacement
        )
    elif new_root.kind == "binary":
        new_root.left = replace_subtree(
            new_root.left, target, replacement
        )
        new_root.right = replace_subtree(
            new_root.right, target, replacement
        )

    return new_root


# ============================================================
# AST -> FASTEXPR
# ============================================================

def node_to_expr(node: Node) -> str:
    if node.kind in ("field", "constant"):     # ★ constant
        return node.value
    if node.kind == "unary":
        return f"{node.value}({node_to_expr(node.child)})"
    if node.kind == "ts":
        return f"{node.value}({node_to_expr(node.child)},{node.window})"
    if node.kind == "ts_binary":
        return (
            f"{node.value}("
            f"{node_to_expr(node.left)},"
            f"{node_to_expr(node.right)},"
            f"{node.window})"
        )
    if node.kind == "binary":
        return f"({node_to_expr(node.left)}{node.value}{node_to_expr(node.right)})"
    raise ValueError(f"Unknown AST node: {node.kind}")


# ============================================================
# 构造 helper
# ============================================================

def field(name: str, catalog=None) -> Node:
    if catalog is None:
        catalog = default_catalog()
    f = catalog.get(name)
    if f is None:
        raise KeyError(f"Unknown field: {name!r}")
    return Node(
        kind="field",
        value=name,
        field_type=f.field_type,
    )


def safe_field(name: str, catalog=None) -> Node:
    """
    返回字段的"安全包装"版本：
        winsorize(ts_backfill(name, 120))

    作用：
        - ts_backfill(x, 120) 前向填充缺失值
        - winsorize(...) 截断 4 倍标准差外的极值

    注意：
        - 使用 nested unary + ts，而非 winsorize(x, std=4)
        - 因为 winsorize(x, std=4) 有 named parameter，
          parser 无法正确解析
        - 嵌套版本 AST 完美支持，round-trip 正确
    """
    return unary(
        "winsorize",
        ts("ts_backfill", field(name, catalog), 120),
    )


def constant(value) -> Node:
    """
    构造常量节点。

    Parameters
    ----------
    value : int / float / str
        常量值（如 -1、0、1、0.5）
    """
    return Node(kind="constant", value=str(value))


def unary(op: str, child: Node) -> Node:
    return Node(kind="unary", value=op, child=child)


def ts(op: str, child: Node, window: int) -> Node:
    return Node(kind="ts", value=op, child=child, window=window)


def ts_binary(op: str, left: Node, right: Node, window: int) -> Node:
    """
    二元时序算子（如 ts_corr / ts_cov）。

    结构：op(left, right, window)
    """
    return Node(
        kind="ts_binary",
        value=op,
        left=left,
        right=right,
        window=window,
    )


def binary(op: str, left: Node, right: Node) -> Node:
    return Node(kind="binary", value=op, left=left, right=right)


