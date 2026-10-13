# -*- coding: utf-8 -*-

"""
结构签名（唯一实现）。

职责边界
--------
只提供 structure_signature(node)。

与 alpha/ast.py 的关系
----------------------
本模块与 alpha/ast.py:structure_signature **行为完全一致**：

    field     → "F:<name>"
    constant  → "C:N"
    unary     → "U:<op>(<child>)"
    ts        → "T:<op>(<child>,N)"
    ts_binary → "TB:<op>(<left>,<right>,N)"
    binary    → "B:<op>(<left>,<right>)"

窗口数字统一替换为 "N"，因此同结构不同窗口视为同一签名。

alpha/ast.py 保留是为了向后兼容（现有模块仍在 import 它）。
新代码应使用 core.signature。

不含：
    - 表达式哈希（在 core/hashing.py）
    - AST 构造 / 序列化（在 alpha/ast.py）
"""

from __future__ import annotations

from typing import Any, Optional


def structure_signature(node: Optional[Any]) -> str:
    """
    生成 AST 结构签名。

    保留字段名、算子名、左右顺序；把所有窗口数字替换为 "N"。

    例：
        rank(ts_sum(debt,30)/assets)
            → "U:rank(B:/(T:ts_sum(F:debt,N),F:assets))"
        rank(ts_sum(debt,20)/assets)
            → 相同字符串 → 视为同结构

        rank(ts_mean(debt,30)/assets)
            → "U:rank(B:/(T:ts_mean(F:debt,N),F:assets))"  → 不同

    Parameters
    ----------
    node : Any | None
        duck typing：只要求有 .kind / .value / .child / .left /
        .right / .window 属性（即 alpha.ast.Node 的形状）。
        不 import alpha.ast，避免与 alpha/ast.py 循环 import。
        None 时返回空字符串。

    Returns
    -------
    str
        结构签名字符串；node 为 None 时返回 ""。
    """
    if node is None:
        return ""
    if node.kind == "field":
        return f"F:{node.value}"
    if node.kind == "constant":
        return "C:N"
    if node.kind == "unary":
        return f"U:{node.value}({structure_signature(node.child)})"
    if node.kind == "ts":
        return f"T:{node.value}({structure_signature(node.child)},N)"
    if node.kind == "ts_binary":
        left = structure_signature(node.left)
        right = structure_signature(node.right)
        return f"TB:{node.value}({left},{right},N)"
    if node.kind == "binary":
        left = structure_signature(node.left)
        right = structure_signature(node.right)
        return f"B:{node.value}({left},{right})"
    return ""
