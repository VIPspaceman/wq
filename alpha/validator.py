# alpha/validator.py
# -*- coding: utf-8 -*-

"""
AST 校验：类型推断 + 语义过滤 + 结构过滤 + 语法/运算符/字段检查。

Step 10-Y 更新
--------------
infer_type 新增 constant 分支：
    - constant 返回 "rank"（可和任何类型做运算）
semantic_filter 新增 constant 拒绝：
    - 单一常量不是有效 alpha

Step 10-Z 更新
--------------
新增 validate_expression(expr)：
    - 先 parse_expression，解析失败 → "语法错误"
    - 再 validate_node
新增 _check_operators(node)：
    - 递归检查所有 unary / ts / binary 运算符是否在 OperatorCatalog 中
新增 _check_fields(node)：
    - 递归检查所有 field 是否在 FieldCatalog 中
validate_node 接入上述两项检查，错误信息更准确。
"""

from __future__ import annotations

from typing import Optional, Tuple

from alpha.ast import (
    Node,
    node_count,
    node_depth,
    node_to_expr,
)
from alpha.fields import default_catalog as _field_catalog
from alpha.operators import default_catalog as _operator_catalog
from alpha.parser import parse_expression


# ============================================================
# 语义分组
# ============================================================

_UNARY_RANK_LIKE = frozenset({
    "rank", "zscore", "scale", "quantile",
    "normalize", "winsorize", "densify",
})

_UNARY_PRESERVE = frozenset({
    "abs", "log", "sqrt", "inverse", "reverse", "signed_power",
})

_TS_RANK_LIKE = frozenset({
    "ts_rank", "ts_zscore", "ts_quantile",
})

_TS_COUNT_LIKE = frozenset({
    "ts_arg_max", "ts_arg_min", "ts_count_nans", "ts_step",
})

_TS_PRESERVE = frozenset({
    "ts_delay", "ts_delta", "ts_mean", "ts_std_dev",
    "ts_sum", "ts_av_diff", "ts_decay_linear",
    "ts_product", "ts_scale", "ts_backfill",
})


# ============================================================
# 类型推断
# ============================================================

def infer_type(node: Optional[Node]) -> Optional[str]:
    if node is None:
        return None

    # -------- Field --------
    if node.kind == "field":
        return node.field_type

    # -------- Constant（Step 10-Y） --------
    if node.kind == "constant":
        # 常量视为 "rank"，可和任何类型做 + - * /
        return "rank"

    # -------- Unary --------
    if node.kind == "unary":
        child_type = infer_type(node.child)
        if child_type is None:
            return None
        if node.value in _UNARY_RANK_LIKE:
            return "rank"
        if node.value in _UNARY_PRESERVE:
            return child_type
        return child_type

    # -------- Time Series --------
    if node.kind == "ts":
        child_type = infer_type(node.child)
        if child_type is None:
            return None
        if node.value in _TS_RANK_LIKE:
            return "rank"
        if node.value in _TS_COUNT_LIKE:
            return "rank"
        if node.value in _TS_PRESERVE:
            return child_type
        return child_type

    # -------- Time Series Binary (ts_corr / ts_cov) --------
    if node.kind == "ts_binary":
        left_type = infer_type(node.left)
        right_type = infer_type(node.right)
        if left_type is None or right_type is None:
            return None
        # ts_corr / ts_cov 输出数值（相关系数/协方差），
        # 归类为 "matrix"（与 fundamental 字段同类型），
        # 以便后续能参与 + - * / 运算
        return "matrix"

    # -------- Binary --------
    if node.kind == "binary":
        left_type = infer_type(node.left)
        right_type = infer_type(node.right)
        if left_type is None or right_type is None:
            return None

        op = node.value

        if op == "+":
            if left_type == right_type:
                return left_type
            return None

        if op == "-":
            if left_type == right_type:
                return left_type
            return None

        if op == "*":
            if left_type == "rank":
                return right_type
            if right_type == "rank":
                return left_type
            if left_type == right_type:
                return left_type
            return None

        if op == "/":
            if left_type == right_type:
                return "ratio"
            if left_type == "rank" or right_type == "rank":
                return "ratio"
            return None

    return None


# ============================================================
# 运算符 / 字段 合法性检查（Step 10-Z）
# ============================================================

def _check_operators(node: Optional[Node]) -> Tuple[bool, str]:
    """
    递归检查所有 unary / ts / binary 运算符是否在 OperatorCatalog 中注册。
    """
    if node is None:
        return True, ""

    catalog = _operator_catalog()

    if node.kind == "unary":
        if catalog.get(node.value) is None:
            return False, f"非法运算符: {node.value}"
        return _check_operators(node.child)

    if node.kind == "ts":
        if catalog.get(node.value) is None:
            return False, f"非法运算符: {node.value}"
        return _check_operators(node.child)

    if node.kind == "ts_binary":
        if catalog.get(node.value) is None:
            return False, f"非法运算符: {node.value}"
        ok, reason = _check_operators(node.left)
        if not ok:
            return False, reason
        return _check_operators(node.right)

    if node.kind == "binary":
        if catalog.get(node.value) is None:
            return False, f"非法运算符: {node.value}"
        ok, reason = _check_operators(node.left)
        if not ok:
            return False, reason
        return _check_operators(node.right)

    return True, ""


def _check_fields(node: Optional[Node]) -> Tuple[bool, str]:
    """
    递归检查所有 field 是否在 FieldCatalog 中。
    """
    if node is None:
        return True, ""

    catalog = _field_catalog()

    if node.kind == "field":
        if catalog.get(node.value) is None:
            return False, f"未知字段: {node.value}"
        return True, ""

    if node.kind in ("unary", "ts"):
        return _check_fields(node.child)

    if node.kind == "ts_binary":
        ok, reason = _check_fields(node.left)
        if not ok:
            return False, reason
        return _check_fields(node.right)

    if node.kind == "binary":
        ok, reason = _check_fields(node.left)
        if not ok:
            return False, reason
        return _check_fields(node.right)

    return True, ""


# ============================================================
# 语义过滤
# ============================================================

def semantic_filter(node: Node) -> Tuple[bool, str]:
    expr = node_to_expr(node)

    if expr == "sign(volume)":
        return False, "sign(volume) 信息量极低"

    # 运行时读取 price 字段（set_default_catalog 后立即生效）
    price_fields = _field_catalog().price_fields()
    if expr.startswith("sign(") and any(x in expr for x in price_fields):
        return False, "sign(price) 信息量极低"

    if "rank(rank(" in expr:
        return False, "rank(rank(x)) 冗余"

    if "abs(abs(" in expr:
        return False, "abs(abs(x)) 冗余"

    if "ts_rank(ts_rank(" in expr:
        return False, "ts_rank 嵌套冗余"

    if node.kind == "binary":
        if node.value == "-":
            left = node_to_expr(node.left)
            right = node_to_expr(node.right)
            if left == right:
                return False, "x-x 恒等于0"
        if node.value == "/":
            left = node_to_expr(node.left)
            right = node_to_expr(node.right)
            if left == right:
                return False, "x/x 恒等于1"

    if node.kind == "field":
        return False, "单一字段不是有效Alpha"

    # Step 10-Y：单一常量
    if node.kind == "constant":
        return False, "单一常量不是有效Alpha"

    # 纯常量运算（如 1+2、3*4）无信息量
    if _is_pure_constant(node):
        return False, "纯常量运算无信息量"

    return True, ""


def _is_pure_constant(node: Node) -> bool:
    """
    判断子树是否完全由常量构成（无 field 叶子）。

    例：
        constant("1")                    → True
        binary("+", constant("1"), constant("2")) → True
        binary("*", constant("-1"), field("close")) → False
    """
    if node.kind == "constant":
        return True
    if node.kind == "field":
        return False
    if node.kind in ("unary", "ts"):
        return node.child is not None and _is_pure_constant(node.child)
    if node.kind == "ts_binary":
        return (
            node.left is not None
            and node.right is not None
            and _is_pure_constant(node.left)
            and _is_pure_constant(node.right)
        )
    if node.kind == "binary":
        return (
            node.left is not None
            and node.right is not None
            and _is_pure_constant(node.left)
            and _is_pure_constant(node.right)
        )
    return False


# ============================================================
# 结构过滤
# ============================================================

def structural_filter(
    node: Node,
    *,
    max_depth: int = 5,
    max_nodes: int = 15,
) -> Tuple[bool, str]:
    depth = node_depth(node)
    count = node_count(node)

    if depth > max_depth:
        return False, f"深度过大: {depth}"
    if count > max_nodes:
        return False, f"节点过多: {count}"

    return True, ""


# ============================================================
# 统一校验入口
# ============================================================

def validate_node(
    node: Node,
    *,
    max_depth: int = 5,
    max_nodes: int = 15,
) -> Tuple[bool, str]:
    ok, reason = structural_filter(
        node, max_depth=max_depth, max_nodes=max_nodes
    )
    if not ok:
        return False, reason

    # Step 10-Z：运算符合法性
    ok, reason = _check_operators(node)
    if not ok:
        return False, reason

    # Step 10-Z：字段存在性
    ok, reason = _check_fields(node)
    if not ok:
        return False, reason

    if infer_type(node) is None:
        return False, "字段类型不兼容"

    ok, reason = semantic_filter(node)
    if not ok:
        return False, reason

    return True, ""


def validate_expression(
    expr: str,
    *,
    max_depth: int = 5,
    max_nodes: int = 15,
) -> Tuple[bool, str]:
    """
    面向 FASTEXPR 字符串的统一校验入口。

    流程：
        1. parse_expression(expr) —— 解析失败（含未闭合括号、非法 token）→ "语法错误"
        2. validate_node(node)    —— 结构 / 运算符 / 字段 / 类型 / 语义

    Returns
    -------
    (ok, reason)
    """
    if not isinstance(expr, str) or not expr.strip():
        return False, "语法错误: 空表达式"

    node = parse_expression(expr)
    if node is None:
        return False, "语法错误: 无法解析表达式"

    return validate_node(node, max_depth=max_depth, max_nodes=max_nodes)
