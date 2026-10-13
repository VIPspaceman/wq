# -*- coding: utf-8 -*-

"""
alpha/parser.py 单元测试。

覆盖：
    - 数字常量解析（含负数、浮点）
    - (-1*x) 形式解析
    - -1*close（无括号）解析
    - 嵌套括号
    - roundtrip 一致性
"""

from __future__ import annotations

import pytest

from alpha.ast import node_to_expr
from alpha.parser import parse_expression


# ============================================================
# 常量
# ============================================================

@pytest.mark.parametrize("expr", ["0", "1", "-1", "42", "0.5", "-0.5"])
def test_parse_constant(expr: str) -> None:
    node = parse_expression(expr)
    assert node is not None
    assert node.kind == "constant"
    assert node_to_expr(node) == expr


# ============================================================
# (-1*x)
# ============================================================

@pytest.mark.parametrize("expr", [
    "(-1*close)",
    "(-1*ts_delta(close,3))",
    "(-1*rank(ts_delta(close,10)))",
    "(-1*(rank(close)+abs(volume)))",
])
def test_parse_neg_one(expr: str) -> None:
    node = parse_expression(expr)
    assert node is not None
    assert node_to_expr(node) == expr


# ============================================================
# -1*close（无括号）
# ============================================================

def test_parse_neg_one_no_parens() -> None:
    """
    `-1*close` 应被解析为 binary("*", constant("-1"), field("close"))。
    """
    node = parse_expression("-1*close")
    assert node is not None
    assert node.kind == "binary"
    assert node.value == "*"
    assert node.left.kind == "constant"
    assert node.left.value == "-1"
    assert node.right.kind == "field"
    assert node.right.value == "close"


# ============================================================
# 老路径不受影响
# ============================================================

@pytest.mark.parametrize("expr", [
    "rank(ts_delta(close,3))",
    "abs(ts_std_dev(returns,10))",
    "(close/ts_mean(close,20))",
])
def test_backward_compat(expr: str) -> None:
    node = parse_expression(expr)
    assert node is not None
    assert node_to_expr(node) == expr


# ============================================================
# 非法输入
# ============================================================

@pytest.mark.parametrize("expr", ["", "   ", "unknown_field_xyz"])
def test_parse_invalid(expr: str) -> None:
    assert parse_expression(expr) is None
