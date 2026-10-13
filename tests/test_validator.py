# -*- coding: utf-8 -*-

"""
alpha/validator.py 单元测试。

覆盖：
    - validate_expression 的语法错误（未闭合括号、空表达式、非法 token）
    - _check_operators 的非法运算符识别
    - _check_fields 的未知字段识别
    - infer_type 的类型推断
    - semantic_filter 的语义拒绝
    - structural_filter 的深度/节点数限制
    - validate_node 的完整链路
"""

from __future__ import annotations

import pytest

from alpha.ast import binary, constant, field, ts, unary
from alpha.fields import (
    FieldCatalog,
    FieldDefinition,
    default_catalog as default_field_catalog,
    set_default_catalog as set_field_catalog,
)
from alpha.operators import (
    OperatorCatalog,
    default_catalog as default_operator_catalog,
    set_default_catalog as set_operator_catalog,
)
from alpha.validator import (
    _check_fields,
    _check_operators,
    _is_pure_constant,
    infer_type,
    semantic_filter,
    structural_filter,
    validate_expression,
    validate_node,
)


# ============================================================
# Fixtures：使用受控的 catalog，避免依赖真实平台数据
# ============================================================

@pytest.fixture(autouse=True)
def _controlled_catalogs():
    """
    每个测试前设置一个最小可用的 field / operator catalog，
    测试后恢复默认。
    """
    saved_fields = default_field_catalog()
    saved_ops = default_operator_catalog()

    fields = FieldCatalog([
        FieldDefinition(name="close", field_type="price"),
        FieldDefinition(name="open", field_type="price"),
        FieldDefinition(name="volume", field_type="volume"),
        FieldDefinition(name="returns", field_type="returns"),
    ])
    ops = OperatorCatalog(
        unary=["rank", "abs", "log", "sign"],
        ts=["ts_mean", "ts_delta", "ts_rank"],
        binary=["+", "-", "*", "/"],
        windows=[5, 10, 20],
    )
    set_field_catalog(fields)
    set_operator_catalog(ops)

    yield

    set_field_catalog(saved_fields)
    set_operator_catalog(saved_ops)


# ============================================================
# validate_expression：语法错误
# ============================================================

def test_validate_expression_empty() -> None:
    ok, reason = validate_expression("")
    assert not ok
    assert "语法错误" in reason


def test_validate_expression_whitespace_only() -> None:
    ok, reason = validate_expression("   ")
    assert not ok
    assert "语法错误" in reason


def test_validate_expression_unclosed_paren() -> None:
    ok, reason = validate_expression("rank(close")
    assert not ok
    assert "语法错误" in reason


def test_validate_expression_extra_close_paren() -> None:
    ok, reason = validate_expression("rank(close))")
    assert not ok
    assert "语法错误" in reason


def test_validate_expression_nested_unclosed() -> None:
    ok, reason = validate_expression("rank(ts_mean(close, 5)")
    assert not ok
    assert "语法错误" in reason


def test_validate_expression_garbage() -> None:
    ok, reason = validate_expression("!!!@@@")
    assert not ok
    assert "语法错误" in reason


def test_validate_expression_valid() -> None:
    ok, reason = validate_expression("rank(close)")
    assert ok, reason


# ============================================================
# _check_operators：非法运算符
# ============================================================

def test_check_operators_valid_unary() -> None:
    node = unary("rank", field("close"))
    ok, reason = _check_operators(node)
    assert ok, reason


def test_check_operators_invalid_unary() -> None:
    node = unary("bogus_op", field("close"))
    ok, reason = _check_operators(node)
    assert not ok
    assert "非法运算符" in reason
    assert "bogus_op" in reason


def test_check_operators_invalid_ts() -> None:
    node = ts("ts_bogus", field("close"), 5)
    ok, reason = _check_operators(node)
    assert not ok
    assert "非法运算符" in reason


def test_check_operators_invalid_binary() -> None:
    node = binary("^", field("close"), field("open"))
    ok, reason = _check_operators(node)
    assert not ok
    assert "非法运算符" in reason


def test_check_operators_nested_invalid() -> None:
    inner = unary("bogus", field("close"))
    node = unary("rank", inner)
    ok, reason = _check_operators(node)
    assert not ok
    assert "bogus" in reason


def test_check_operators_none() -> None:
    ok, reason = _check_operators(None)
    assert ok


# ============================================================
# _check_fields：未知字段
# ============================================================

def test_check_fields_valid() -> None:
    node = field("close")
    ok, reason = _check_fields(node)
    assert ok, reason


def test_check_fields_unknown() -> None:
    node = field("nonexistent_field")
    ok, reason = _check_fields(node)
    assert not ok
    assert "未知字段" in reason
    assert "nonexistent_field" in reason


def test_check_fields_nested_unknown() -> None:
    inner = field("ghost")
    node = unary("rank", inner)
    ok, reason = _check_fields(node)
    assert not ok
    assert "ghost" in reason


def test_check_fields_binary_right_unknown() -> None:
    node = binary("+", field("close"), field("ghost"))
    ok, reason = _check_fields(node)
    assert not ok
    assert "ghost" in reason


def test_check_fields_none() -> None:
    ok, reason = _check_fields(None)
    assert ok


# ============================================================
# infer_type
# ============================================================

def test_infer_type_field() -> None:
    assert infer_type(field("close")) == "price"


def test_infer_type_constant() -> None:
    assert infer_type(constant("1")) == "rank"


def test_infer_type_rank_like() -> None:
    node = unary("rank", field("close"))
    assert infer_type(node) == "rank"


def test_infer_type_preserve() -> None:
    node = unary("abs", field("close"))
    assert infer_type(node) == "price"


def test_infer_type_binary_same_type() -> None:
    node = binary("+", field("close"), field("open"))
    assert infer_type(node) == "price"


def test_infer_type_binary_mismatch() -> None:
    node = binary("+", field("close"), field("volume"))
    assert infer_type(node) is None


def test_infer_type_none() -> None:
    assert infer_type(None) is None


# ============================================================
# semantic_filter
# ============================================================

def test_semantic_filter_single_field() -> None:
    ok, reason = semantic_filter(field("close"))
    assert not ok
    assert "单一字段" in reason


def test_semantic_filter_single_constant() -> None:
    ok, reason = semantic_filter(constant("1"))
    assert not ok
    assert "单一常量" in reason


def test_semantic_filter_pure_constant() -> None:
    node = binary("+", constant("1"), constant("2"))
    ok, reason = semantic_filter(node)
    assert not ok
    assert "纯常量" in reason


def test_semantic_filter_x_minus_x() -> None:
    node = binary("-", field("close"), field("close"))
    ok, reason = semantic_filter(node)
    assert not ok
    assert "恒等于0" in reason


def test_semantic_filter_x_div_x() -> None:
    node = binary("/", field("close"), field("close"))
    ok, reason = semantic_filter(node)
    assert not ok
    assert "恒等于1" in reason


def test_semantic_filter_rank_rank() -> None:
    inner = unary("rank", field("close"))
    node = unary("rank", inner)
    ok, reason = semantic_filter(node)
    assert not ok
    assert "冗余" in reason


def test_semantic_filter_valid() -> None:
    node = unary("rank", field("close"))
    ok, reason = semantic_filter(node)
    assert ok, reason


# ============================================================
# _is_pure_constant
# ============================================================

def test_is_pure_constant_single() -> None:
    assert _is_pure_constant(constant("1"))


def test_is_pure_constant_binary() -> None:
    node = binary("+", constant("1"), constant("2"))
    assert _is_pure_constant(node)


def test_is_pure_constant_with_field() -> None:
    node = binary("*", constant("-1"), field("close"))
    assert not _is_pure_constant(node)


# ============================================================
# structural_filter
# ============================================================

def test_structural_filter_ok() -> None:
    node = unary("rank", field("close"))
    ok, reason = structural_filter(node)
    assert ok, reason


def test_structural_filter_too_deep() -> None:
    node = field("close")
    for _ in range(10):
        node = unary("rank", node)
    ok, reason = structural_filter(node, max_depth=5)
    assert not ok
    assert "深度过大" in reason


def test_structural_filter_too_many_nodes() -> None:
    node = field("close")
    for _ in range(20):
        node = unary("rank", node)
    ok, reason = structural_filter(node, max_nodes=15)
    assert not ok
    assert "节点过多" in reason


# ============================================================
# validate_node：完整链路
# ============================================================

def test_validate_node_valid() -> None:
    node = unary("rank", field("close"))
    ok, reason = validate_node(node)
    assert ok, reason


def test_validate_node_unknown_field() -> None:
    node = unary("rank", field("ghost"))
    ok, reason = validate_node(node)
    assert not ok
    assert "未知字段" in reason


def test_validate_node_invalid_operator() -> None:
    node = unary("bogus", field("close"))
    ok, reason = validate_node(node)
    assert not ok
    assert "非法运算符" in reason


def test_validate_node_type_mismatch() -> None:
    node = binary("+", field("close"), field("volume"))
    ok, reason = validate_node(node)
    assert not ok
    assert "字段类型不兼容" in reason


def test_validate_node_semantic_reject() -> None:
    node = binary("-", field("close"), field("close"))
    ok, reason = validate_node(node)
    assert not ok
    assert "恒等于0" in reason


# ============================================================
# validate_expression：端到端
# ============================================================

def test_validate_expression_unknown_field() -> None:
    ok, reason = validate_expression("rank(ghost_field)")
    assert not ok
    assert "未知字段" in reason


def test_validate_expression_invalid_operator() -> None:
    ok, reason = validate_expression("bogus_op(close)")
    assert not ok
    assert "非法运算符" in reason


def test_validate_expression_valid_ts() -> None:
    ok, reason = validate_expression("ts_mean(close, 5)")
    assert ok, reason
