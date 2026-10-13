# -*- coding: utf-8 -*-

"""
agent/cleaner.py 单元测试。

覆盖：
    - _wrap_neg_one 各种输入
    - clean_llm_output 的 markdown / 前缀 / 注释清洗
    - is_probably_valid_expression
"""

from __future__ import annotations

import pytest

from agent.cleaner import (
    clean_llm_output,
    is_probably_valid_expression,
)


# ============================================================
# _wrap_neg_one（通过 clean_llm_output 间接测试）
# ============================================================

@pytest.mark.parametrize("raw,expected", [
    ("-1 * zscore(ts_delta(close, 3))",
     "(-1*zscore(ts_delta(close, 3)))"),
    ("-1*zscore(ts_delta(close, 3))",
     "(-1*zscore(ts_delta(close, 3)))"),
    ("-1 * rank(ts_delta(close, 10))",
     "(-1*rank(ts_delta(close, 10)))"),
    ("- 1 * rank(close)",
     "(-1*rank(close))"),
])
def test_wrap_neg_one(raw: str, expected: str) -> None:
    assert clean_llm_output(raw) == expected


def test_no_neg_one_unchanged() -> None:
    assert clean_llm_output("rank(close)") == "rank(close)"


# ============================================================
# markdown / 前缀 / 注释
# ============================================================

def test_strip_code_fence() -> None:
    raw = "```python\nrank(close)\n```"
    assert clean_llm_output(raw) == "rank(close)"


def test_strip_prefix() -> None:
    raw = "expression: rank(close)"
    assert clean_llm_output(raw) == "rank(close)"


def test_strip_inline_comment() -> None:
    raw = "rank(close)  # this is a comment"
    assert clean_llm_output(raw) == "rank(close)"


def test_strip_backticks() -> None:
    raw = "`rank(close)`"
    assert clean_llm_output(raw) == "rank(close)"


# ============================================================
# 空输入
# ============================================================

@pytest.mark.parametrize("raw", [None, "", "   ", "\n\n"])
def test_empty_input(raw) -> None:
    assert clean_llm_output(raw) is None


# ============================================================
# is_probably_valid_expression
# ============================================================

def test_valid_expression() -> None:
    assert is_probably_valid_expression("rank(close)")
    assert is_probably_valid_expression("(-1*rank(close))")


@pytest.mark.parametrize("expr", [
    "",
    "rank",              # 无括号
    "rank(close",        # 括号不匹配
    "rank)close(",       # 括号反向
    "rank(收盘价)",       # 含中文
])
def test_invalid_expression(expr: str) -> None:
    assert not is_probably_valid_expression(expr)
