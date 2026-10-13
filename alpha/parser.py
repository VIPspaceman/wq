# alpha/parser.py
# -*- coding: utf-8 -*-

r"""
FASTEXPR -> AST 反解析。

Step 10-Y 更新
--------------
新增 **数字常量** 识别：
    当 token 匹配 -?\d+(\.\d+)? 时，返回 constant 节点。
    例：`(-1*x)` 里的 `-1` → constant("-1")

Step 10-Y 修正
--------------
1. split_top_level_binary 加 i > 0 判定
   - i=0 的 -/+ 是符号（unary），不是二元运算符
   - 否则 `-1*close` 会被切成 ("", "-", "1*close")
2. docstring 改用 raw string，避免 `\d` 转义警告
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from alpha.ast import (
    Node,
    binary,
    constant,
    field,
    ts,
    ts_binary,
    unary,
)
from alpha.fields import default_catalog as _field_catalog
from alpha.operators import default_catalog as _op_catalog


# 数字正则（整数或浮点，可带负号）
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _is_number(s: str) -> bool:
    return bool(_NUMBER_RE.match(s))


def _split_top_level_commas(text: str) -> list:
    """
    按顶层（不在括号内）逗号切分，返回各段（已 strip）。

    例：
        "close,volume,20"          → ["close", "volume", "20"]
        "ts_mean(close,20),volume,10"
            → ["ts_mean(close,20)", "volume", "10"]
    """
    parts: list = []
    depth = 0
    cur = ""
    for ch in text:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _starts_with_number_then_op(s: str) -> bool:
    """
    判断 s 是否形如 "数字 + 运算符 + ..."（如 "1*close"、"2+3"）。

    用于区分：
        "-1*close"  → 应走 binary 分支（constant("-1") * close）
        "-close"    → 应走 -x 分支（(-1*close)）
    """
    m = re.match(r"^-?\d+(\.\d+)?", s)
    if not m:
        return False
    rest = s[m.end():]
    return bool(rest) and rest[0] in "+-*/"


# ============================================================
# 顶层二元切分
# ============================================================

def split_top_level_binary(
    text: str,
) -> Optional[Tuple[str, str, str]]:
    """
    在顶层（不在括号内）找到第一个二元运算符。

    ⚠️ 关键：跳过 i=0 的 -/+（那是符号，不是二元运算符）
        否则 `-1*close` 会被切成 ("", "-", "1*close")

    例：
        "a+b"        → ("a", "+", "b")
        "a-b"        → ("a", "-", "b")
        "-1*close"   → ("-1", "*", "close")      ← i=0 的 - 是符号
        "-1"         → None                      ← 无二元运算符
        "close/ts_mean(close,20)" → ("close", "/", "ts_mean(close,20)")
    """
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and i > 0 and ch in "+-*/":
            # i > 0：跳过开头的符号
            # 跳过连续运算符（如 a--b 的第二个 -）
            if ch in "+-" and text[i - 1] in "+-*/":
                continue
            return text[:i], ch, text[i + 1:]
    return None


# ============================================================
# Parser
# ============================================================

def parse_expression(expr: str) -> Optional[Node]:
    # Bug 2 修复：LLM（DeepSeek）输出常带空格，
    # 如 "(-1 * (winsorize(...)))"。先去掉所有空格再解析。
    # 注意：此处已存在该归一化；若仍解析失败，
    # 原因多半不是空格，而是表达式本身语法问题（见下方 TODO）。
    # TODO: 若日志里出现"无法解析"但表达式肉眼合法，
    #       需打印完整表达式（当前日志可能被截断）再排查。
    expr = expr.replace(" ", "")

    if not expr:
        return None

    # ---- Field ----
    fcat = _field_catalog()
    if fcat.get(expr) is not None:
        return field(expr)

    # ---- 数字常量 ----
    if _is_number(expr):
        return constant(expr)

    # ---- 负号 + 字段/表达式：-x → (-1*x) ----
    # ⚠️ 仅当 `-` 后面不是"数字 + 运算符"时才走此分支。
    #    否则 `-1*close` 会被误判为 `-(1*close)`。
    if expr.startswith("-") and len(expr) > 1:
        inner = expr[1:]
        # 若 inner 是纯数字 → 已在上面处理
        # 若 inner 以数字开头且后面还有运算符 → 交给 binary 分支
        if not _is_number(inner) and not _starts_with_number_then_op(inner):
            child = parse_expression(inner)
            if child is not None:
                return binary("*", constant("-1"), child)

    # ---- 运行时读取 operators ----
    ocat = _op_catalog()
    unary_ops = ocat.unary()
    ts_ops = ocat.ts()
    ts_binary_ops = ocat.ts_binary()

    # ---- Unary ----
    for op in unary_ops:
        prefix = op + "("
        if expr.startswith(prefix) and expr.endswith(")"):
            inner = expr[len(prefix):-1]
            child = parse_expression(inner)
            if child is None:
                return None
            return unary(op, child)

    # ---- Binary Time Series（ts_corr / ts_cov）----
    # 结构：op(x, y, d) —— 顶层有 2 个逗号
    for op in ts_binary_ops:
        prefix = op + "("
        if expr.startswith(prefix) and expr.endswith(")"):
            inner = expr[len(prefix):-1]

            parts = _split_top_level_commas(inner)
            if len(parts) != 3:
                return None

            left_expr, right_expr, window_expr = parts
            try:
                window = int(window_expr)
            except ValueError:
                return None

            left = parse_expression(left_expr)
            right = parse_expression(right_expr)
            if left is None or right is None:
                return None
            return ts_binary(op, left, right, window)

    # ---- Time Series ----
    for op in ts_ops:
        prefix = op + "("
        if expr.startswith(prefix) and expr.endswith(")"):
            inner = expr[len(prefix):-1]

            depth = 0
            split_pos = None
            for i, ch in enumerate(inner):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif ch == "," and depth == 0:
                    split_pos = i

            if split_pos is None:
                return None

            child_expr = inner[:split_pos]
            window_expr = inner[split_pos + 1:]

            try:
                window = int(window_expr)
            except ValueError:
                return None

            child = parse_expression(child_expr)
            if child is None:
                return None
            return ts(op, child, window)

    # ---- Binary ----
    # 1) 带外层括号：(a+b)
    if expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1]
        result = split_top_level_binary(inner)
        if result is None:
            return None
        left_expr, op, right_expr = result

        left = parse_expression(left_expr)
        right = parse_expression(right_expr)
        if left is None or right is None:
            return None

        return binary(op, left, right)

    # 2) 无外层括号的顶层二元：-1*close、a+b 等
    result = split_top_level_binary(expr)
    if result is not None:
        left_expr, op, right_expr = result
        left = parse_expression(left_expr)
        right = parse_expression(right_expr)
        if left is None or right is None:
            return None
        return binary(op, left, right)

    return None
