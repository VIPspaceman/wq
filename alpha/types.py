# alpha/types.py
# -*- coding: utf-8 -*-

"""
Alpha 表达式类型系统。

Step 10-Y 更新
--------------
新增 CONSTANT 节点类型：
    - 用于表达 `-1` / `0` / `1` 等数字常量
    - 由 parser 从 LLM 输出或已有表达式中识别
    - AST 序列化 / 深度 / 计数均已支持
"""

from __future__ import annotations

from enum import Enum


class NodeKind(str, Enum):
    """AST 节点种类（与 V3 一致 + Step 10-Y 新增 constant）."""
    FIELD = "field"
    UNARY = "unary"
    TS = "ts"
    BINARY = "binary"
    CONSTANT = "constant"     # ★ Step 10-Y 新增


class FieldType(str, Enum):
    PRICE = "price"
    VOLUME = "volume"
    RETURN = "return"
    RATIO = "ratio"
    RANK = "rank"
    MATRIX = "matrix"