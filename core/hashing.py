# -*- coding: utf-8 -*-

"""
表达式哈希（唯一实现）。

职责边界
--------
只提供 expression_hash / normalize_expression。

与 utils/hashing.py 的关系
--------------------------
本模块与 utils/hashing.py **行为完全一致**：

    normalize_expression(expr) = expr.replace(" ", "").lower()
    expression_hash(expr)      = md5(normalize_expression(expr))

utils/hashing.py 保留是为了向后兼容（现有模块仍在 import 它）。
新代码应使用 core.hashing。

不含：
    - 结构签名（在 core/signature.py）
    - 其它哈希（如 uuid）
"""

from __future__ import annotations

import hashlib


def normalize_expression(expr: str) -> str:
    """
    归一化表达式：去掉所有空格 + 转小写。

    与 utils/hashing.py:normalize_expression 完全一致。
    """
    return expr.replace(" ", "").lower()


def expression_hash(expr: str) -> str:
    """
    计算表达式的 MD5 哈希。

    与 utils/hashing.py:expression_hash 完全一致。

    Parameters
    ----------
    expr : str
        表达式字符串（可为空）。

    Returns
    -------
    str
        32 位十六进制 MD5。
    """
    normalized = normalize_expression(expr)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()
