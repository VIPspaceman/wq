# agent/cleaner.py
# -*- coding: utf-8 -*-

"""
LLM 输出清洗。

Step 10-Y 更新
--------------
新增 `_wrap_neg_one`：
    把 `-1 * x` 形式（无括号）转成 `(-1*x)`（带括号）。
    这样 parser 能正确识别常量与二元运算符。
"""

from __future__ import annotations

import re
from typing import Optional


# 预编译正则
_RE_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_\-]*")
_RE_PREFIX = re.compile(
    r"^(expression|output|result|fastexpr|answer)\s*[:：]\s*",
    re.IGNORECASE,
)
_RE_BACKTICK = re.compile(r"`+")
_RE_INLINE_COMMENT = re.compile(r"\s*#.*$")

# -1 * xxx 形式（无括号）
_RE_NEG_ONE = re.compile(r"^\s*-\s*1\s*\*\s*(.+?)\s*$")


# ============================================================
# 归一化：-1 * x → (-1*x)
# ============================================================

def _wrap_neg_one(expr: str) -> str:
    """
    `-1 * x`   → `(-1*x)`
    `-1*x`     → `(-1*x)`
    `-1 * (a + b)` → `(-1*(a + b))`
    """
    m = _RE_NEG_ONE.match(expr)
    if m:
        inner = m.group(1).strip()
        return f"(-1*{inner})"
    return expr


# ============================================================
# 主清洗函数
# ============================================================

def clean_llm_output(raw: Optional[str]) -> Optional[str]:
    """
    清洗 LLM 输出，返回单行 FASTEXPR；无法解析时返回 None。
    """
    if not raw or not isinstance(raw, str):
        return None

    text = raw

    # 1) 去掉 markdown 代码块标记
    text = _RE_CODE_FENCE.sub("", text)

    # 2) 逐行扫描，找第一个"看起来像表达式"的行
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        # 去掉常见前缀
        s = _RE_PREFIX.sub("", s)

        # 去掉反引号
        s = _RE_BACKTICK.sub("", s)

        # 去掉行内注释
        s = _RE_INLINE_COMMENT.sub("", s).strip()

        if not s:
            continue

        # 去掉行首行尾的引号
        s = s.strip("\"'").strip()

        if not s:
            continue

        # ★ Step 10-Y：-1 * x → (-1*x)
        s = _wrap_neg_one(s)

        return s

    return None


def is_probably_valid_expression(expr: str) -> bool:
    """
    轻量检查：只做"看上去像 FASTEXPR"的判断。
    """
    if not expr or not isinstance(expr, str):
        return False

    if "(" not in expr or ")" not in expr:
        return False

    for ch in expr:
        if "\u4e00" <= ch <= "\u9fff":
            return False

    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False

    return True