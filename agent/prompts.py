# agent/prompts.py
# -*- coding: utf-8 -*-

"""
动态 Prompt 构建器。

职责边界
--------
只负责根据 DiagnosticReport 生成 prompt 文本。

Step 10-Y 修正 v2（本次）
-------------------------
build_system_prompt() 改为 **动态**：
    - 从 alpha.operators.default_catalog() 读算子
    - 从 alpha.fields.default_catalog() 读字段
    - catalog 为空时降级到硬编码白名单

动态的好处：
    - LLM 看到的算子/字段 = 实际 catalog
    - Capability Discovery 的价值传导到 LLM
    - 不再遗漏 ts_quantile / ts_scale 等远程算子

不含：
    - LLM 调用
    - 输出清洗
    - 熔断计数
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from diagnostic.report import DiagnosticReport
from diagnostic.types import DefectType


logger = logging.getLogger(__name__)


# ============================================================
# Fallback 白名单
# ============================================================
#
# 当 catalog 未加载 / 加载失败时用。
# 这些是 V3 已验证 + pv1 常见的算子/字段。
#

ALLOWED_OPERATORS: List[str] = [
    # 一元
    "rank", "zscore", "scale", "quantile",
    "normalize", "winsorize",
    "abs", "sign", "reverse",
    # 时间序列
    "ts_delay", "ts_delta", "ts_mean", "ts_std_dev", "ts_rank",
    "ts_zscore", "ts_sum", "ts_av_diff", "ts_decay_linear",
    "ts_arg_max", "ts_arg_min", "ts_backfill",
    # 二元
    "+", "-", "*", "/",
]

ALLOWED_FIELDS: List[str] = [
    "close", "open", "high", "low", "vwap",
    "volume", "returns", "cap",
]


# ============================================================
# System Prompt 模板
# ============================================================

_SYSTEM_TEMPLATE = """\
You are a WorldQuant BRAIN FASTEXPR expert.

Your ONLY job is to output ONE single-line FASTEXPR expression that fixes the given defects.

STRICT RULES:
1. Output MUST be exactly one line, no markdown, no code fence, no explanation.
2. You may ONLY use these operators: {operators}
3. You may ONLY use these fields: {fields}
4. The expression must be valid FASTEXPR for the BRAIN platform.
5. Do NOT output anything other than the expression itself.
6. Prefer minimal modifications to the original expression.

SYNTAX RULES (CRITICAL):
- Binary operations MUST be enclosed in parentheses: `(a + b)`, `(-1 * x)`, `(x - y)`
- DO NOT write `-1 * x` (missing outer parentheses).
- DO NOT write `-1*x` (missing outer parentheses).
- DO write `(-1 * x)`.
- Negative constants are allowed only inside parentheses.

Output format:
<single_line_fastexpr>
"""


# ============================================================
# 动态读取 catalog
# ============================================================

def _get_dynamic_operators() -> List[str]:
    """
    从当前 OperatorCatalog 读算子。

    - 顺序：unary → ts → binary（先常用后特殊）
    - 空 catalog → 返回 fallback
    """
    try:
        from alpha.operators import default_catalog
        cat = default_catalog()
        operators: List[str] = []
        if cat.unary():
            operators.extend(cat.unary())
        if cat.ts():
            operators.extend(cat.ts())
        if cat.binary():
            operators.extend(cat.binary())
        if operators:
            return operators
    except Exception:
        logger.exception("读取 OperatorCatalog 失败，使用 fallback")

    return list(ALLOWED_OPERATORS)


def _get_dynamic_fields() -> List[str]:
    """
    从当前 FieldCatalog 读字段。

    - 空 catalog → 返回 fallback
    """
    try:
        from alpha.fields import default_catalog
        cat = default_catalog()
        fields = list(cat.names())
        if fields:
            return fields
    except Exception:
        logger.exception("读取 FieldCatalog 失败，使用 fallback")

    return list(ALLOWED_FIELDS)


# ============================================================
# System Prompt 构建
# ============================================================

def build_system_prompt(
    operators: Optional[Iterable[str]] = None,
    fields: Optional[Iterable[str]] = None,
) -> str:
    """
    构建 system prompt。

    Parameters
    ----------
    operators : 显式指定算子列表（None = 动态从 catalog 读）
    fields    : 显式指定字段列表（None = 动态从 catalog 读）

    Returns
    -------
    str
    """
    ops = (
        list(operators) if operators is not None
        else _get_dynamic_operators()
    )
    flds = (
        list(fields) if fields is not None
        else _get_dynamic_fields()
    )
    return _SYSTEM_TEMPLATE.format(
        operators=", ".join(ops),
        fields=", ".join(flds),
    )


# ============================================================
# 按 DefectType 生成修复指引
# ============================================================

_DEFECT_HINTS = {
    DefectType.HIGH_TURNOVER: (
        "HIGH_TURNOVER: wrap the core signal with ts_decay_linear(x, N) "
        "or ts_mean(x, N), N in [5, 30]. Keep the inner structure intact."
    ),
    DefectType.LOW_FITNESS: (
        "LOW_FITNESS: try stabilizing the output distribution. "
        "Wrap the outer expression with zscore(...) or rank(...). "
        "Also consider normalizing by ts_std_dev(x, N) where sensible."
    ),
    DefectType.LOW_SHARPE: (
        "LOW_SHARPE: the signal may be inverted or too weak. "
        "Try inverting the sign by wrapping the WHOLE expression in "
        "`(-1 * (...))` — parentheses are mandatory. "
        "Or combine with volume / ts_std_dev(returns, N) filters."
    ),
    DefectType.UNITS_ERROR: (
        "UNITS_ERROR: avoid unit-sensitive operators (sqrt, log, inverse) "
        "on price/volume fields. Prefer ts_delta / ts_std_dev / ts_mean."
    ),
    DefectType.SYNTAX_ERROR: (
        "SYNTAX_ERROR: rebuild the expression using only allowed "
        "operators and fields."
    ),
}


def _format_defects(defects: Iterable[DefectType]) -> str:
    lines = []
    for d in defects:
        hint = _DEFECT_HINTS.get(d)
        if hint:
            lines.append(f"- {hint}")
        else:
            lines.append(f"- {d.value}")
    return "\n".join(lines) if lines else "(no specific defects)"


# ============================================================
# 负方向强化提示
# ============================================================

_NEGATIVE_DIRECTION_HINT = """
⚠️ CRITICAL DIRECTION HINT:
The current sharpe is NEGATIVE, which means the signal is likely INVERTED.
You MUST INVERT the signal direction.

ACCEPTABLE inversion patterns (choose ONE):
  1. `(-1 * <original_expression>)`       ← parentheses required
  2. `reverse(<original_expression>)`     ← if reverse is semantically valid
  3. Complement of ranking:
     `(1 - rank(<expression>))` — but ONLY if the inner signal is a rank

❌ DO NOT output `-1 * x` (no outer parentheses).
❌ DO NOT add filters or normalization — ONLY invert the direction.
✅ PRESERVE the inner structure of the original expression as much as possible.

Example:
  original: rank(ts_delta(close, 3))
  FIXED   : (-1*rank(ts_delta(close, 3)))
"""


# ============================================================
# User Prompt
# ============================================================

_USER_TEMPLATE = """\
Original expression:
{expression}

Backtest metrics:
  sharpe   = {sharpe:+.4f}
  fitness  = {fitness:+.4f}
  turnover = {turnover:.4f}

Detected defects:
{defects}

Error message (if any):
{error_message}
{negative_hint}
Task: Output ONE fixed FASTEXPR expression on a single line.
"""


def build_user_prompt(report: DiagnosticReport) -> str:
    negative_hint = ""
    if report.direction_hint == "negative":
        negative_hint = _NEGATIVE_DIRECTION_HINT

    return _USER_TEMPLATE.format(
        expression=report.expression,
        sharpe=report.sharpe,
        fitness=report.fitness,
        turnover=report.turnover,
        defects=_format_defects(report.defects),
        error_message=report.error_message or "(none)",
        negative_hint=negative_hint,
    )