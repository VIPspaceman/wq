# agent/refine_router.py
# -*- coding: utf-8 -*-

"""
规则路由器：优先用确定性规则修复简单缺陷，LLM 只处理复杂场景。

Step A / v3 修正
----------------
三条规则：
    1. 负 sharpe（无高 turnover）           → (-1*x)
    2. 正 sharpe + 高 turnover（无 fitness 问题） → ts_decay_linear(x, 10)
    3. 正 sharpe + 低 fitness（无 turnover 问题） → zscore(x)
    其它 → LLM

v3 修正原因（关键）
------------------
LOW_SHARPE 是 **所有 CANDIDATE 的基础缺陷**（candidate 判定就是 sharpe < 1.25），
不应该作为规则 2/3 的排除条件。
v2 用 "LOW_SHARPE not in defects" 导致规则 2/3 几乎永远不命中。

v3 改用：
    - 规则 2 排除条件：LOW_FITNESS 存在（复合缺陷，交给 LLM）
    - 规则 3 排除条件：HIGH_TURNOVER 存在（复合缺陷，交给 LLM）

职责边界
--------
只做"是否可以用规则"的判断 + 生成规则表达式。

不做：
    - 回测（由 search/refine_loop.py 做）
    - 熔断计数（由 agent/failures.py 做）
    - tag 管理（由上层做）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from alpha.parser import parse_expression
from alpha.validator import validate_node
from diagnostic.report import DiagnosticReport
from diagnostic.types import DefectType

logger = logging.getLogger(__name__)


# 修复时允许的深度/节点放宽（因为修复本身就是"加一层包装"）
REFINE_DEPTH_BONUS = 1
REFINE_NODE_BONUS = 2


# ============================================================
# 结果
# ============================================================

@dataclass
class RuleRefineResult:
    success: bool
    original_expression: str
    refined_expression: Optional[str] = None
    method: str = ""               # "rule_invert" / "rule_decay" / "rule_zscore"
    error: Optional[str] = None


# ============================================================
# 主入口
# ============================================================

def try_rule_refine(
    report: DiagnosticReport,
    *,
    max_depth: int = 6,
    max_nodes: int = 20,
) -> Optional[RuleRefineResult]:
    """
    尝试用确定性规则修复。

    Returns
    -------
    None                    : 规则不适用 → 应走 LLM
    RuleRefineResult(...)   : 规则已处理（成功或失败）→ 不走 LLM
    """
    defects = set(report.defects)
    direction = report.direction_hint

    has_low_fitness = DefectType.LOW_FITNESS in defects
    has_high_turnover = DefectType.HIGH_TURNOVER in defects

    # ---- 规则 1：负 sharpe → 取反 ----
    # 允许：负方向 + 无 HIGH_TURNOVER
    # 负 sharpe 时 fitness 往往也负（数学相关），不需要额外排除
    if direction == "negative" and not has_high_turnover:
        cand = f"(-1*{report.expression})"
        return _validate(report, cand, "rule_invert",
                         max_depth, max_nodes)

    # ---- 规则 2：正 sharpe + 高 turnover（无 fitness 问题） ----
    # 排除条件：同时有 LOW_FITNESS → 复合缺陷，交给 LLM
    if (direction == "positive"
            and has_high_turnover
            and not has_low_fitness):
        cand = f"ts_decay_linear({report.expression},10)"
        return _validate(report, cand, "rule_decay",
                         max_depth, max_nodes)

    # ---- 规则 3：正 sharpe + 低 fitness（无 turnover 问题） ----
    # 排除条件：同时有 HIGH_TURNOVER → 复合缺陷，交给 LLM
    if (direction == "positive"
            and has_low_fitness
            and not has_high_turnover):
        cand = f"zscore({report.expression})"
        return _validate(report, cand, "rule_zscore",
                         max_depth, max_nodes)

    return None  # 规则不适用 → 走 LLM


def _validate(
    report: DiagnosticReport,
    cand: str,
    method: str,
    max_depth: int,
    max_nodes: int,
) -> RuleRefineResult:
    """对规则输出的表达式做 parse + validate。"""
    node = parse_expression(cand)
    if node is None:
        logger.info(
            "规则生成表达式无法解析: %s", cand,
        )
        return RuleRefineResult(
            success=False,
            original_expression=report.expression,
            method=method,
            error="rule output failed to parse",
        )

    ok, reason = validate_node(
        node, max_depth=max_depth, max_nodes=max_nodes
    )
    if not ok:
        logger.info(
            "规则生成表达式未过校验（%s）: %s", reason, cand,
        )
        return RuleRefineResult(
            success=False,
            original_expression=report.expression,
            method=method,
            error=f"rule output failed validation: {reason}",
        )

    logger.info(
        "规则修复 OK [%s]: %s → %s",
        method, report.expression[:50], cand[:80],
    )
    return RuleRefineResult(
        success=True,
        original_expression=report.expression,
        refined_expression=cand,
        method=method,
    )