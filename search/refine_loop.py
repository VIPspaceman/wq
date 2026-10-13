# search/refine_loop.py
# -*- coding: utf-8 -*-

"""
规则 + LLM 混合修复编排。

Step A 更新
-----------
- 先调 try_rule_refine(report)；规则成功则用规则结果（0 LLM 调用）
- 规则失败/不适用再走 LLM
- 统计细分：rule_* vs llm_*
- Tag：AI_REFINED（总）+ RULE_REFINED / LLM_REFINED（细分）
        AI_FIXED   （总）+ RULE_FIXED   / LLM_FIXED  （细分）

职责边界
--------
    - 接收待修复的 (Task, DiagnosticReport) 列表
    - 先规则后 LLM
    - 二次校验 + 去重
    - 通过 executor 重新回测
    - 返回修复后的 AlphaRecord

不含：
    - 诊断（diagnostic/ 已做）
    - 规则实现（agent/refine_router.py）
    - LLM 调用细节（agent/refine_agent.py）
    - 精英选择（evolution/ 已做）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.failures import RefineOutcome
from agent.refine_agent import LLMRefiningAgent
from agent.refine_router import (
    REFINE_DEPTH_BONUS,
    REFINE_NODE_BONUS,
    try_rule_refine,
)
from alpha.parser import parse_expression
from alpha.validator import validate_node
from core.hashing import expression_hash
from core.signature import structure_signature
from diagnostic.diagnoser import diagnose
from diagnostic.report import DiagnosticReport
from evaluation.record import AlphaRecord, build_record
from execution.executor import Executor
from execution.task import Task, TaskState
from search.events import EventType


logger = logging.getLogger(__name__)


# ============================================================
# 结果
# ============================================================

@dataclass
class RefineOutcomeSummary:
    """一次批修复的统计。"""
    submitted: int = 0
    succeeded: int = 0
    backtest_failed: int = 0  # 回测失败（task 未 COMPLETED）
    exhausted: int = 0        # LLM 熔断
    llm_failed: int = 0       # LLM 调用失败 / 输出非法

    # Step A：细分统计
    rule_submitted: int = 0
    rule_succeeded: int = 0
    llm_submitted: int = 0
    llm_succeeded: int = 0

    records: List[AlphaRecord] = field(default_factory=list)
    ai_fixed_records: List[AlphaRecord] = field(default_factory=list)


# ============================================================
# 主入口
# ============================================================

def refine_candidates(
    *,
    generation: int,
    candidates: List[Tuple[Task, DiagnosticReport]],
    agent: LLMRefiningAgent,
    executor: Executor,
    evaluated_hashes: Set[str],
    max_depth: int = 6,
    max_nodes: int = 20,
    on_event: Optional[Any] = None,
    event_types: Optional[Dict[str, Any]] = None,
    structure_seen: Optional[Set[str]] = None,
) -> RefineOutcomeSummary:
    """
    对 CANDIDATE 列表批量修复 + 回测。

    Step A：先规则后 LLM。
    """
    summary = RefineOutcomeSummary()

    if not candidates:
        return summary

    if structure_seen is None:
        structure_seen = set()

    emit = _make_emitter(on_event)

    # 修复时允许深度/节点放宽（因为修复本身就是"加一层包装"）
    refine_max_depth = max_depth + REFINE_DEPTH_BONUS
    refine_max_nodes = max_nodes + REFINE_NODE_BONUS

    # -------- 1. 逐条修复 --------
    refine_tasks: List[Task] = []
    method_by_expr: Dict[str, str] = {}

    for src_task, report in candidates:
        emit(EventType.AI_REFINE_STARTED, {
            "generation": generation,
            "original_expression": report.expression,
            "defects": [d.value for d in report.defects],
        })

        # ★ Step A：先尝试规则
        rule_result = try_rule_refine(
            report,
            max_depth=refine_max_depth,
            max_nodes=refine_max_nodes,
        )

        method = ""
        refined_expr: Optional[str] = None

        if rule_result is not None and rule_result.success:
            # 规则成功
            refined_expr = rule_result.refined_expression
            method = rule_result.method
        else:
            # 规则不适用 / 失败 → 走 LLM
            if rule_result is not None:
                logger.info(
                    "  规则未通过（%s），回退 LLM",
                    rule_result.error,
                )

            llm_result = agent.refine(report)

            if not llm_result.success or not llm_result.refined_expression:
                if llm_result.outcome == RefineOutcome.EXHAUSTED:
                    summary.exhausted += 1
                    emit(EventType.AI_REFINE_EXHAUSTED, {
                        "generation": generation,
                        "original_expression": report.expression,
                    })
                else:
                    summary.llm_failed += 1
                    emit(EventType.AI_REFINE_FAILED, {
                        "generation": generation,
                        "original_expression": report.expression,
                        "outcome": llm_result.outcome.value,
                        "error": llm_result.error,
                    })
                continue

            refined_expr = llm_result.refined_expression
            method = "llm_refine"

        # -------- 2. 去重 + 校验 --------
        h = expression_hash(refined_expr)
        if h in evaluated_hashes:
            logger.info(
                "Refined expr 已评估过，跳过: %s", refined_expr,
            )
            continue

        node = parse_expression(refined_expr)
        if node is None:
            logger.warning(
                "Refined expr 无法解析: %s", refined_expr,
            )
            continue

        ok, reason = validate_node(
            node,
            max_depth=refine_max_depth,
            max_nodes=refine_max_nodes,
        )
        if not ok:
            logger.info(
                "Refined expr 未过校验（%s）: %s",
                reason, refined_expr,
            )
            continue

        # -------- 2b. 结构级去重（FIX-1） --------
        sig = structure_signature(node)
        if sig and sig in structure_seen:
            logger.debug(
                "Refine: 结构重复，跳过: %s (sig=%s)",
                refined_expr[:80], sig,
            )
            continue
        if sig:
            structure_seen.add(sig)

        # -------- 3. 建 task --------
        evaluated_hashes.add(h)
        refine_tasks.append(Task(
            expression=refined_expr,
            source="AI_REFINED",
        ))
        method_by_expr[refined_expr] = method

        # 真正提交后才计数
        if method.startswith("rule_"):
            summary.rule_submitted += 1
        elif method == "llm_refine":
            summary.llm_submitted += 1

    if not refine_tasks:
        return summary

    # -------- 4. 重新回测 --------
    summary.submitted = len(refine_tasks)
    logger.info(
        "Generation %d: 提交 %d 条修复后的 alpha "
        "(rule=%d, llm=%d)",
        generation,
        len(refine_tasks),
        summary.rule_submitted,
        summary.llm_submitted,
    )
    executor.run(refine_tasks)

    # -------- 5. 收集结果 --------
    for task in refine_tasks:
        if task.state != TaskState.COMPLETED or task.result is None:
            summary.backtest_failed += 1
            continue

        node = parse_expression(task.expression)
        if node is None:
            summary.backtest_failed += 1
            continue

        record = build_record(node, task.result)

        # 打 tag
        method = method_by_expr.get(task.expression, "")
        record.add_tag("AI_REFINED")   # 总 tag（保持兼容）
        if method.startswith("rule_"):
            record.add_tag("RULE_REFINED")
        elif method == "llm_refine":
            record.add_tag("LLM_REFINED")

        # 二次诊断
        post_report = diagnose(task.result, task.expression)
        if post_report.is_pass:
            record.add_tag("AI_FIXED")   # 总 tag
            if method.startswith("rule_"):
                record.add_tag("RULE_FIXED")
            elif method == "llm_refine":
                record.add_tag("LLM_FIXED")
            summary.ai_fixed_records.append(record)

        summary.records.append(record)
        summary.succeeded += 1

        if method.startswith("rule_"):
            summary.rule_succeeded += 1
        elif method == "llm_refine":
            summary.llm_succeeded += 1

        logger.info(
            "  → [%s] %-12s | sharpe=%+.4f fitness=%+.4f "
            "turnover=%.4f | %s",
            method or "?",
            record.alpha_id or "N/A",
            record.sharpe,
            record.fitness,
            record.turnover,
            record.expression,
        )

        emit(EventType.AI_REFINE_SUCCEEDED, {
            "generation": generation,
            "expression": task.expression,
            "alpha_id": record.alpha_id,
            "sharpe": record.sharpe,
            "fitness": record.fitness,
            "turnover": record.turnover,
            "is_pass": post_report.is_pass,
            "method": method,
        })

    return summary


# ============================================================
# 事件发射器
# ============================================================

def _make_emitter(
    on_event: Optional[Any],
):
    if on_event is None:
        return lambda ev, pl: None

    def _emit(event_value, payload: Dict[str, Any]) -> None:
        try:
            on_event(event_value, payload)
        except Exception:
            logger.exception("event hook failed")

    return _emit
