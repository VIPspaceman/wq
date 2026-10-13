# -*- coding: utf-8 -*-

"""
单代流程。

职责边界
--------
    1. 从 population 生成 Task 列表（去重 + 校验）
    2. 调用 Executor 并发执行
    3. 从 Task 结果构造 AlphaRecord
    4. （可选）调用 agent 修复 candidates
    5. （可选）Sign-Flip 自动搜索
    6. （可选）Combine 自动搜索

说明
----
- search/refine_loop.py / search/sign_flip.py / search/combiner.py
  均用 try/except ImportError 兜底，缺失时优雅跳过。
- GenerationOptions 由上层（Coordinator）构造并传入。
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
)

from alpha.ast import Node, node_to_expr
from alpha.families import detect_family
from alpha.field_quota import get_main_field
from alpha.saturation import (
    get_saturation_reason,
    is_brain_saturated,
)
from alpha.validator import validate_node
from core.hashing import expression_hash
from core.signature import structure_signature
from diagnostic.diagnoser import diagnose
from evaluation.record import AlphaRecord, build_record
from execution.executor import Executor
from execution.task import Task, TaskState
from pipeline.filters import FilterContext
from pipeline.pipeline import Pipeline
from search.archive import AlphaArchive

if TYPE_CHECKING:
    from agent.refine_agent import LLMRefiningAgent

logger = logging.getLogger(__name__)


# ============================================================
# 结果
# ============================================================

@dataclass
class GenerationResult:
    generation: int
    submitted: List[Task] = field(default_factory=list)
    completed: List[AlphaRecord] = field(default_factory=list)
    failed: List[Task] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    refine_submitted: int = 0
    refine_succeeded: int = 0
    refine_failed: int = 0
    refine_exhausted: int = 0

    sign_flip_submitted: int = 0
    sign_flip_succeeded: int = 0
    combine_submitted: int = 0
    combine_succeeded: int = 0

    # BUG-1：真正提交 simulation 的次数（含 refine / sign_flip / combine）
    submitted_count: int = 0

    # FIELD-QUOTA：本代因字段配额被跳过的条数
    field_skipped: int = 0

    # 过滤漏斗统计（各环节拦截计数）
    stats: Dict[str, int] = field(default_factory=dict)


# ============================================================
# 配置（从 MiningConfig 传入）
# ============================================================

@dataclass
class GenerationOptions:
    """run_generation 的可选步骤配置。"""
    enable_sign_flip: bool = False
    enable_combine: bool = True
    max_sign_flips_per_gen: int = 5
    max_combines_per_gen: int = 3
    combine_size: int = 3
    combine_max_depth: int = 10
    combine_max_nodes: int = 30


# ============================================================
# 单代流程
# ============================================================

def run_generation(
    *,
    generation: int,
    population: List[Node],
    executor: Executor,
    evaluated_hashes: Set[str],
    max_depth: int = 5,
    max_nodes: int = 15,
    agent: Optional["LLMRefiningAgent"] = None,
    on_event: Optional[Any] = None,
    options: Optional[GenerationOptions] = None,
    on_record_complete: Optional[
        Callable[[AlphaRecord], None]
    ] = None,
    structure_seen: Optional[Set[str]] = None,
    archive: Optional[AlphaArchive] = None,
    enable_brain_saturation_filter: bool = True,
    max_single_field_ratio: float = 0.30,
) -> GenerationResult:
    if options is None:
        options = GenerationOptions()

    if structure_seen is None:
        structure_seen = set()

    expr_to_node: Dict[str, Node] = {}
    tasks: List[Task] = []
    skipped: List[str] = []
    # BUG-1：本代真正提交的 sim 数
    submitted_count: int = 0

    # FIELD-QUOTA：单字段配额（防止单一字段垄断种群）
    _FIELD_QUOTA_RATIO = max_single_field_ratio
    field_quota = max(1, int(len(population) * _FIELD_QUOTA_RATIO))

    # -------- 1. 准备 tasks（Step 4：统一走 Pipeline）--------
    context = FilterContext(
        evaluated_hashes=evaluated_hashes,
        structure_seen=structure_seen,
        field_counts=Counter(),
        field_quota=field_quota,
        max_depth=max_depth,
        max_nodes=max_nodes,
        enable_brain_saturation_filter=enable_brain_saturation_filter,
    )

    pipeline = Pipeline()
    passed_nodes, stats = pipeline.run(population, context)

    # 从 stats 还原日志所需的计数
    sat_skipped = stats["saturation_skipped"]
    field_skipped = stats["field_quota_skipped"]

    # 被跳过的表达式（用于 GenerationResult.skipped）
    passed_exprs = {node_to_expr(n) for n in passed_nodes}
    for node in population:
        expr = node_to_expr(node)
        if expr not in passed_exprs:
            skipped.append(expr)

    # 构造 tasks
    for node in passed_nodes:
        expr = node_to_expr(node)
        expr_to_node[expr] = node
        tasks.append(Task(expression=expr))

    logger.info(
        "Generation %d: submit=%d skipped=%d",
        generation, len(tasks), len(skipped),
    )
    if sat_skipped:
        logger.info(
            "本代因 BRAIN 饱和被跳过: %d 条", sat_skipped,
        )
    if field_skipped:
        logger.info(
            "本代因字段配额被跳过: %d 条", field_skipped,
        )

    # 过滤漏斗统计
    logger.info(
        "Generation %d 过滤漏斗: 输入=%d | hash=%d | "
        "validate=%d | structure=%d | saturation=%d | "
        "field_quota=%d | 通过=%d",
        generation,
        stats["total_input"], stats["hash_skipped"],
        stats["validate_skipped"], stats["structure_skipped"],
        stats["saturation_skipped"],
        stats["field_quota_skipped"], stats["passed"],
    )

    # -------- 2. 执行 --------
    if tasks:
        executor.run(tasks)
        # BUG-1：累加真正提交的 sim 数
        submitted_count += len(tasks)

    # -------- 3. 收集结果 --------
    completed: List[AlphaRecord] = []
    failed: List[Task] = []
    candidates_for_refine = []

    for task in tasks:
        evaluated_hashes.add(expression_hash(task.expression))

        if task.state != TaskState.COMPLETED or task.result is None:
            failed.append(task)
            continue

        node = expr_to_node.get(task.expression)
        if node is None:
            failed.append(task)
            continue

        if node.family is None:
            node.family = detect_family(task.expression)

        avg_corr = (
            archive.avg_corr_for(node)
            if archive is not None else 0.0
        )
        record = build_record(node, task.result, avg_corr=avg_corr)
        completed.append(record)
        _notify_record(on_record_complete, record)

        logger.info(
            "  → %-12s | sharpe=%+.4f fitness=%+.4f "
            "turnover=%.4f | %s",
            record.alpha_id or "N/A",
            record.sharpe, record.fitness, record.turnover,
            record.expression,
        )

        if agent is not None:
            report = diagnose(task.result, task.expression)
            if report.needs_refine:
                candidates_for_refine.append((task, report))

    result = GenerationResult(
        generation=generation,
        submitted=tasks,
        completed=completed,
        failed=failed,
        skipped=skipped,
        submitted_count=submitted_count,
        field_skipped=field_skipped,
        stats=stats,
    )

    # -------- 4. AI 修复 --------
    if agent is not None and candidates_for_refine:
        logger.info(
            "Generation %d: %d 条 CANDIDATE 进入 AI 修复",
            generation, len(candidates_for_refine),
        )
        try:
            from search.refine_loop import refine_candidates
        except ImportError:
            logger.warning(
                "search.refine_loop 不存在，跳过 AI 修复"
            )
        else:
            refine_summary = refine_candidates(
                generation=generation,
                candidates=candidates_for_refine,
                agent=agent,
                executor=executor,
                evaluated_hashes=evaluated_hashes,
                max_depth=max_depth,
                max_nodes=max_nodes,
                on_event=on_event,
                structure_seen=structure_seen,
            )

            for rec in refine_summary.records:
                result.completed.append(rec)
                _notify_record(on_record_complete, rec)
            result.refine_submitted = refine_summary.submitted
            result.refine_succeeded = refine_summary.succeeded
            result.refine_failed = (
                refine_summary.backtest_failed
                + refine_summary.llm_failed
            )
            result.refine_exhausted = refine_summary.exhausted
            # BUG-1：refine 提交的 sim 数
            submitted_count += refine_summary.submitted

    # -------- 5. Sign-Flip --------
    if options.enable_sign_flip and options.max_sign_flips_per_gen > 0:
        try:
            from search.sign_flip import build_sign_flip_tasks
        except ImportError:
            logger.warning(
                "search.sign_flip 不存在，跳过 Sign-Flip"
            )
        else:
            flip_tasks = build_sign_flip_tasks(
                result.completed,
                evaluated_hashes,
                max_flips=options.max_sign_flips_per_gen,
                max_depth=max_depth + 1,
                max_nodes=max_nodes + 2,
                structure_seen=structure_seen,
            )
            if flip_tasks:
                result.sign_flip_submitted = len(flip_tasks)
                executor.run(flip_tasks)
                # BUG-1：sign_flip 提交的 sim 数
                submitted_count += len(flip_tasks)

                for task in flip_tasks:
                    if (
                        task.state != TaskState.COMPLETED
                        or task.result is None
                    ):
                        continue
                    node = _parse_safe(task.expression)
                    if node is None:
                        continue
                    if node.family is None:
                        node.family = detect_family(task.expression)
                    avg_corr = (
                        archive.avg_corr_for(node)
                        if archive is not None else 0.0
                    )
                    record = build_record(
                        node, task.result, avg_corr=avg_corr,
                    )
                    record.add_tag("SIGN_FLIPPED")
                    result.completed.append(record)
                    _notify_record(on_record_complete, record)
                    result.sign_flip_succeeded += 1

    # -------- 6. Combine --------
    if options.enable_combine and options.max_combines_per_gen > 0:
        try:
            from search.combiner import build_combine_tasks
        except ImportError:
            logger.warning(
                "search.combiner 不存在，跳过 Combine"
            )
        else:
            combine_tasks = build_combine_tasks(
                result.completed,
                evaluated_hashes,
                max_combines=options.max_combines_per_gen,
                combine_size=options.combine_size,
                combine_max_depth=options.combine_max_depth,
                combine_max_nodes=options.combine_max_nodes,
                structure_seen=structure_seen,
            )
            if combine_tasks:
                result.combine_submitted = len(combine_tasks)
                executor.run(combine_tasks)
                # BUG-1：combine 提交的 sim 数
                submitted_count += len(combine_tasks)

                for task in combine_tasks:
                    if (
                        task.state != TaskState.COMPLETED
                        or task.result is None
                    ):
                        continue
                    node = _parse_safe(task.expression)
                    if node is None:
                        continue
                    if node.family is None:
                        node.family = detect_family(task.expression)
                    avg_corr = (
                        archive.avg_corr_for(node)
                        if archive is not None else 0.0
                    )
                    record = build_record(
                        node, task.result, avg_corr=avg_corr,
                    )
                    record.add_tag("COMBINED")
                    result.completed.append(record)
                    _notify_record(on_record_complete, record)
                    result.combine_succeeded += 1

    # BUG-1：同步最终 submitted_count（含 refine / sign_flip / combine）
    result.submitted_count = submitted_count

    return result


# ============================================================
# 内部辅助
# ============================================================

def _notify_record(
    callback: Optional[Callable[[AlphaRecord], None]],
    record: AlphaRecord,
) -> None:
    """安全触发 on_record_complete 回调；失败不阻塞主流程。"""
    if callback is None:
        return
    try:
        callback(record)
    except Exception:
        logger.exception("on_record_complete callback failed")


def _parse_safe(expr: str) -> Optional[Node]:
    """安全解析表达式；失败返回 None。"""
    try:
        from alpha.parser import parse_expression
        return parse_expression(expr)
    except Exception:
        logger.exception("解析表达式失败: %s", expr[:80])
        return None
