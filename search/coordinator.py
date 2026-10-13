# search/coordinator.py
# -*- coding: utf-8 -*-

"""
多代协调。

AB 更新
-------
- MiningConfig 增加 Sign-Flip / Combine 相关字段
- run_generation 接收 GenerationOptions
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from alpha.ast import Node
from alpha.parser import parse_expression
from core.signature import structure_signature
from evaluation.classifier import Classification
from evaluation.record import AlphaRecord
from evolution.population import (
    create_initial_population,
    create_next_generation,
)
from evolution.selection import select_elites
from execution.executor import Executor
from search.archive import AlphaArchive
from search.events import EventType
from search.generation import GenerationOptions, run_generation
from utils.random import set_seed


logger = logging.getLogger(__name__)


EventHook = Callable[[EventType, Dict[str, Any]], None]
GenerationCallback = Callable[["MiningState"], None]


def _fmt_duration(sec: float) -> str:
    """把秒数格式化为人类可读的时长。"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}h{m}m"
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


@dataclass
class MiningConfig:
    population_size: int = 30
    generations: int = 10
    elite_size: int = 8
    mutation_rate: float = 0.75
    crossover_rate: float = 0.50
    reverse_probability: float = 0.15
    max_depth: int = 5
    max_nodes: int = 15
    max_same_family: int = 3
    random_seed: int = 20261006

    # AB：Sign-Flip
    enable_sign_flip: bool = True
    max_sign_flips_per_gen: int = 5

    # AB：Combine
    enable_combine: bool = True
    max_combines_per_gen: int = 3
    combine_size: int = 3
    combine_max_depth: int = 10
    combine_max_nodes: int = 30

    # FIELD-QUOTA：单字段在一代种群中最多占的比例
    max_single_field_ratio: float = 0.30

    # FAMILY-QUOTA：初始种群中单 family 最多占的比例
    max_same_family_initial: float = 0.30

    # BRAIN 饱和过滤：是否在 simulation 前拦截"纯基本面"结构
    enable_brain_saturation_filter: bool = True


@dataclass
class MiningState:
    generation: int = -1
    evaluated_hashes: Set[str] = field(default_factory=set)
    archive: AlphaArchive = field(default_factory=AlphaArchive)
    population: List[Node] = field(default_factory=list)
    elites: List[AlphaRecord] = field(default_factory=list)
    final_concurrency: Optional[int] = None
    # BUG-1：真正提交 simulation 的次数（不含被去重跳过的）
    submitted_count: int = 0
    # FIX-6a-2：上一代精英池的结构签名集合（跨代传递）
    elite_signatures: Set[str] = field(default_factory=set)


class Coordinator:
    def __init__(
        self,
        executor: Executor,
        config: MiningConfig,
        on_event: Optional[EventHook] = None,
        agent: Optional[Any] = None,
        on_record_complete: Optional[
            Callable[[AlphaRecord], None]
        ] = None,
    ) -> None:
        self._executor = executor
        self._config = config
        self._on_event = on_event
        self._agent = agent
        self._on_record_complete = on_record_complete

    def run(
        self,
        initial_state: Optional[MiningState] = None,
        on_generation: Optional[GenerationCallback] = None,
    ) -> MiningState:
        cfg = self._config
        set_seed(cfg.random_seed)

        # AB：构造 GenerationOptions
        gen_options = GenerationOptions(
            enable_sign_flip=cfg.enable_sign_flip,
            enable_combine=cfg.enable_combine,
            max_sign_flips_per_gen=cfg.max_sign_flips_per_gen,
            max_combines_per_gen=cfg.max_combines_per_gen,
            combine_size=cfg.combine_size,
            combine_max_depth=cfg.combine_max_depth,
            combine_max_nodes=cfg.combine_max_nodes,
        )

        state = initial_state or MiningState()
        is_resume = bool(state.population)

        if not is_resume:
            # 初始种群生成阶段查历史 hash：
            # 与 run_generation 的去重口径保持一致，避免
            # create 生成的表达式在 run_generation 里被 hash 挡掉
            # （历史 1906 条里价量结构命中率高）。
            # structure_seen 仍在 run_generation 里生效，
            # 结构级去重不受影响；evaluated_hashes 仍会累加。
            # 临时修复（2026-10-11）：不再传 evaluated_hashes。
            # 日志证据：1919 条历史 hash 把所有模板生成的表达式都挡住
            # （reasons={'dup_evaluated': 10000}），导致 0/20。
            # 代价：会重新评估部分历史表达式。
            # 真正的解决要靠生成器产生新结构（后续）。
            state.population = create_initial_population(
                cfg.population_size,
                evaluated_hashes=None,
                max_same_family_ratio=cfg.max_same_family_initial,
            )
            start_gen = 0
            logger.info(
                "Initial population: %d", len(state.population)
            )
        else:
            start_gen = state.generation + 1
            logger.info(
                "Resume from generation %d (population=%d, "
                "archive=%d)",
                start_gen,
                len(state.population), len(state.archive),
            )

        if self._agent is not None:
            logger.info("AI Refine Agent: ENABLED")
        else:
            logger.info("AI Refine Agent: disabled")

        self._emit(EventType.PROGRAM_STARTED, {
            "population_size": cfg.population_size,
            "generations": cfg.generations,
            "random_seed": cfg.random_seed,
            "is_resume": is_resume,
            "start_generation": start_gen,
            "agent_enabled": self._agent is not None,
            "sign_flip_enabled": cfg.enable_sign_flip,
            "combine_enabled": cfg.enable_combine,
        })

        try:
            if start_gen >= cfg.generations:
                logger.warning(
                    "start_gen %d >= generations %d, nothing to do",
                    start_gen, cfg.generations,
                )

            archive_count_at_start = len(state.archive)
            structure_seen = set()
            run_start_time = time.time()

            for gen in range(start_gen, cfg.generations):
                state.generation = gen
                gen_start_time = time.time()

                self._emit(EventType.GENERATION_STARTED, {
                    "generation": gen,
                    "population_size": len(state.population),
                })

                try:
                    result = run_generation(
                        generation=gen,
                        population=state.population,
                        executor=self._executor,
                        evaluated_hashes=state.evaluated_hashes,
                        max_depth=cfg.max_depth,
                        max_nodes=cfg.max_nodes,
                        agent=self._agent,
                        on_event=self._on_event,
                        options=gen_options,
                        on_record_complete=self._on_record_complete,
                        structure_seen=structure_seen,
                        archive=state.archive,
                        max_single_field_ratio=(
                            cfg.max_single_field_ratio
                        ),
                        enable_brain_saturation_filter=(
                            cfg.enable_brain_saturation_filter
                        ),
                    )
                except Exception as e:
                    logger.exception(
                        "Generation %d 异常，跳过该代", gen
                    )
                    self._emit(EventType.SIMULATION_ERROR, {
                        "generation": gen,
                        "error": repr(e),
                    })
                    # 保存当前状态（如果有 on_generation 回调）
                    self._maybe_persist(state, on_generation)
                    continue

                # -------- FIX-5：archive 阶段二次结构去重 --------
                # 只对 CANDIDATE / STRONG / FINAL 做检查；
                # NEGATIVE / REJECT 不检查，正常加入。
                _DEDUP_CLASSES = (
                    Classification.CANDIDATE,
                    Classification.STRONG,
                    Classification.FINAL,
                )
                dup_skipped: List[str] = []

                for record in result.completed:
                    if record.classification in _DEDUP_CLASSES:
                        node = parse_expression(record.expression)
                        sig = (
                            structure_signature(node)
                            if node is not None else ""
                        )
                        if sig and not state.archive.try_add_structure(sig):
                            dup_skipped.append(record.expression)
                            continue

                    state.archive.add(record)
                    self._emit_found(record)

                if dup_skipped:
                    logger.info(
                        "Generation %d: archive 结构去重跳过 %d 条",
                        gen, len(dup_skipped),
                    )
                    for expr in dup_skipped[:3]:
                        logger.info(
                            "  [DUPLICATE_STRUCTURE] %s", expr[:80],
                        )

                # BUG-1：累加真正提交的 sim 数
                state.submitted_count += result.submitted_count

                for task in result.failed:
                    self._emit(EventType.SIMULATION_ERROR, {
                        "generation": gen,
                        "task_id": task.task_id,
                        "expression": task.expression,
                        "error": task.error,
                    })

                self._emit(EventType.GENERATION_COMPLETED, {
                    "generation": gen,
                    "submitted": len(result.submitted),
                    "completed": len(result.completed),
                    "failed": len(result.failed),
                    "skipped": len(result.skipped),
                    "archive_size": len(state.archive),
                    "refine_submitted": result.refine_submitted,
                    "refine_succeeded": result.refine_succeeded,
                    "refine_failed": result.refine_failed,
                    "refine_exhausted": result.refine_exhausted,
                    "sign_flip_submitted": result.sign_flip_submitted,
                    "sign_flip_succeeded": result.sign_flip_succeeded,
                    "combine_submitted": result.combine_submitted,
                    "combine_succeeded": result.combine_succeeded,
                })

                # 时间估算：用实际每代平均 sim 数（动态适应跨实验去重）
                gen_elapsed = time.time() - gen_start_time
                total_elapsed = time.time() - run_start_time

                sims_done = len(state.archive) - archive_count_at_start
                gens_done = gen - start_gen + 1
                gens_remaining = cfg.generations - gen - 1

                if gens_remaining <= 0:
                    # 最后一代：不再算剩余
                    eta_str = "-"
                elif sims_done > 0 and gens_done > 0:
                    avg_sims_per_gen = sims_done / gens_done
                    est_remaining_sims = (
                        avg_sims_per_gen * gens_remaining
                    )
                    avg_per_sim = total_elapsed / sims_done
                    est_remaining = avg_per_sim * est_remaining_sims
                    eta_str = _fmt_duration(est_remaining)
                else:
                    eta_str = "-"

                logger.info(
                    "Generation %d/%d 完成 | 本代 %s | 累计 %s | "
                    "已跑 %d 条 sim | 预计剩余 %s",
                    gen + 1, cfg.generations,
                    _fmt_duration(gen_elapsed),
                    _fmt_duration(total_elapsed),
                    sims_done,
                    eta_str,
                )

                # 整代全部被跳过：推进 seed 并重新生成种群
                if (
                    len(result.submitted) == 0
                    and gen < cfg.generations - 1
                ):
                    logger.warning(
                        "Generation %d 全部被跳过，推进 seed 重新生成",
                        gen,
                    )
                    set_seed(cfg.random_seed + gen + 1)
                    # 临时修复（2026-10-11）：同初始种群，不传
                    # evaluated_hashes，避免历史 hash 挡住所有生成。
                    state.population = create_initial_population(
                        cfg.population_size,
                        evaluated_hashes=None,
                        max_same_family_ratio=(
                            cfg.max_same_family_initial
                        ),
                    )
                    continue  # 跳过 select_elites / create_next_generation

                # 每代都更新 elites（含最后一代，供 checkpoint 使用）
                # FIX-6a-2：传入上一代精英池签名集合作为参考
                state.elites = select_elites(
                    result.completed,
                    elite_size=cfg.elite_size,
                    max_same_family=cfg.max_same_family,
                    reference_signatures=(
                        state.elite_signatures
                        if state.elite_signatures else None
                    ),
                )
                logger.info(
                    "Generation %d elites: %d",
                    gen, len(state.elites),
                )

                # FIX-6a-2：更新精英池签名集合，供下一代使用
                state.elite_signatures = {
                    sig
                    for sig in (
                        structure_signature(
                            parse_expression(r.expression)
                        )
                        for r in state.elites
                    )
                    if sig
                }

                self._maybe_persist(state, on_generation)

                if gen >= cfg.generations - 1:
                    break

                state.population = create_next_generation(
                    state.elites,
                    size=cfg.population_size,
                    mutation_rate=cfg.mutation_rate,
                    crossover_rate=cfg.crossover_rate,
                    reverse_probability=cfg.reverse_probability,
                    evaluated_hashes=state.evaluated_hashes,
                )

            self._emit(EventType.MINING_COMPLETED, {
                "generations": state.generation + 1,
                "archive_size": len(state.archive),
                "counts": state.archive.counts(),
            })

        except Exception as e:
            logger.exception("Mining failed")
            self._emit(EventType.MINING_FAILED, {"error": repr(e)})
            raise

        return state

    def _maybe_persist(
        self,
        state: MiningState,
        on_generation: Optional[GenerationCallback],
    ) -> None:
        # 无条件更新 final_concurrency（供 checkpoint 使用）
        try:
            state.final_concurrency = (
                self._executor.concurrency_controller.current()
            )
        except Exception:
            logger.debug(
                "读取 concurrency_controller.current() 失败",
                exc_info=True,
            )

        if on_generation is None:
            return
        try:
            on_generation(state)
        except Exception:
            # 回调失败不应中断 mining 主流程
            logger.exception("on_generation 回调失败")

    def _emit_found(self, record: AlphaRecord) -> None:
        if record.classification == Classification.FINAL:
            self._emit(EventType.FINAL_ALPHA_FOUND, {"record": record})
        elif record.classification == Classification.STRONG:
            self._emit(EventType.STRONG_ALPHA_FOUND, {"record": record})
        elif record.classification == Classification.CANDIDATE:
            self._emit(EventType.CANDIDATE_FOUND, {"record": record})

    def _emit(
        self, event: EventType, payload: Dict[str, Any],
    ) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event, payload)
        except Exception:
            logger.exception("event hook failed")
