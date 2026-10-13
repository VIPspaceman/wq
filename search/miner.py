# search/miner.py
# -*- coding: utf-8 -*-

"""
顶层入口。

AB 更新
-------
build_mining_config() 传递 Sign-Flip / Combine 相关配置。
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from config.settings import AppSettings
from execution.concurrency import (
    ConcurrencySettings,
    build_concurrency_controller,
)
from execution.executor import Executor
from execution.rate_limiter import RateLimiter
from search.coordinator import (
    EventHook,
    MiningConfig,
    MiningState,
)
from search.orchestrator import Orchestrator

if TYPE_CHECKING:
    from agent.refine_agent import LLMRefiningAgent
    from storage.store import MiningStore


logger = logging.getLogger(__name__)


def build_executor(
    settings: AppSettings,
    initial_concurrency: Optional[int] = None,
    on_task_complete=None,
) -> Executor:
    exec_cfg = settings.execution
    rate_limiter: Optional[RateLimiter] = None
    if exec_cfg.rate_limit_per_second is not None:
        rate_limiter = RateLimiter(
            rate_per_second=exec_cfg.rate_limit_per_second,
        )

    controller = build_concurrency_controller(
        ConcurrencySettings(
            max_workers=exec_cfg.max_workers,
            adaptive=exec_cfg.adaptive,
            min_workers=exec_cfg.min_workers,
            max_workers_cap=exec_cfg.max_workers_cap,
            adaptive_success_threshold=(
                exec_cfg.adaptive_success_threshold
            ),
        ),
        initial_current=initial_concurrency,
    )

    return Executor(
        credentials=settings.credentials,
        endpoints=settings.endpoints,
        polling=settings.polling,
        simulation_settings=settings.simulation,
        concurrency=controller,
        rate_limiter=rate_limiter,
        max_retries=exec_cfg.max_retries,
        retry_backoff=exec_cfg.retry_backoff,
        concurrency_backoff=exec_cfg.concurrency_backoff,
        on_task_complete=on_task_complete,
    )


def build_mining_config(settings: AppSettings) -> MiningConfig:
    evo = settings.evolution
    return MiningConfig(
        population_size=evo.population_size,
        generations=evo.generations,
        elite_size=evo.elite_size,
        mutation_rate=evo.mutation_rate,
        crossover_rate=evo.crossover_rate,
        reverse_probability=evo.reverse_probability,
        max_depth=evo.max_depth,
        max_nodes=evo.max_nodes,
        max_same_family=evo.max_same_family_elite,
        random_seed=evo.random_seed,
        # AB
        enable_sign_flip=evo.enable_sign_flip,
        max_sign_flips_per_gen=evo.max_sign_flips_per_gen,
        enable_combine=evo.enable_combine,
        max_combines_per_gen=evo.max_combines_per_gen,
        combine_size=evo.combine_size,
        combine_max_depth=evo.combine_max_depth,
        combine_max_nodes=evo.combine_max_nodes,
        # BRAIN 饱和过滤
        enable_brain_saturation_filter=(
            evo.enable_brain_saturation_filter
        ),
        # FIELD-QUOTA / FAMILY-QUOTA
        max_single_field_ratio=evo.max_single_field_ratio,
        max_same_family_initial=evo.max_same_family_initial,
    )


def build_agent(
    settings: AppSettings,
) -> Optional["LLMRefiningAgent"]:
    agent_cfg = settings.agent
    if not agent_cfg.enabled:
        logger.info("AgentSettings.enabled=False，跳过 AI 修复")
        return None

    # 优先使用 settings 里的 api_key；否则从环境变量读
    has_key = bool(
        agent_cfg.api_key
        or os.environ.get("DEEPSEEK_API_KEY_WQ")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY_WQ")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not has_key:
        logger.warning(
            "AgentSettings.enabled=True 但缺少 API Key，跳过 AI 修复"
        )
        return None

    try:
        from agent.refine_agent import LLMRefiningAgent
        agent = LLMRefiningAgent(
            api_key=agent_cfg.api_key,
            base_url=agent_cfg.base_url,
            model=agent_cfg.model,
            timeout=agent_cfg.timeout,
            max_retries=agent_cfg.max_retries,
            max_attempts_per_expression=(
                agent_cfg.max_attempts_per_expression
            ),
        )
        logger.info(
            "✅ AI Refining Agent 已启用: model=%s base_url=%s",
            agent_cfg.model, agent_cfg.base_url,
        )
        return agent
    except Exception:
        logger.exception("构造 LLMRefiningAgent 失败")
        return None


def mine(
    settings: AppSettings,
    on_event: Optional[EventHook] = None,
    store: Optional["MiningStore"] = None,
    initial_state: Optional[MiningState] = None,
    initial_concurrency: Optional[int] = None,
    agent: Optional["LLMRefiningAgent"] = None,
) -> MiningState:
    executor = build_executor(
        settings,
        initial_concurrency=initial_concurrency,
        on_task_complete=(
            store.on_task_complete if store is not None else None
        ),
    )
    config = build_mining_config(settings)

    on_record_complete = None
    if store is not None:
        def _append_record(rec):
            store.append_record(rec)

        on_record_complete = _append_record

    # 跨实验去重策略（2026-10-11 修改）
    # ----------------------------------------
    # 旧行为：从 DB 预加载历史 expr_hash 到 state.evaluated_hashes，
    #         导致 1919 条历史 hash 覆盖模板池能生成的所有精确表达式，
    #         生成器永远产出不了新东西（0/20）。
    #
    # 新行为：不再预加载历史 hash。
    #         - 跨实验去重交给 structure_seen（结构级，在
    #           run_generation 里已工作）
    #         - evaluated_hashes 只做 **本实验内** 去重
    #           （run_generation 里累加本实验跑过的 hash）
    #
    # 副作用：同结构的窗口变体（如 ts_delta(close,5) 与
    #         ts_delta(close,10)）会同时被生成，但 structure_seen
    #         会挡掉其中一个，实际提交数不会暴增。
    if initial_state is None:
        initial_state = MiningState()
    # 确保 evaluated_hashes 为空集（本实验内去重从零开始）
    if not initial_state.evaluated_hashes:
        initial_state.evaluated_hashes = set()
    logger.info(
        "跨实验去重交给 structure_seen，"
        "evaluated_hashes 只做本实验内去重"
    )

    # Step 6：改用 Orchestrator（薄委托包装）
    orchestrator = Orchestrator(
        executor=executor,
        config=config,
        on_event=on_event,
        agent=agent,
        on_record_complete=on_record_complete,
    )

    on_generation = None
    if store is not None:
        on_generation = store.persist

    state = orchestrator.run(
        initial_state=initial_state,
        on_generation=on_generation,
    )

    try:
        state.final_concurrency = (
            executor.concurrency_controller.current()
        )
    except Exception:
        logger.exception("读取 concurrency_controller.current() 失败")
        state.final_concurrency = None

    if store is not None:
        store.persist(state)

    return state
