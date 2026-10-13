# -*- coding: utf-8 -*-

"""
Orchestrator：Mining 编排的统一入口（Step 6 薄委托版）。

职责边界
--------
本模块是 **薄委托包装**：
    - 对外暴露 Orchestrator 类
    - 内部直接调用现有 Coordinator + run_generation
    - 不重写 15 状态机（全重写留待后续 Step）

设计动机
--------
Step 6 的目标是把 search/miner.py + search/coordinator.py +
search/generation.py 合并为单一 orchestrator，用 AlphaCandidate 贯穿。

本次只做"接口层"：
    - 新建 Orchestrator，签名与 Coordinator.run 兼容
    - search/miner.py:mine 改为调用 Orchestrator
    - 保留 coordinator.py / generation.py 作为参考实现

这样做的原因：
    - Coordinator.run 有 ~15 个状态（elites / elite_signatures /
      submitted_count / final_concurrency / structure_seen / archive /
      resume / seed 推进 / ETA / _maybe_persist）
    - 一次性重写风险高，且无法在无测试的情况下验证
    - 薄包装先验证接口兼容，全重写留待有测试后

不含：
    - 过滤逻辑（在 pipeline/）
    - 生成逻辑（在 alpha/）
    - 评估逻辑（在 evaluation/）
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from evaluation.record import AlphaRecord
from execution.executor import Executor
from search.coordinator import (
    Coordinator,
    EventHook,
    MiningConfig,
    MiningState,
)


logger = logging.getLogger(__name__)


# ============================================================
# Orchestrator
# ============================================================

class Orchestrator:
    """
    Mining 编排器（薄委托版）。

    内部持有 Coordinator，run() 直接委托。

    Parameters
    ----------
    executor : Executor
    config   : MiningConfig
    on_event : EventHook | None
    agent    : LLMRefiningAgent | None
    on_record_complete : Callable[[AlphaRecord], None] | None
    """

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

        # 内部委托给现有 Coordinator
        self._coordinator = Coordinator(
            executor=executor,
            config=config,
            on_event=on_event,
            agent=agent,
            on_record_complete=on_record_complete,
        )

    # --------------------------------------------------------
    # 主入口（签名与 Coordinator.run 兼容）
    # --------------------------------------------------------

    def run(
        self,
        initial_state: Optional[MiningState] = None,
        on_generation: Optional[Callable[[MiningState], None]] = None,
    ) -> MiningState:
        """
        执行 mining。

        直接委托给 Coordinator.run。
        """
        logger.info(
            "Orchestrator.run: generations=%d population=%d "
            "agent=%s",
            self._config.generations,
            self._config.population_size,
            "ENABLED" if self._agent is not None else "disabled",
        )
        return self._coordinator.run(
            initial_state=initial_state,
            on_generation=on_generation,
        )

    # --------------------------------------------------------
    # 属性（供上层读取）
    # --------------------------------------------------------

    @property
    def config(self) -> MiningConfig:
        return self._config

    @property
    def coordinator(self) -> Coordinator:
        """暴露内部 Coordinator（供测试 / 调试）。"""
        return self._coordinator
