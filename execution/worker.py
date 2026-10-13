# execution/worker.py
# -*- coding: utf-8 -*-

"""
单 worker 执行循环。

职责边界
--------
    - 从 TaskQueue 取 Task
    - 通过预认证 Session 构造 SimulationRunner
    - 调用 runner.simulate_alpha(expression)
    - 更新 Task 状态
    - 支持 Task 级别重试
    - **Step 8a**：成功 / 429 时通知并发控制器

日志上下文
----------
进入 _process 时用 utils.logging_context.set_log_context 注入
(worker, task_id, expr)，退出时 reset。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

import requests

from brain.client import BrainClient
from brain.polling import SimulationPoller
from brain.simulation import (
    ConcurrencyLimitExceeded,
    SimulationRunner,
)
from execution.queue import TaskQueue
from execution.rate_limiter import RateLimiter
from execution.task import Task, TaskState
from utils.logging_context import (
    reset_log_context,
    set_log_context,
)

if TYPE_CHECKING:
    from brain.models import SimulationSettings
    from config.settings import BrainEndpoints, PollingSettings


logger = logging.getLogger(__name__)


# ============================================================
# 表达式缩略
# ============================================================

def _short_expr(expr: str, limit: int = 40) -> str:
    if len(expr) <= limit:
        return expr
    return expr[: limit - 3] + "..."


# ============================================================
# Worker
# ============================================================

class Worker(threading.Thread):
    def __init__(
        self,
        name: str,
        task_queue: TaskQueue,
        session: requests.Session,
        endpoints: "BrainEndpoints",
        polling: "PollingSettings",
        simulation_settings: "SimulationSettings",
        rate_limiter: Optional[RateLimiter] = None,
        max_retries: int = 1,
        retry_backoff: float = 1.0,
        concurrency_backoff: float = 30.0,
        concurrency_controller=None,
        on_task_complete: Optional[Callable[[Task], None]] = None,
    ) -> None:
        super().__init__(name=name, daemon=True)

        self._task_queue = task_queue
        self._rate_limiter = rate_limiter
        self._max_retries = max(0, max_retries)
        self._retry_backoff = max(0.0, retry_backoff)
        self._concurrency_backoff = max(0.0, concurrency_backoff)
        self._concurrency_controller = concurrency_controller
        self._on_task_complete = on_task_complete

        client = BrainClient(session, endpoints, polling)
        poller = SimulationPoller(client, endpoints, polling)
        self._runner = SimulationRunner(
            client=client,
            poller=poller,
            settings=simulation_settings,
        )

    # --------------------------------------------------------
    # 主循环
    # --------------------------------------------------------

    def run(self) -> None:
        while True:
            task = self._task_queue.get()
            if task is None:
                self._task_queue.task_done()
                return
            try:
                self._process(task)
            finally:
                self._task_queue.task_done()

    # --------------------------------------------------------
    # 处理单个 Task
    # --------------------------------------------------------

    def _process(self, task: Task) -> None:
        token = set_log_context(
            worker=self.name,
            task_id=task.task_id,
            expr=_short_expr(task.expression),
        )
        try:
            self._process_with_context(task)
        finally:
            reset_log_context(token)

    def _process_with_context(self, task: Task) -> None:
        task.state = TaskState.RUNNING
        task.started_at = time.time()

        last_error: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            task.attempts += 1

            try:
                if self._rate_limiter is not None:
                    self._rate_limiter.acquire()

                result = self._runner.simulate_alpha(task.expression)

                if result is not None:
                    task.state = TaskState.COMPLETED
                    task.result = result
                    task.error = None
                    task.finished_at = time.time()
                    duration = task.finished_at - task.started_at
                    logger.info(
                        "→ COMPLETED (attempts=%d, %.0fs)",
                        task.attempts, duration,
                    )

                    # ---- 任务完成回调 ----
                    if self._on_task_complete is not None:
                        try:
                            self._on_task_complete(task)
                        except Exception:
                            logger.exception(
                                "on_task_complete 回调失败"
                            )

                    # ---- Step 8a：通知控制器成功 ----
                    if self._concurrency_controller is not None:
                        try:
                            self._concurrency_controller.on_success()
                        except Exception:
                            logger.exception(
                                "controller.on_success 失败"
                            )
                    return

                last_error = "Simulation returned None"

            except ConcurrencyLimitExceeded as e:
                last_error = f"concurrency limit: {e}"

                # ---- Step 8a：通知控制器降并发 ----
                if self._concurrency_controller is not None:
                    try:
                        self._concurrency_controller \
                            .on_concurrency_limit()
                    except Exception:
                        logger.exception(
                            "controller.on_concurrency_limit 失败"
                        )

                logger.warning(
                    "concurrency limit hit, "
                    "backoff %.1fs (attempt %d/%d)",
                    self._concurrency_backoff,
                    attempt + 1, self._max_retries + 1,
                )
                if attempt < self._max_retries:
                    time.sleep(self._concurrency_backoff)
                    continue

            except Exception as e:
                last_error = repr(e)
                logger.exception(
                    "attempt %d failed", attempt + 1,
                )

            if attempt < self._max_retries:
                backoff = self._retry_backoff * (2 ** attempt)
                logger.warning(
                    "retrying in %.2fs (%d/%d)",
                    backoff, attempt + 1, self._max_retries,
                )
                time.sleep(backoff)

        task.state = TaskState.FAILED
        task.error = last_error
        task.finished_at = time.time()
        logger.warning("→ FAILED: %s", last_error)
