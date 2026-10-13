# execution/executor.py
# -*- coding: utf-8 -*-

"""
执行器：并发执行一批 Task。

职责边界
--------
    - 创建 / 认证 N 个 Session
    - 创建 TaskQueue、启动 N 个 Worker
    - 阻塞直到队列清空
    - 发送 sentinel、回收线程
    - 返回被原地更新状态的 tasks

Step 9
------
    - 暴露 concurrency_controller 属性，供上层在 run 结束后读
      current() 值并持久化。

不含：
    - Task 生成（由 search/ 提供）
    - 结果解析（由 evaluation/ 提供）
    - 通知（由 notification/ 提供）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional

from brain.authentication import create_authenticated_session
from execution.queue import TaskQueue
from execution.rate_limiter import RateLimiter
from execution.task import Task
from execution.worker import Worker

if TYPE_CHECKING:
    from brain.models import SimulationSettings
    from config.settings import (
        BrainCredentials,
        BrainEndpoints,
        PollingSettings,
    )


logger = logging.getLogger(__name__)


class Executor:
    def __init__(
        self,
        credentials: "BrainCredentials",
        endpoints: "BrainEndpoints",
        polling: "PollingSettings",
        simulation_settings: "SimulationSettings",
        concurrency,  # ConcurrencyController | AdaptiveConcurrencyController
        rate_limiter: Optional[RateLimiter] = None,
        max_retries: int = 1,
        retry_backoff: float = 1.0,
        concurrency_backoff: float = 30.0,
        on_task_complete: Optional[Callable[[Task], None]] = None,
    ) -> None:
        self._credentials = credentials
        self._endpoints = endpoints
        self._polling = polling
        self._simulation_settings = simulation_settings
        self._concurrency = concurrency
        self._rate_limiter = rate_limiter
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._concurrency_backoff = concurrency_backoff
        self._on_task_complete = on_task_complete

    # --------------------------------------------------------
    # 属性（Step 9）
    # --------------------------------------------------------

    @property
    def concurrency_controller(self):
        return self._concurrency

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def run(self, tasks: List[Task]) -> List[Task]:
        if not tasks:
            return tasks

        n_from_controller = self._concurrency.effective_workers()
        n = max(1, min(n_from_controller, len(tasks)))

        logger.info(
            "Executor.run: tasks=%d workers=%d (controller=%d)",
            len(tasks), n, n_from_controller,
        )

        # -------- 1. 预先认证 N 个 Session --------
        sessions = []
        try:
            for _ in range(n):
                sessions.append(
                    create_authenticated_session(
                        self._credentials,
                        self._endpoints,
                        request_timeout=self._polling.request_timeout,
                    )
                )
        except Exception:
            for s in sessions:
                try:
                    s.close()
                except Exception:
                    pass
            raise

        # -------- 2. 入队 --------
        task_queue = TaskQueue()
        for t in tasks:
            task_queue.put(t)

        # -------- 3. 启动 workers --------
        workers: List[Worker] = []
        for i in range(n):
            w = Worker(
                name=f"worker-{i}",
                task_queue=task_queue,
                session=sessions[i],
                endpoints=self._endpoints,
                polling=self._polling,
                simulation_settings=self._simulation_settings,
                rate_limiter=self._rate_limiter,
                max_retries=self._max_retries,
                retry_backoff=self._retry_backoff,
                concurrency_backoff=self._concurrency_backoff,
                concurrency_controller=self._concurrency,
                on_task_complete=self._on_task_complete,
            )
            w.start()
            workers.append(w)

        # -------- 4. 等待所有任务完成 --------
        try:
            task_queue.join()
        finally:
            for _ in workers:
                task_queue.put_sentinel()
            for w in workers:
                w.join()

            # 显式关闭所有 session，避免资源泄漏
            for s in sessions:
                try:
                    s.close()
                except Exception:
                    logger.exception("关闭 session 失败")

        return tasks
