# execution/__init__.py
# -*- coding: utf-8 -*-

"""
execution：并发 Simulation 执行层。

子模块：
    task         : Task + TaskState
    queue        : TaskQueue（线程安全）
    rate_limiter : RateLimiter（最小间隔节流）
    concurrency  : ConcurrencyController / AdaptiveConcurrencyController
    worker       : Worker（单 worker 执行循环）
    executor     : Executor（统一入口）
"""

from execution.concurrency import (
    AdaptiveConcurrencyController,
    ConcurrencyController,
    ConcurrencySettings,
    build_concurrency_controller,
)
from execution.executor import Executor
from execution.queue import TaskQueue
from execution.rate_limiter import RateLimiter
from execution.task import Task, TaskState, new_task_id
from execution.worker import Worker

__all__ = [
    "Task", "TaskState", "new_task_id",
    "TaskQueue",
    "RateLimiter",
    "ConcurrencyController",
    "AdaptiveConcurrencyController",
    "ConcurrencySettings",
    "build_concurrency_controller",
    "Worker",
    "Executor",
]