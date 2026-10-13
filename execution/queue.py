# execution/queue.py
# -*- coding: utf-8 -*-

"""
线程安全 Task 队列。

职责边界
--------
只是 queue.Queue 的薄封装 + 类型提示 + sentinel 语义。

sentinel 约定
-------------
put_sentinel() 向队列放入 None；
worker 读到 None 即视为“停止信号”。
"""

from __future__ import annotations

import queue
from typing import Optional

from execution.task import Task


class TaskQueue:
    def __init__(self) -> None:
        self._q: "queue.Queue[Optional[Task]]" = queue.Queue()

    # --------------------------------------------------------
    # 生产者
    # --------------------------------------------------------

    def put(self, task: Task) -> None:
        self._q.put(task)

    def put_sentinel(self) -> None:
        self._q.put(None)

    # --------------------------------------------------------
    # 消费者
    # --------------------------------------------------------

    def get(self) -> Optional[Task]:
        """阻塞获取；返回 None 表示 sentinel。"""
        return self._q.get()

    def task_done(self) -> None:
        self._q.task_done()

    # --------------------------------------------------------
    # 同步
    # --------------------------------------------------------

    def join(self) -> None:
        """阻塞直到队列中所有 item 都被 task_done()。"""
        self._q.join()

    # --------------------------------------------------------
    # 只读属性
    # --------------------------------------------------------

    def qsize(self) -> int:
        return self._q.qsize()