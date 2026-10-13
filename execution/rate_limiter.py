# execution/rate_limiter.py
# -*- coding: utf-8 -*-

"""
最小间隔节流器。

语义
----
rate_per_second = N  → 任意两次 acquire() 之间至少间隔 1/N 秒。
rate_per_second = None → 不做任何节流。

线程安全
--------
多个 worker 并发调用 acquire() 时，通过单锁串行化"预约下一个时间槽"，
sleep 在锁外执行，避免阻塞其它调用者的时间槽预约。

⚠️ 这是 **工程策略**，不是 BRAIN 官方 API 语义。
   真实并发上限 / 速率上限应由 Capability Discovery + 服务端 429 反馈决定。
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class RateLimiter:
    def __init__(self, rate_per_second: Optional[float]) -> None:
        if rate_per_second is not None and rate_per_second <= 0:
            raise ValueError(
                "rate_per_second must be > 0 or None"
            )
        self._rate = rate_per_second
        self._min_interval = (
            1.0 / rate_per_second if rate_per_second else 0.0
        )
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def acquire(self) -> None:
        if self._rate is None:
            return

        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = (
                max(now, self._next_allowed) + self._min_interval
            )

        if wait > 0:
            time.sleep(wait)