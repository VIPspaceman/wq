# agent/failures.py
# -*- coding: utf-8 -*-

"""
熔断计数器。

职责边界
--------
用 **原始表达式 hash** 作为 key，记录该表达式已被尝试修复的次数。

超过 max_attempts 后：
    - can_attempt() 返回 False
    - 调用方跳过该表达式（不再消耗 LLM 配额 + BRAIN 配额）

内存态
------
本 Step 使用 **进程级内存字典**。
Step 10d 会将其序列化到 checkpoint，实现跨运行持久化。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

from core.hashing import expression_hash


logger = logging.getLogger(__name__)


class RefineOutcome(str, Enum):
    """一次修复尝试的结果。"""
    SUCCESS = "SUCCESS"            # LLM 返回了有效输出
    LLM_FAILED = "LLM_FAILED"      # LLM 调用失败 / 空输出
    INVALID_OUTPUT = "INVALID_OUTPUT"  # 输出清洗后仍不合法
    EXHAUSTED = "EXHAUSTED"        # 达到重试上限
    SKIPPED = "SKIPPED"            # 无需修复


# outcomes 保留上限：避免极端情况下无限增长
_MAX_OUTCOMES_KEPT = 20


@dataclass
class _AttemptRecord:
    attempts: int = 0
    outcomes: list = field(default_factory=list)


class RefineFailureTracker:
    """
    线程安全的熔断计数器。
    """

    def __init__(self, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._max = max_attempts
        self._records: Dict[str, _AttemptRecord] = {}
        self._lock = threading.Lock()

    # --------------------------------------------------------
    # 查询
    # --------------------------------------------------------

    def can_attempt(self, expression: str) -> bool:
        h = expression_hash(expression)
        with self._lock:
            rec = self._records.get(h)
            if rec is None:
                return True
            return rec.attempts < self._max

    def attempts(self, expression: str) -> int:
        h = expression_hash(expression)
        with self._lock:
            rec = self._records.get(h)
            return rec.attempts if rec else 0

    # --------------------------------------------------------
    # 记录
    # --------------------------------------------------------

    def record(
        self,
        expression: str,
        outcome: RefineOutcome,
    ) -> None:
        h = expression_hash(expression)
        with self._lock:
            rec = self._records.get(h)
            if rec is None:
                rec = _AttemptRecord()
                self._records[h] = rec
            rec.attempts += 1
            rec.outcomes.append(outcome.value)
            # 只保留最近 N 条，避免无限增长
            if len(rec.outcomes) > _MAX_OUTCOMES_KEPT:
                del rec.outcomes[:-_MAX_OUTCOMES_KEPT]

        logger.debug(
            "RefineFailureTracker: %s | attempts=%d | outcome=%s",
            expression[:40], rec.attempts, outcome.value,
        )

    # --------------------------------------------------------
    # 重置（供测试或特殊场景用）
    # --------------------------------------------------------

    def reset(self, expression: str) -> None:
        h = expression_hash(expression)
        with self._lock:
            self._records.pop(h, None)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._records)
