# execution/task.py
# -*- coding: utf-8 -*-

"""
Task：一次 Simulation 任务。

Step 10c 新增
-------------
- source 字段：区分来源
    "GENERATED"  — 由 generate_initial_alpha / mutation 产出
    "AI_REFINED" — 由 LLM Refining Agent 修复产出
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ============================================================
# TaskState
# ============================================================

class TaskState(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"


# ============================================================
# task_id 生成器
# ============================================================

_task_counter = itertools.count(1)


def new_task_id() -> str:
    return f"t{next(_task_counter):06d}"


# ============================================================
# Task
# ============================================================

@dataclass
class Task:
    expression: str
    task_id: str = field(default_factory=new_task_id)

    state: TaskState = TaskState.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    attempts: int = 0

    # Step 10c：来源标记
    source: str = "GENERATED"

    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # --------------------------------------------------------
    # 便捷查询
    # --------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.state in (TaskState.COMPLETED, TaskState.FAILED)

    @property
    def duration(self) -> Optional[float]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at