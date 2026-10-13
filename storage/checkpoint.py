# storage/checkpoint.py
# -*- coding: utf-8 -*-

"""
Checkpoint 存取。

职责边界
--------
    - Checkpoint 数据类
    - CheckpointStore：save(state, concurrency_current) / load()

Step 9
------
新增 concurrency_current 字段，用于自适应并发度的跨运行继承。
旧版 checkpoint（无此字段）向后兼容。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from alpha.ast import node_to_expr
from storage.json_store import atomic_write_json

if TYPE_CHECKING:
    from search.coordinator import MiningState


logger = logging.getLogger(__name__)


# ============================================================
# Checkpoint
# ============================================================

@dataclass
class Checkpoint:
    generation: int
    evaluated_hashes: List[str]
    archive: List[Dict[str, Any]]
    population: List[str]              # 表达式字符串
    elites: List[Dict[str, Any]]
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    )
    # Step 9 新增；旧文件缺失时为 None
    concurrency_current: Optional[int] = None
    # 新增：所属 experiment（旧文件缺失时为 None）
    experiment_id: Optional[str] = None


# ============================================================
# CheckpointStore
# ============================================================

class CheckpointStore:
    def __init__(self, path: str) -> None:
        self._path = path

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    def save(
        self,
        state: MiningState,
        concurrency_current: Optional[int] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        data = {
            "generation": state.generation,
            "evaluated_hashes": list(state.evaluated_hashes),
            "archive": [r.to_dict() for r in state.archive.all()],
            "population": [
                node_to_expr(n) for n in state.population
            ],
            "elites": [r.to_dict() for r in state.elites],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "concurrency_current": concurrency_current,
            "experiment_id": experiment_id,
        }
        atomic_write_json(self._path, data)

    # --------------------------------------------------------
    # 读取
    # --------------------------------------------------------

    def load(self) -> Optional[Checkpoint]:
        if not os.path.exists(self._path):
            return None

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Checkpoint 读取失败（%s），忽略: %s",
                self._path, e,
            )
            return None

        return Checkpoint(
            generation=int(data.get("generation", -1)),
            evaluated_hashes=list(data.get("evaluated_hashes", [])),
            archive=list(data.get("archive", [])),
            population=list(data.get("population", [])),
            elites=list(data.get("elites", [])),
            timestamp=data.get("timestamp", ""),
            concurrency_current=data.get("concurrency_current"),
            experiment_id=data.get("experiment_id"),
        )
