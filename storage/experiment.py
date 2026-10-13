# storage/experiment.py
# -*- coding: utf-8 -*-

"""
Experiment 元数据与索引。

职责边界
--------
    - ExperimentMeta：单个 experiment 的元数据
    - ExperimentLayout：目录布局
    - ExperimentRepository：meta 读写 + index 维护

设计
----
每个 experiment 一个子目录：
    <base>/experiments/<experiment_id>/
        ├── meta.json
        ├── alpha_checkpoint.json
        ├── alpha_results.json / .csv
        └── ...

<base>/experiments/index.json 保存所有 experiment_id 列表。

现有 OutputPaths / MiningStore / CheckpointStore **不需要改**；
只需把 base_dir 指向该子目录。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from storage.json_store import atomic_write_json


# ============================================================
# ID 生成
# ============================================================

def new_experiment_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"exp_{ts}_{suffix}"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# ExperimentMeta
# ============================================================

@dataclass
class ExperimentMeta:
    experiment_id: str
    created_at: str
    updated_at: str

    random_seed: int
    config_snapshot: Dict[str, Any]

    status: str = "RUNNING"        # RUNNING / COMPLETED / FAILED / INTERRUPTED
    generation: int = -1
    total_evaluated: int = 0
    archive_size: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    best_alpha: Optional[Dict[str, Any]] = None
    final_concurrency: Optional[int] = None

    # --------------------------------------------------------
    # 序列化
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentMeta":
        return cls(
            experiment_id=d["experiment_id"],
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
            random_seed=int(d.get("random_seed", 0)),
            config_snapshot=dict(d.get("config_snapshot", {})),
            status=d.get("status", "RUNNING"),
            generation=int(d.get("generation", -1)),
            total_evaluated=int(d.get("total_evaluated", 0)),
            archive_size=int(d.get("archive_size", 0)),
            counts=dict(d.get("counts", {})),
            best_alpha=d.get("best_alpha"),
            final_concurrency=d.get("final_concurrency"),
        )

    # --------------------------------------------------------
    # 工厂
    # --------------------------------------------------------

    @classmethod
    def new(
        cls,
        experiment_id: str,
        random_seed: int,
        config_snapshot: Dict[str, Any],
    ) -> "ExperimentMeta":
        now = _now()
        return cls(
            experiment_id=experiment_id,
            created_at=now,
            updated_at=now,
            random_seed=random_seed,
            config_snapshot=config_snapshot,
            status="RUNNING",
        )


# ============================================================
# 目录布局
# ============================================================

class ExperimentLayout:
    def __init__(self, base_dir: str) -> None:
        self._base = base_dir
        self._experiments_dir = os.path.join(base_dir, "experiments")
        self._index_path = os.path.join(
            self._experiments_dir, "index.json"
        )

    @property
    def base_dir(self) -> str:
        return self._base

    @property
    def experiments_dir(self) -> str:
        return self._experiments_dir

    @property
    def index_path(self) -> str:
        return self._index_path

    def experiment_dir(self, experiment_id: str) -> str:
        return os.path.join(self._experiments_dir, experiment_id)

    def meta_path(self, experiment_id: str) -> str:
        return os.path.join(
            self.experiment_dir(experiment_id), "meta.json"
        )

    def checkpoint_path(self, experiment_id: str) -> str:
        return os.path.join(
            self.experiment_dir(experiment_id),
            "alpha_checkpoint.json",
        )

    def ensure(self) -> None:
        os.makedirs(self._experiments_dir, exist_ok=True)


# ============================================================
# Repository
# ============================================================

class ExperimentRepository:
    def __init__(self, layout: ExperimentLayout) -> None:
        self._layout = layout

    # --------------------------------------------------------
    # 索引
    # --------------------------------------------------------

    def list_ids(self) -> List[str]:
        path = self._layout.index_path
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("experiment_ids", []))

    def _add_to_index(self, experiment_id: str) -> None:
        ids = self.list_ids()
        if experiment_id in ids:
            return
        ids.append(experiment_id)
        self._layout.ensure()
        atomic_write_json(
            self._layout.index_path,
            {"experiment_ids": ids},
        )

    # --------------------------------------------------------
    # Meta
    # --------------------------------------------------------

    def save_meta(self, meta: ExperimentMeta) -> None:
        meta.updated_at = _now()
        os.makedirs(
            self._layout.experiment_dir(meta.experiment_id),
            exist_ok=True,
        )
        atomic_write_json(
            self._layout.meta_path(meta.experiment_id),
            meta.to_dict(),
        )
        self._add_to_index(meta.experiment_id)

    def load_meta(
        self, experiment_id: str,
    ) -> Optional[ExperimentMeta]:
        path = self._layout.meta_path(experiment_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentMeta.from_dict(data)

    def exists(self, experiment_id: str) -> bool:
        return os.path.exists(
            self._layout.meta_path(experiment_id)
        )

    # --------------------------------------------------------
    # 列表（简表）
    # --------------------------------------------------------

    def list_metas(self) -> List[ExperimentMeta]:
        result: List[ExperimentMeta] = []
        for eid in self.list_ids():
            meta = self.load_meta(eid)
            if meta is not None:
                result.append(meta)
        return result
