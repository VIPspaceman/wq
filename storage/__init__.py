# storage/__init__.py
# -*- coding: utf-8 -*-

"""
storage：持久化层。

子模块：
    repository  : AlphaRepository 抽象
    json_store  : JsonAlphaRepository
    csv_store   : CsvAlphaRepository
    checkpoint  : Checkpoint + CheckpointStore
    experiment  : ExperimentMeta / Layout / Repository
    store       : MiningStore（facade）

⚠️ 本层：
    - 与业务逻辑解耦
    - 业务代码不直接 open() 文件
    - 通过 Repository 接口访问
"""

from storage.checkpoint import Checkpoint, CheckpointStore
from storage.csv_store import CsvAlphaRepository
from storage.experiment import (
    ExperimentLayout,
    ExperimentMeta,
    ExperimentRepository,
    new_experiment_id,
)
from storage.json_store import JsonAlphaRepository
from storage.repository import AlphaRepository
from storage.store import MiningStore

__all__ = [
    "AlphaRepository",
    "JsonAlphaRepository",
    "CsvAlphaRepository",
    "Checkpoint",
    "CheckpointStore",
    "ExperimentMeta",
    "ExperimentLayout",
    "ExperimentRepository",
    "new_experiment_id",
    "MiningStore",
]