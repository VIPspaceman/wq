# -*- coding: utf-8 -*-

"""
storage/checkpoint.py 单元测试。

重点验证 P1-8 修复：
    Checkpoint 新增 experiment_id 字段，且向后兼容旧文件。
"""

from __future__ import annotations

import json
import os

import pytest

from search.coordinator import MiningState
from storage.checkpoint import Checkpoint, CheckpointStore


# ============================================================
# 基本存取
# ============================================================

def test_save_and_load(tmp_path) -> None:
    path = str(tmp_path / "ckpt.json")
    store = CheckpointStore(path)

    state = MiningState()
    state.generation = 3

    store.save(state, concurrency_current=2, experiment_id="exp_abc")

    ckpt = store.load()
    assert ckpt is not None
    assert ckpt.generation == 3
    assert ckpt.concurrency_current == 2
    assert ckpt.experiment_id == "exp_abc"


def test_load_missing_file(tmp_path) -> None:
    path = str(tmp_path / "nope.json")
    store = CheckpointStore(path)
    assert store.load() is None


# ============================================================
# 向后兼容：旧文件无 experiment_id
# ============================================================

def test_load_legacy_without_experiment_id(tmp_path) -> None:
    path = str(tmp_path / "legacy.json")
    legacy = {
        "generation": 1,
        "evaluated_hashes": [],
        "archive": [],
        "population": [],
        "elites": [],
        "timestamp": "2026-01-01 00:00:00",
        "concurrency_current": 1,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(legacy, f)

    store = CheckpointStore(path)
    ckpt = store.load()
    assert ckpt is not None
    assert ckpt.generation == 1
    assert ckpt.experiment_id is None


def test_load_corrupt_file(tmp_path) -> None:
    path = str(tmp_path / "corrupt.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")

    store = CheckpointStore(path)
    assert store.load() is None
