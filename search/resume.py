# search/resume.py
# -*- coding: utf-8 -*-

"""
从 Checkpoint 恢复 MiningState。

职责边界
--------
    - 反序列化 Checkpoint 中的表达式字符串
    - 重建 AlphaRecord / AlphaArchive
    - 返回可传给 Coordinator.run(initial_state=...) 的 MiningState

不含：
    - 文件 IO（由 storage/ 负责）
    - 事件钩子
    - 并发控制
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from alpha.ast import Node
from alpha.parser import parse_expression
from evaluation.record import AlphaRecord
from search.coordinator import MiningState
from storage.checkpoint import Checkpoint

if TYPE_CHECKING:
    from storage.database import WQDatabase


logger = logging.getLogger(__name__)


def build_state_from_checkpoint(
    ckpt: Checkpoint,
) -> MiningState:
    state = MiningState()

    # 代数
    state.generation = ckpt.generation

    # 已评估 hash
    state.evaluated_hashes = set(ckpt.evaluated_hashes)

    # 存档
    for d in ckpt.archive:
        try:
            record = AlphaRecord.from_dict(d)
        except Exception:
            logger.warning(
                "跳过无法反序列化的 archive record: %r", d
            )
            continue
        state.archive.add(record)

    # 种群
    bad = 0
    for expr in ckpt.population:
        node: Node | None = parse_expression(expr)
        if node is None:
            bad += 1
            continue
        state.population.append(node)
    if bad:
        logger.warning(
            "恢复种群时 %d 个表达式无法解析，已跳过", bad
        )

    # 精英
    for d in ckpt.elites:
        try:
            state.elites.append(AlphaRecord.from_dict(d))
        except Exception:
            logger.warning(
                "跳过无法反序列化的 elite record"
            )

    logger.info(
        "Resumed state: generation=%d population=%d "
        "archive=%d elites=%d hashes=%d",
        state.generation,
        len(state.population),
        len(state.archive),
        len(state.elites),
        len(state.evaluated_hashes),
    )

    return state


def build_state_from_db(
    db: "WQDatabase",
    experiment_id: str,
) -> Optional[MiningState]:
    """从 SQLite DB 恢复 MiningState。

    恢复内容：
    - generation: 从 snapshot 或 evaluations 推算
    - population: 从 snapshot 的 population JSON
    - elites: 从 snapshot 的 elites alpha_id 反查
    - archive: 从 evaluations 重建 AlphaRecord
    - evaluated_hashes: 全局（load_all_expr_hashes）

    若该 experiment 在 DB 中完全没记录，返回 None。
    """
    # -------- 1. 检查 experiment 是否存在 --------
    try:
        exp = db.get_experiment(experiment_id)
    except Exception:
        logger.exception("get_experiment 失败: %s", experiment_id)
        exp = None

    try:
        evals = db.get_experiment_evaluations(experiment_id)
    except Exception:
        logger.exception(
            "get_experiment_evaluations 失败: %s", experiment_id
        )
        evals = []

    if exp is None and not evals:
        return None

    state = MiningState()

    # -------- 2. snapshot --------
    snapshot = None
    try:
        snapshot = db.load_latest_snapshot(experiment_id)
    except Exception:
        logger.exception(
            "load_latest_snapshot 失败: %s", experiment_id
        )
        snapshot = None

    if snapshot:
        try:
            state.generation = int(snapshot.get("generation", -1))
        except Exception:
            logger.exception("解析 snapshot.generation 失败")
            state.generation = -1
        for expr in snapshot.get("population", []) or []:
            try:
                n = parse_expression(expr)
                if n is not None:
                    state.population.append(n)
            except Exception:
                logger.exception("解析 snapshot population 失败")

    # -------- 3. 从 evaluations 重建 archive --------
    for e in evals:
        try:
            d = {
                "expression": e.get("expression") or "",
                "alpha_id": e.get("alpha_id") or "",
                "sharpe": e.get("sharpe") or 0.0,
                "fitness": e.get("fitness") or 0.0,
                "turnover": e.get("turnover") or 0.0,
                "score": e.get("score") or 0.0,
                "classification": (
                    e.get("classification") or "REJECT"
                ),
                "family": e.get("family") or "OTHER",
                "complexity": e.get("complexity") or 0,
                "depth": e.get("depth") or 0,
                "official_checks": e.get("official_checks") or [],
                "tags": e.get("tags") or [],
                "extra": {},
            }
            record = AlphaRecord.from_dict(d)
            state.archive.add(record)
        except Exception:
            logger.exception("重建 archive 失败")

    # 若 snapshot 缺失 generation，从 evals 推算
    if state.generation < 0 and evals:
        try:
            state.generation = max(
                (e.get("generation") or 0) for e in evals
            )
        except Exception:
            logger.exception("从 evals 推算 generation 失败")

    # -------- 4. 全局 evaluated_hashes --------
    try:
        state.evaluated_hashes = db.load_all_expr_hashes()
    except Exception:
        logger.exception("load_all_expr_hashes 失败")
        state.evaluated_hashes = set()

    # -------- 5. 从 snapshot.elites 反查 archive --------
    if snapshot:
        try:
            alpha_ids = set(snapshot.get("elites", []) or [])
            for r in state.archive.all():
                if r.alpha_id in alpha_ids:
                    state.elites.append(r)
        except Exception:
            logger.exception("恢复 elites 失败")

    logger.info(
        "Resumed state from DB: generation=%d population=%d "
        "archive=%d elites=%d hashes=%d",
        state.generation,
        len(state.population),
        len(state.archive),
        len(state.elites),
        len(state.evaluated_hashes),
    )

    return state
