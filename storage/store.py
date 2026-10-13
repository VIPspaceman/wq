# storage/store.py
# -*- coding: utf-8 -*-

"""
MiningStore：一次 Mining 运行的全部落盘操作。

Step 10d 更新
-------------
- 新增 ai_fixed_alphas.json（只含 AI_FIXED 标签的记录）
"""

from __future__ import annotations

import json
import logging
import os
import socket
from typing import Optional

from alpha.parser import parse_expression
from config.settings import OutputPaths
from core.hashing import expression_hash
from evaluation.classifier import Classification
from evaluation.record import AlphaRecord, build_record
from execution.task import Task, TaskState
from search.coordinator import MiningState
from storage.checkpoint import CheckpointStore
from storage.csv_store import CsvAlphaRepository
from storage.database import WQDatabase
from storage.json_store import JsonAlphaRepository


logger = logging.getLogger(__name__)


class MiningStore:
    def __init__(
        self,
        paths: OutputPaths,
        elite_size: int = 8,
        db_path: Optional[str] = "data/wq.db",
    ) -> None:
        self._paths = paths
        self._elite_size = elite_size

        # 实验上下文（可选）
        self._experiment_id: Optional[str] = None
        self._current_generation: int = 0

        # SQLite 双写（可选增强，失败降级为仅写文件）
        self._db: Optional[WQDatabase] = None
        self._device_id: str = ""
        if db_path is not None:
            try:
                self._db = WQDatabase(db_path)
                self._db.init_schema()
                self._device_id = self._resolve_device_id()
            except Exception:
                logger.exception("WQDatabase 初始化失败，降级为仅写文件")
                self._db = None
                self._device_id = ""

        self._results_json = JsonAlphaRepository(
            paths.path(paths.results_json)
        )
        self._results_csv = CsvAlphaRepository(
            paths.path(paths.results_csv)
        )
        self._candidate_json = JsonAlphaRepository(
            paths.path(paths.candidate_json)
        )
        self._strong_json = JsonAlphaRepository(
            paths.path(paths.strong_json)
        )
        self._final_json = JsonAlphaRepository(
            paths.path(paths.final_json)
        )
        self._elite_json = JsonAlphaRepository(
            paths.path(paths.elite_json)
        )
        # Step 10d
        self._ai_fixed_json = JsonAlphaRepository(
            paths.path(paths.ai_fixed_json)
        )
        self._checkpoint = CheckpointStore(
            paths.path(paths.checkpoint)
        )

    # --------------------------------------------------------
    # 实验上下文
    # --------------------------------------------------------

    def set_experiment(
        self,
        experiment_id: str,
        random_seed: int = 0,
        config_snapshot: Optional[dict] = None,
    ) -> None:
        """设置当前 experiment_id，并在 DB 中创建 experiment 记录。"""
        self._experiment_id = experiment_id
        if self._db is None:
            return
        try:
            self._db.create_experiment(
                experiment_id=experiment_id,
                device_id=self._device_id,
                random_seed=random_seed or 0,
                config_snapshot=config_snapshot or {},
            )
        except Exception:
            logger.exception("create_experiment 失败")

    def set_generation(self, generation: int) -> None:
        """设置当前 generation（用于 DB 双写）。"""
        self._current_generation = generation

    # --------------------------------------------------------
    # DB 查询
    # --------------------------------------------------------

    def load_all_expr_hashes(self) -> set:
        """从 DB 预加载所有历史 expr_hash；失败返回空集合。"""
        if self._db is None:
            return set()
        try:
            return self._db.load_all_expr_hashes()
        except Exception:
            logger.exception("load_all_expr_hashes 失败")
            return set()

    def is_evaluated_in_db(self, expr_hash: str) -> bool:
        """查询 DB 中是否已评估；失败返回 False。"""
        if self._db is None:
            return False
        try:
            return self._db.is_evaluated(expr_hash)
        except Exception:
            logger.exception("is_evaluated_in_db 失败: %s", expr_hash)
            return False

    # --------------------------------------------------------
    # Worker 回调
    # --------------------------------------------------------

    def on_task_complete(self, task: Task) -> None:
        """Worker 每条 task 完成时调用（成功或失败）。

        - 更新 DB 的 pending_tasks 状态
        - 若 task 成功，立即 build_record 并写 evaluations
        - 失败不抛异常
        """
        try:
            state_str = (
                "COMPLETED" if task.state == TaskState.COMPLETED
                else "FAILED"
            )
            expr_h = expression_hash(task.expression)

            # 1. 记录/更新 pending_tasks
            if self._db is not None:
                try:
                    self._db.save_task(
                        task_id=task.task_id,
                        experiment_id=self._experiment_id or "unknown",
                        generation=self._current_generation,
                        expression=task.expression,
                        expr_hash=expr_h,
                        source=task.source,
                        state=state_str,
                    )
                    self._db.update_task_state(task.task_id, state_str)
                except Exception:
                    logger.exception("save_task 失败")

            # 2. 成功的 task：立即写 evaluations
            if (
                task.state == TaskState.COMPLETED
                and task.result is not None
                and self._db is not None
            ):
                try:
                    node = parse_expression(task.expression)
                    if node is not None:
                        record = build_record(node, task.result)
                        ok = self._db.save_record(
                            record=record.to_dict(),
                            experiment_id=(
                                self._experiment_id or "unknown"
                            ),
                            device_id=self._device_id,
                            generation=self._current_generation,
                        )
                        if ok:
                            logger.info(
                                "on_task_complete: DB 写入 %s",
                                record.alpha_id,
                            )
                        else:
                            logger.warning(
                                "on_task_complete: DB 写入失败 %s",
                                record.alpha_id,
                            )
                except Exception:
                    logger.exception(
                        "on_task_complete 写 evaluations 失败"
                    )
        except Exception:
            logger.exception("on_task_complete 失败")

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def persist(self, state: MiningState) -> None:
        # 重复调用检测：同一 state 对象、同一代数则跳过
        sig = (id(state), state.generation)
        if sig == getattr(self, "_last_persist_sig", None):
            return
        self._last_persist_sig = sig
        try:
            self._persist_impl(state)
        except Exception:
            logger.exception("MiningStore.persist 失败")

    # --------------------------------------------------------
    # 单条追加（容错）
    # --------------------------------------------------------

    def append_record(self, record: AlphaRecord) -> None:
        """追加一条记录到 results_json 和 results_csv。

        使用原子写，失败不抛异常。
        """
        try:
            self._append_record_impl(record)
        except Exception:
            logger.exception("MiningStore.append_record 失败")

    def _append_record_impl(self, record: AlphaRecord) -> None:
        json_path = self._paths.path(self._paths.results_json)
        csv_path = self._paths.path(self._paths.results_csv)

        # -------- 1. 读取现有 JSON --------
        existing: list = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    existing = data
            except Exception:
                logger.exception(
                    "append_record: 读取 %s 失败，视为空列表",
                    json_path,
                )
                existing = []

        # -------- 2. 去重 --------
        def _key(d: dict) -> str:
            return str(
                d.get("alpha_id") or d.get("expression") or ""
            )

        new_key = _key(record.to_dict())
        if new_key and any(_key(d) == new_key for d in existing):
            logger.debug(
                "append_record: 已存在，跳过 %s", new_key,
            )
            return

        # -------- 3. 追加 --------
        existing.append(record.to_dict())

        # -------- 4. 原子写回 JSON --------
        tmp_json = json_path + ".tmp"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        os.replace(tmp_json, json_path)

        # -------- 5. 原子写回 CSV --------
        records = [AlphaRecord.from_dict(d) for d in existing]
        tmp_csv = csv_path + ".tmp"
        self._write_csv_atomic(records, tmp_csv)
        os.replace(tmp_csv, csv_path)

        # -------- 6. SQLite 双写（可选增强） --------
        self._save_record_to_db(record)

    def _write_csv_atomic(
        self,
        records: list,
        path: str,
    ) -> None:
        """把 records 写入指定路径（CSV）。"""
        import csv as _csv

        if not records:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("")
            return

        fieldnames = list(records[0].to_dict().keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_dict())

    # --------------------------------------------------------
    # 内部
    # --------------------------------------------------------

    def _resolve_device_id(self) -> str:
        """解析本机 device_id（写入 data/device_id.txt）。"""
        try:
            name = socket.gethostname() or "unknown-device"
        except Exception:
            name = "unknown-device"
        try:
            device_file = os.path.join("data", "device_id.txt")
            parent = os.path.dirname(os.path.abspath(device_file))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            if not os.path.exists(device_file):
                with open(device_file, "w", encoding="utf-8") as f:
                    f.write(name)
            else:
                try:
                    with open(device_file, "r", encoding="utf-8") as f:
                        cached = f.read().strip()
                    if cached:
                        name = cached
                except Exception:
                    logger.exception("读取 device_id.txt 失败")
        except Exception:
            logger.exception("写入 device_id.txt 失败")
        if self._db is None:
            return ""
        try:
            return self._db.get_or_create_device(name)
        except Exception:
            logger.exception("get_or_create_device 失败")
            return ""

    def _save_record_to_db(self, record: AlphaRecord) -> None:
        """将单条记录写入 SQLite；失败不阻塞主流程。"""
        if self._db is None:
            return
        try:
            self._db.save_record(
                record=record.to_dict(),
                experiment_id=self._experiment_id or "unknown",
                device_id=self._device_id,
                generation=self._current_generation or 0,
            )
        except Exception:
            logger.exception("DB 写入失败")

    def _persist_impl(self, state: MiningState) -> None:
        all_records = state.archive.all()

        self._results_json.save_all(all_records)
        self._results_csv.save_all(all_records)

        candidates = [
            r for r in all_records
            if r.classification == Classification.CANDIDATE
        ]
        strong = [
            r for r in all_records
            if r.classification
            in (Classification.STRONG, Classification.FINAL)
        ]
        final = [
            r for r in all_records
            if r.classification == Classification.FINAL
        ]
        elites = state.archive.best_by_score(self._elite_size)

        # Step 10d：AI 修复成功
        ai_fixed = [r for r in all_records if r.has_tag("AI_FIXED")]

        self._candidate_json.save_all(candidates)
        self._strong_json.save_all(strong)
        self._final_json.save_all(final)
        self._elite_json.save_all(elites)
        self._ai_fixed_json.save_all(ai_fixed)

        self._checkpoint.save(
            state,
            concurrency_current=state.final_concurrency,
        )

        # SQLite 双写（可选增强）
        if self._db is not None:
            for record in state.archive.all():
                try:
                    self._db.save_record(
                        record=record.to_dict(),
                        experiment_id=self._experiment_id or "unknown",
                        device_id=self._device_id,
                        generation=self._current_generation,
                    )
                except Exception:
                    logger.exception("persist DB 写入失败")

        # 写 population snapshot（用于 resume）
        try:
            from alpha.ast import node_to_expr
            population_exprs = [
                node_to_expr(n) for n in state.population
            ]
            elites_ids = [
                r.alpha_id for r in state.elites if r.alpha_id
            ]
            self._db.save_snapshot(
                experiment_id=(
                    self._experiment_id or "unknown"
                ),
                generation=state.generation,
                population=population_exprs,
                elites=elites_ids,
            )
        except Exception:
            logger.exception("save_snapshot 失败")

        logger.info(
            "MiningStore.persist: total=%d candidate=%d "
            "strong=%d final=%d elite=%d ai_fixed=%d cc=%s",
            len(all_records),
            len(candidates),
            len(strong),
            len(final),
            len(elites),
            len(ai_fixed),
            state.final_concurrency,
        )
