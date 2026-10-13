# -*- coding: utf-8 -*-

"""
SQLite 持久化层（仅依赖 Python 标准库）。

设计目标
--------
- 单文件 SQLite 数据库，默认位于 ``data/wq.db``。
- 支持 ``:memory:`` 用于测试。
- 6 张表：devices / experiments / expressions / evaluations /
  population_snapshots / pending_tasks。
- 所有写操作使用 ``with self._conn:`` 自动事务。
- 写方法内部捕获异常并记录日志，不向上抛出。
- JSON 字段统一使用 ``json.dumps(..., ensure_ascii=False)`` 写入，
  读取时 ``json.loads(x) if x else None/[]``。

注意
----
本模块 **禁止** import 任何项目内部模块，保持完全解耦。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


logger = logging.getLogger(__name__)


# ============================================================
# Schema
# ============================================================

_SCHEMA_STATEMENTS: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS devices (
        device_id   TEXT PRIMARY KEY,
        device_name TEXT,
        first_seen  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id       TEXT PRIMARY KEY,
        device_id           TEXT NOT NULL,
        created_at          TIMESTAMP,
        updated_at          TIMESTAMP,
        status              TEXT,
        random_seed         INTEGER,
        config_snapshot     TEXT,
        generation_last     INTEGER,
        concurrency_current INTEGER,
        best_alpha_id       TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS expressions (
        expr_hash       TEXT PRIMARY KEY,
        expression      TEXT NOT NULL,
        first_seen_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        first_device_id TEXT,
        first_exp_id    TEXT,
        family          TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluations (
        alpha_id        TEXT PRIMARY KEY,
        expr_hash       TEXT,
        experiment_id   TEXT,
        device_id       TEXT,
        generation      INTEGER,
        sharpe          REAL,
        fitness         REAL,
        turnover        REAL,
        score           REAL,
        classification  TEXT,
        complexity      INTEGER,
        depth           INTEGER,
        tags            TEXT,
        official_checks TEXT,
        raw_json        TEXT,
        evaluated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS population_snapshots (
        experiment_id TEXT,
        generation    INTEGER,
        population    TEXT,
        elites        TEXT,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (experiment_id, generation)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pending_tasks (
        task_id       TEXT PRIMARY KEY,
        experiment_id TEXT,
        generation    INTEGER,
        expression    TEXT,
        expr_hash     TEXT,
        source        TEXT,
        state         TEXT,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at  TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_eval_expr_hash ON evaluations(expr_hash)",
    "CREATE INDEX IF NOT EXISTS idx_eval_sharpe ON evaluations(sharpe DESC)",
    "CREATE INDEX IF NOT EXISTS idx_eval_classification ON evaluations(classification)",
    "CREATE INDEX IF NOT EXISTS idx_pending_exp ON pending_tasks(experiment_id, state)",
]


_TERMINAL_STATES = {"COMPLETED", "FAILED", "ORPHANED"}


# ============================================================
# Helpers
# ============================================================

def _now() -> str:
    """返回 ISO 格式的当前时间字符串。"""
    return datetime.now().isoformat()


def _dumps(value: Any) -> Optional[str]:
    """将任意值序列化为 JSON 字符串；None 原样返回。"""
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        logger.exception("JSON 序列化失败: %r", value)
        return None


def _loads(value: Any, default: Any = None) -> Any:
    """将 JSON 字符串反序列化；空值返回 default。"""
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        logger.exception("JSON 反序列化失败: %r", value)
        return default


def _expr_hash(expression: str) -> str:
    """与 utils.hashing.expression_hash 完全等价。

    注意：不要 import utils，保持 database.py 只依赖标准库。
    """
    normalized = expression.replace(" ", "").lower()
    return hashlib.md5(
        normalized.encode("utf-8")
    ).hexdigest()


def _extract_checks_map(raw_json: Any) -> Dict[str, str]:
    """
    从 raw_json 中提取 {check_name: result}。

    支持：
        - {"is_raw": {"checks": [...]}}
        - {"is":     {"checks": [...]}}
        - {"checks": [...]}
        - JSON 字符串（上述任一形态）

    不抛异常；无数据返回 {}。
    注意：不 import evaluation.official_checks，保持 database.py
    只依赖标准库。
    """
    if raw_json is None:
        return {}
    if isinstance(raw_json, str):
        if not raw_json.strip():
            return {}
        try:
            raw_json = json.loads(raw_json)
        except (TypeError, ValueError):
            return {}
    if not isinstance(raw_json, dict):
        return {}

    checks_list = None
    for key in ("is_raw", "is"):
        block = raw_json.get(key)
        if isinstance(block, dict):
            c = block.get("checks")
            if isinstance(c, list):
                checks_list = c
                break
    if checks_list is None:
        c = raw_json.get("checks")
        if isinstance(c, list):
            checks_list = c

    if not checks_list:
        return {}

    out: Dict[str, str] = {}
    for c in checks_list:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "") or "")
        if not name:
            continue
        out[name] = str(c.get("result", "UNKNOWN") or "UNKNOWN")
    return out


# ============================================================
# WQDatabase
# ============================================================

class WQDatabase:
    """SQLite 持久化封装。"""

    def __init__(self, db_path: str = "data/wq.db") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self.init_schema()

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    def init_schema(self) -> None:
        """创建所有表与索引（幂等）。"""
        try:
            with self._lock:
                with self._conn:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA foreign_keys=ON")
                    for stmt in _SCHEMA_STATEMENTS:
                        self._conn.execute(stmt)

                    # 迁移：加 submitted / self_corr 字段（幂等）
                    # SQLite 不支持 ADD COLUMN IF NOT EXISTS，用 try/except
                    for col_name, col_def in [
                        ("submitted", "BOOLEAN DEFAULT 0"),
                        ("submitted_at", "TIMESTAMP"),
                        ("self_corr", "TEXT DEFAULT NULL"),
                        ("self_corr_checked_at", "TIMESTAMP DEFAULT NULL"),
                        ("seed_tier", "TEXT DEFAULT NULL"),
                        ("seed_weight", "REAL DEFAULT NULL"),
                        ("parent_structure", "TEXT DEFAULT NULL"),
                    ]:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE evaluations "
                                f"ADD COLUMN {col_name} {col_def}"
                            )
                        except sqlite3.OperationalError:
                            pass
                    self._conn.commit()
        except Exception:
            logger.exception("init_schema 失败")

    # --------------------------------------------------------
    # devices
    # --------------------------------------------------------

    def get_or_create_device(self, device_name: str) -> str:
        """按名称获取或创建设备，返回 device_id。"""
        try:
            with self._lock:
                with self._conn:
                    row = self._conn.execute(
                        "SELECT device_id FROM devices WHERE device_name = ?",
                        (device_name,),
                    ).fetchone()
                    if row is not None:
                        device_id = row["device_id"]
                        self._conn.execute(
                            "UPDATE devices SET last_seen = ? WHERE device_id = ?",
                            (_now(), device_id),
                        )
                        return device_id
                    device_id = str(uuid.uuid4())
                    self._conn.execute(
                        "INSERT INTO devices (device_id, device_name, first_seen, last_seen) "
                        "VALUES (?, ?, ?, ?)",
                        (device_id, device_name, _now(), _now()),
                    )
                    return device_id
        except Exception:
            logger.exception("get_or_create_device 失败: %s", device_name)
            return ""

    # --------------------------------------------------------
    # experiments
    # --------------------------------------------------------

    def create_experiment(
        self,
        experiment_id: str,
        device_id: str,
        random_seed: Optional[int],
        config_snapshot: Any,
    ) -> None:
        """创建 experiment 记录。"""
        try:
            now = _now()
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO experiments ("
                        "experiment_id, device_id, created_at, updated_at, status, "
                        "random_seed, config_snapshot, generation_last, "
                        "concurrency_current, best_alpha_id"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            experiment_id,
                            device_id,
                            now,
                            now,
                            "RUNNING",
                            random_seed,
                            _dumps(config_snapshot),
                            0,
                            0,
                            None,
                        ),
                    )
        except Exception:
            logger.exception("create_experiment 失败: %s", experiment_id)

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """读取单个 experiment。"""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_experiment(row)
        except Exception:
            logger.exception("get_experiment 失败: %s", experiment_id)
            return None

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["config_snapshot"] = _loads(d.get("config_snapshot"), {})
        return d

    # --------------------------------------------------------
    # expressions / evaluations
    # --------------------------------------------------------

    def is_evaluated(self, expr_hash: str) -> bool:
        """判断 expr_hash 是否已评估。"""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT 1 FROM evaluations WHERE expr_hash = ? LIMIT 1",
                    (expr_hash,),
                ).fetchone()
            return row is not None
        except Exception:
            logger.exception("is_evaluated 失败: %s", expr_hash)
            return False

    def load_all_expr_hashes(self) -> Set[str]:
        """加载所有已评估的 expr_hash。"""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT DISTINCT expr_hash FROM evaluations"
                ).fetchall()
            return {r["expr_hash"] for r in rows if r["expr_hash"]}
        except Exception:
            logger.exception("load_all_expr_hashes 失败")
            return set()

    def save_record(
        self,
        record: Dict[str, Any],
        experiment_id: str,
        device_id: str,
        generation: int,
        parent_structure: Optional[str] = None,
    ) -> bool:
        """保存一条评估记录（expressions + evaluations，单事务）。

        Parameters
        ----------
        parent_structure : str | None
            父种子结构签名（变体实验用）。缺省 None，不影响现有调用。

        成功返回 True，失败返回 False。
        """
        try:
            expression_text = str(record.get("expression") or "")
            expr_hash = _expr_hash(expression_text)
            alpha_id = str(record.get("alpha_id") or uuid.uuid4())
            family = record.get("family")
            tags = record.get("tags") or []
            official_checks = record.get("official_checks") or []
            raw_json = record.get("raw_json")
            if raw_json is None:
                raw_json = record.get("extra") or {}

            with self._lock:
                with self._conn:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO expressions ("
                        "expr_hash, expression, first_seen_at, "
                        "first_device_id, first_exp_id, family"
                        ") VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            expr_hash,
                            expression_text,
                            _now(),
                            device_id,
                            experiment_id,
                            family,
                        ),
                    )
                    self._conn.execute(
                        "INSERT OR IGNORE INTO evaluations ("
                        "alpha_id, expr_hash, experiment_id, device_id, generation, "
                        "sharpe, fitness, turnover, score, classification, "
                        "complexity, depth, tags, official_checks, raw_json, "
                        "evaluated_at, parent_structure"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            alpha_id,
                            expr_hash,
                            experiment_id,
                            device_id,
                            generation,
                            record.get("sharpe"),
                            record.get("fitness"),
                            record.get("turnover"),
                            record.get("score"),
                            record.get("classification"),
                            record.get("complexity"),
                            record.get("depth"),
                            _dumps(tags),
                            _dumps(official_checks),
                            _dumps(raw_json),
                            _now(),
                            parent_structure,
                        ),
                    )
            return True
        except Exception:
            logger.exception("save_record 失败: %s", record.get("alpha_id"))
            return False

    def top_alphas(
        self,
        n: int = 20,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按 score 降序返回前 n 条记录。"""
        try:
            with self._lock:
                if classification is None:
                    rows = self._conn.execute(
                        "SELECT * FROM evaluations ORDER BY score DESC LIMIT ?",
                        (n,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM evaluations WHERE classification = ? "
                        "ORDER BY score DESC LIMIT ?",
                        (classification, n),
                    ).fetchall()
            return [self._row_to_evaluation(r) for r in rows]
        except Exception:
            logger.exception("top_alphas 失败")
            return []

    def get_experiment_evaluations(
        self,
        experiment_id: str,
    ) -> List[Dict[str, Any]]:
        """返回某个 experiment 的所有 evaluations 记录。

        每条形如：
        {
            "alpha_id": "...",
            "expr_hash": "...",
            "expression": "...",   # 从 expressions 表 join 得到
            "generation": 0,
            "sharpe": 1.79,
            "fitness": 0.92,
            "turnover": 0.67,
            "score": 0.98,
            "classification": "CANDIDATE",
            "family": "MOMENTUM",
            "complexity": 4,
            "depth": 3,
            "tags": [...],              # 已 json.loads
            "official_checks": [...],   # 已 json.loads
            "raw_json": "...",
            "evaluated_at": "...",
        }

        按 evaluated_at DESC 排序。
        """
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT "
                    "e.alpha_id, e.expr_hash, x.expression, "
                    "e.experiment_id, e.generation, "
                    "e.sharpe, e.fitness, e.turnover, e.score, "
                    "e.classification, e.complexity, e.depth, "
                    "e.tags, e.official_checks, e.raw_json, e.evaluated_at "
                    "FROM evaluations e "
                    "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
                    "WHERE e.experiment_id = ? "
                    "ORDER BY e.evaluated_at DESC",
                    (experiment_id,),
                ).fetchall()
            results: List[Dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                d["tags"] = _loads(d.get("tags"), [])
                d["official_checks"] = _loads(
                    d.get("official_checks"), []
                )
                d["raw_json"] = _loads(d.get("raw_json"), {})
                results.append(d)
            return results
        except Exception:
            logger.exception(
                "get_experiment_evaluations 失败: %s", experiment_id
            )
            return []

    @staticmethod
    def _row_to_evaluation(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["tags"] = _loads(d.get("tags"), [])
        d["official_checks"] = _loads(d.get("official_checks"), [])
        d["raw_json"] = _loads(d.get("raw_json"), {})
        return d

    # --------------------------------------------------------
    # 提交跟踪
    # --------------------------------------------------------

    # --------------------------------------------------------
    # self-corr 检查
    # --------------------------------------------------------

    # sim 阶段 7 项（不含 SELF_CORRELATION）
    _SIM_STAGE_CHECKS = (
        "LOW_SHARPE",
        "LOW_FITNESS",
        "LOW_TURNOVER",
        "HIGH_TURNOVER",
        "CONCENTRATED_WEIGHT",
        "LOW_SUB_UNIVERSE_SHARPE",
        "MATCHES_COMPETITION",
    )

    def get_pending_self_corr(
        self, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        返回 sim 阶段 7 项全 PASS 但 self_corr 未查的候选。

        判断依据（Python 侧解析 raw_json）：
            - raw_json.is_raw.checks 中 SELF_CORRELATION == "PENDING"
            - 其余 7 项（_SIM_STAGE_CHECKS）都 == "PASS"
            - evaluations.self_corr IS NULL

        返回每条形如：
            {"alpha_id": "...", "expr_hash": "...", "expression": "..."}
        """
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT "
                    "e.alpha_id, e.expr_hash, x.expression, e.raw_json "
                    "FROM evaluations e "
                    "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
                    "WHERE e.self_corr IS NULL "
                    "ORDER BY e.score DESC"
                ).fetchall()
        except Exception:
            logger.exception("get_pending_self_corr 查询失败")
            return []

        out: List[Dict[str, Any]] = []
        for row in rows:
            raw = _loads(row["raw_json"], {})
            checks = _extract_checks_map(raw)
            if not checks:
                continue
            if checks.get("SELF_CORRELATION") != "PENDING":
                continue
            if not all(
                checks.get(name) == "PASS"
                for name in self._SIM_STAGE_CHECKS
            ):
                continue
            out.append({
                "alpha_id": str(row["alpha_id"] or ""),
                "expr_hash": str(row["expr_hash"] or ""),
                "expression": str(row["expression"] or ""),
            })
            if len(out) >= limit:
                break
        return out

    def update_self_corr(
        self, alpha_id: str, result: str,
    ) -> bool:
        """
        更新 self_corr 列 + self_corr_checked_at。

        result 取值：PASS / FAIL / PENDING / ERROR
        成功返回 True，失败返回 False。
        """
        try:
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        "UPDATE evaluations "
                        "SET self_corr = ?, self_corr_checked_at = ? "
                        "WHERE alpha_id = ?",
                        (result, _now(), alpha_id),
                    )
            return True
        except Exception:
            logger.exception("update_self_corr 失败: %s", alpha_id)
            return False

    # --------------------------------------------------------
    # 种子分层（Phase 1）
    # --------------------------------------------------------

    # tier → weight 映射
    _SEED_WEIGHTS = {
        "gold": 1.0,
        "silver": 0.6,
        "iron": 0.2,
    }

    def update_seed_tiers(self) -> Dict[str, int]:
        """
        遍历所有 7 项 PASS 的候选，按 self_corr 结果设置
        seed_tier / seed_weight / parent_structure。

        规则（按优先级）：
            submitted == 1        → gold   (1.0)
            submitted == -1       → iron   (0.2)
            self_corr == 'PASS'   → gold   (1.0)
            self_corr == 'FAIL'   → iron   (0.2)
            self_corr IS NULL     → silver (0.6)
            self_corr == 'ERROR'  → silver (0.6, 待重查)
            其它（PENDING 等）    → 不设置（保持 NULL）

        额外：self_corr == 'ERROR' 时，把 self_corr 重置为 NULL，
              让 daemon 下次重查。

        parent_structure 用 core.signature.structure_signature 计算；
        表达式解析失败则留 NULL。

        返回 {"gold": n, "silver": n, "iron": n, "skipped": n}
        """
        # 延迟 import，避免 database.py 顶层依赖项目内部模块
        try:
            from alpha.parser import parse_expression
            from core.signature import structure_signature
        except Exception:
            logger.exception("update_seed_tiers: import 失败")
            return {"gold": 0, "silver": 0, "iron": 0, "skipped": 0}

        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT "
                    "e.alpha_id, e.self_corr, e.submitted, "
                    "x.expression, e.raw_json "
                    "FROM evaluations e "
                    "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash"
                ).fetchall()
        except Exception:
            logger.exception("update_seed_tiers: 查询失败")
            return {"gold": 0, "silver": 0, "iron": 0, "skipped": 0}

        counts = {"gold": 0, "silver": 0, "iron": 0, "skipped": 0}

        for row in rows:
            raw = _loads(row["raw_json"], {})
            checks = _extract_checks_map(raw)
            if not checks:
                counts["skipped"] += 1
                continue
            if not all(
                checks.get(name) == "PASS"
                for name in self._SIM_STAGE_CHECKS
            ):
                counts["skipped"] += 1
                continue

            sc = row["self_corr"]
            submitted = row["submitted"]

            # 按优先级判断 tier
            if submitted == 1:
                tier = "gold"
            elif submitted == -1:
                tier = "iron"
            elif sc == "PASS":
                tier = "gold"
            elif sc == "FAIL":
                tier = "iron"
            elif sc is None:
                tier = "silver"
            elif sc == "ERROR":
                # 待重查：标 silver，并把 self_corr 重置为 NULL
                tier = "silver"
                try:
                    with self._lock:
                        with self._conn:
                            self._conn.execute(
                                "UPDATE evaluations "
                                "SET self_corr = NULL, "
                                "self_corr_checked_at = NULL "
                                "WHERE alpha_id = ?",
                                (row["alpha_id"],),
                            )
                except Exception:
                    logger.exception(
                        "update_seed_tiers: 重置 ERROR 失败 %s",
                        row["alpha_id"],
                    )
            else:
                counts["skipped"] += 1
                continue

            weight = self._SEED_WEIGHTS.get(tier)

            # 计算 parent_structure
            expr = str(row["expression"] or "")
            parent_sig = None
            if expr:
                try:
                    node = parse_expression(expr)
                    if node is not None:
                        parent_sig = structure_signature(node)
                except Exception:
                    parent_sig = None

            try:
                with self._lock:
                    with self._conn:
                        self._conn.execute(
                            "UPDATE evaluations "
                            "SET seed_tier = ?, seed_weight = ?, "
                            "parent_structure = ? "
                            "WHERE alpha_id = ?",
                            (tier, weight, parent_sig, row["alpha_id"]),
                        )
                counts[tier] += 1
            except Exception:
                logger.exception(
                    "update_seed_tiers: 更新失败 %s", row["alpha_id"],
                )
                counts["skipped"] += 1

        logger.info("update_seed_tiers 完成: %s", counts)
        return counts

    def get_gold_seeds(self) -> List[Dict[str, Any]]:
        """
        返回 seed_tier='gold' 的所有种子。

        每项含：alpha_id, expression, sharpe, fitness, turnover
        按 sharpe DESC 排序。
        """
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT "
                    "e.alpha_id, x.expression, "
                    "e.sharpe, e.fitness, e.turnover "
                    "FROM evaluations e "
                    "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
                    "WHERE e.seed_tier = 'gold' "
                    "ORDER BY e.sharpe DESC"
                ).fetchall()
        except Exception:
            logger.exception("get_gold_seeds 失败")
            return []

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "alpha_id": str(r["alpha_id"] or ""),
                "expression": str(r["expression"] or ""),
                "sharpe": r["sharpe"],
                "fitness": r["fitness"],
                "turnover": r["turnover"],
            })
        return out

    def get_seed_stats(self) -> Dict[str, Any]:
        """
        返回种子分层统计（供 baseline_stats 使用）。

        返回：
        {
            "gold": n, "silver": n, "iron": n,
            "iron_unique_structures": n,
            "total_evaluations": n,
            "seven_pass": n,
            "self_corr_pass": n,
        }
        """
        out: Dict[str, Any] = {
            "gold": 0, "silver": 0, "iron": 0,
            "iron_unique_structures": 0,
            "total_evaluations": 0,
            "seven_pass": 0,
            "self_corr_pass": 0,
        }
        try:
            with self._lock:
                # 总数
                row = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM evaluations"
                ).fetchone()
                out["total_evaluations"] = int(row["c"] or 0)

                # tier 计数
                for tier in ("gold", "silver", "iron"):
                    row = self._conn.execute(
                        "SELECT COUNT(*) AS c FROM evaluations "
                        "WHERE seed_tier = ?",
                        (tier,),
                    ).fetchone()
                    out[tier] = int(row["c"] or 0)

                # 铁种子去重结构数
                row = self._conn.execute(
                    "SELECT COUNT(DISTINCT parent_structure) AS c "
                    "FROM evaluations "
                    "WHERE seed_tier = 'iron' "
                    "AND parent_structure IS NOT NULL"
                ).fetchone()
                out["iron_unique_structures"] = int(row["c"] or 0)

                # self_corr PASS 数
                # 包含两类：
                #   1. self_corr 明确为 'PASS'
                #   2. 已提交成功（submitted=1）—— BRAIN 已判定过 self-corr
                row = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM evaluations "
                    "WHERE (self_corr = 'PASS' OR submitted = 1)"
                ).fetchone()
                out["self_corr_pass"] = int(row["c"] or 0)

                # 7 项 PASS 数（Python 侧解析）
                rows = self._conn.execute(
                    "SELECT raw_json FROM evaluations"
                ).fetchall()
        except Exception:
            logger.exception("get_seed_stats 失败")
            return out

        seven_pass = 0
        for r in rows:
            raw = _loads(r["raw_json"], {})
            checks = _extract_checks_map(raw)
            if not checks:
                continue
            if all(
                checks.get(name) == "PASS"
                for name in self._SIM_STAGE_CHECKS
            ):
                seven_pass += 1
        out["seven_pass"] = seven_pass
        return out

    # --------------------------------------------------------
    # population_snapshots
    # --------------------------------------------------------

    def save_snapshot(
        self,
        experiment_id: str,
        generation: int,
        population: Any,
        elites: Any,
    ) -> None:
        """保存某一代的种群快照。"""
        try:
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO population_snapshots ("
                        "experiment_id, generation, population, elites, created_at"
                        ") VALUES (?, ?, ?, ?, ?)",
                        (
                            experiment_id,
                            generation,
                            _dumps(population),
                            _dumps(elites),
                            _now(),
                        ),
                    )
        except Exception:
            logger.exception("save_snapshot 失败: %s/%s", experiment_id, generation)

    def load_latest_snapshot(
        self,
        experiment_id: str,
    ) -> Optional[Dict[str, Any]]:
        """加载最新一代的种群快照。"""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM population_snapshots WHERE experiment_id = ? "
                    "ORDER BY generation DESC LIMIT 1",
                    (experiment_id,),
                ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["population"] = _loads(d.get("population"), [])
            d["elites"] = _loads(d.get("elites"), [])
            return d
        except Exception:
            logger.exception("load_latest_snapshot 失败: %s", experiment_id)
            return None

    # --------------------------------------------------------
    # pending_tasks
    # --------------------------------------------------------

    def save_task(
        self,
        task_id: str,
        experiment_id: str,
        generation: int,
        expression: str,
        expr_hash: str,
        source: str,
        state: str = "PENDING",
    ) -> bool:
        """保存一条待处理任务。成功返回 True，失败返回 False。"""
        try:
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO pending_tasks ("
                        "task_id, experiment_id, generation, expression, "
                        "expr_hash, source, state, created_at, completed_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            task_id,
                            experiment_id,
                            generation,
                            expression,
                            expr_hash,
                            source,
                            state,
                            _now(),
                            None,
                        ),
                    )
            return True
        except Exception:
            logger.exception("save_task 失败: %s", task_id)
            return False

    def update_task_state(self, task_id: str, state: str) -> bool:
        """更新任务状态；终态时写入 completed_at。

        成功返回 True，失败返回 False。
        """
        try:
            with self._lock:
                with self._conn:
                    if state in _TERMINAL_STATES:
                        self._conn.execute(
                            "UPDATE pending_tasks SET state = ?, completed_at = ? "
                            "WHERE task_id = ?",
                            (state, _now(), task_id),
                        )
                    else:
                        self._conn.execute(
                            "UPDATE pending_tasks SET state = ? WHERE task_id = ?",
                            (state, task_id),
                        )
            return True
        except Exception:
            logger.exception("update_task_state 失败: %s", task_id)
            return False

    # --------------------------------------------------------
    # close
    # --------------------------------------------------------

    def close(self) -> None:
        """关闭数据库连接。"""
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            logger.exception("close 失败")


# ============================================================
# self-test
# ============================================================

if __name__ == "__main__":
    db = WQDatabase(":memory:")
    dev = db.get_or_create_device("test-device")
    db.create_experiment("test-exp", dev, 42, {})
    db.save_record({
        "expression": "rank(ts_delta(close,3))",
        "alpha_id": "AAA", "sharpe": 1.79, "fitness": 0.92,
        "turnover": 0.67, "score": 0.98,
        "classification": "CANDIDATE", "family": "MOMENTUM",
        "complexity": 4, "depth": 3,
        "official_checks": [], "extra": {}, "tags": ["AI_REFINED"],
    }, "test-exp", dev, 0)
    print("is_evaluated:", db.is_evaluated("any-hash"))
    print("experiments:", len(db.list_experiments()))
    print("top_alphas:", len(db.top_alphas(10)))
    db.close()
    print("self-test OK")
