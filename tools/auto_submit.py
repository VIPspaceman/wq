# -*- coding: utf-8 -*-

"""
自动提交框架（默认关闭、默认演练模式，不启用）。

目标
----
从 DB 筛选"通过 BRAIN 完整检查"的候选 alpha，导出成清单供人工提交；
未来可选接入 BRAIN API 自动提交。

当前阶段
--------
只做框架：
    - 不做定时器
    - 不做后台任务
    - 不自动启用（auto_submit.enabled 默认 False）

用法
----
    # 演练模式（默认）：只导出 CSV，不提交
    python tools/auto_submit.py

    # 限定某个实验
    python tools/auto_submit.py --experiment-id exp_xxx

    # 最多导出 5 条
    python tools/auto_submit.py --limit 5

    # 真正提交（需同时满足 dry_run=False 且加 --execute）
    python tools/auto_submit.py --execute

流程
----
    1. 从 DB 查询所有 evaluations：
       - 满足 sharpe/fitness/turnover 门槛
       - raw_json 里的 BRAIN checks 全部 PASS 或 PENDING
       - submitted IS NULL OR submitted = 0（未提交）
    2. 按 score 降序排序
    3. 结构去重（structure_signature）：每个结构只保留 top-1
    4. 截断到 max_daily_submissions（默认 1）
    5. dry_run 或未加 --execute → 导出到 CSV
    6. 加 --execute 且 dry_run=False → 调 brain.client.submit_alpha

约束
----
- 不修改任何现有搜索/评估逻辑
- 不做定时器、不做后台任务
- auto_submit.enabled 为 False 时，工具只能手动跑

TODO
----
- 每日配额目前是"单次运行上限"，未做跨天累计。
  未来若需真正的"每日配额"，需在 DB 里记录当日提交数。
- --execute 路径依赖 brain.client.submit_alpha 的端点正确性
  （社区参考，非官方文档），使用前请手动验证。
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让脚本在 tools/ 目录下也能 import 项目根下的包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import AutoSubmitSettings
from evaluation.official_checks import all_checks_pass


logger = logging.getLogger(__name__)


DEFAULT_DB_PATH = "data/wq.db"
ENV_DB_PATH = "WQ_DB_PATH"


# ============================================================
# 正则工具（不依赖 parse_expression）
# ============================================================
#
# 背景
# ----
# parse_expression 用本地 hardcoded catalog（只有 pv1 6 个字段），
# DB 里的表达式含 debt/assets/ebit 等基本面字段时 parse 返回 None，
# 导致 structure_signature 为空 → 结构去重失效。
#
# 因此这里改用"正则结构签名"：把所有数字替换为 N。
# 这是近似值（不区分算子语义），但对"同结构不同窗口"足够。

# 价量字段（含任一即视为"含价量字段"）
_PV_FIELDS = frozenset({
    "close", "open", "high", "low", "volume", "returns", "vwap",
})

# 基本面字段（与 alpha/families.py 的 _FUNDAMENTAL_FIELD_POOL 一致）
_FUND_FIELDS = frozenset({
    "debt", "assets", "equity", "ebit", "cash", "employee",
    "enterprise_value", "cogs", "capex", "eps", "current_ratio",
    "bookvalue_ps", "cashflow", "cashflow_op", "ebitda",
})


def _quick_sig(expr: str) -> str:
    """
    用正则计算结构签名近似值：把所有数字替换为 N。

    例：
        "rank(ts_mean(debt,120))"  → "rank(ts_mean(debt,N))"
        "rank(ts_mean(debt,20))"   → "rank(ts_mean(debt,N))"  （同结构）
    """
    return re.sub(r"\d+", "N", expr or "")


def _is_saturated_regex(expr: str) -> bool:
    """
    正则版 saturation：含基本面字段 + 无价量字段 → 饱和。

    用词边界匹配（(?<![a-z_]) / (?![a-z_])），
    避免 'cash' 匹配 'cashflow'。

    TODO: 与 alpha/saturation.py 的 is_brain_saturated 语义保持一致。
          若后者规则变化，此处需同步。
    """
    e = (expr or "").lower()
    has_fund = any(
        re.search(rf"(?<![a-z_]){f}(?![a-z_])", e)
        for f in _FUND_FIELDS
    )
    has_pv = any(
        re.search(rf"(?<![a-z_]){f}(?![a-z_])", e)
        for f in _PV_FIELDS
    )
    return has_fund and not has_pv


# ============================================================
# DB 路径
# ============================================================

def _resolve_db_path(cli_db: Optional[str]) -> str:
    """优先级：--db > WQ_DB_PATH > 默认。"""
    if cli_db:
        return cli_db
    env = os.environ.get(ENV_DB_PATH)
    if env:
        return env
    return DEFAULT_DB_PATH


# ============================================================
# 查询
# ============================================================

def _query_candidates(
    db_path: str,
    settings: AutoSubmitSettings,
    experiment_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    查询满足门槛的候选 alpha（含 raw_json，用于 checks 过滤）。

    返回按 score DESC 排序的 dict 列表。
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT "
            "e.alpha_id, x.expression, "
            "e.sharpe, e.fitness, e.turnover, e.score, "
            "e.classification, e.experiment_id, e.evaluated_at, "
            "e.raw_json, e.submitted "
            "FROM evaluations e "
            "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
            "WHERE e.sharpe >= ? AND e.fitness >= ? "
            "AND e.turnover >= ? AND e.turnover <= ? "
        )
        params: List[Any] = [
            settings.min_sharpe,
            settings.min_fitness,
            settings.min_turnover,
            settings.max_turnover,
        ]

        if settings.exclude_submitted:
            # 显式排除已提交（submitted=1）和已提交但被拒（submitted=-1）。
            # 注意：不要写成 `submitted != 1`，那样会把 -1 也包含进来。
            #
            # TODO: 若某个被拒的 alpha 在 DB 里还没被标记为 -1
            #       （例如从未手动标记过），本工具无法知道它被拒过。
            #       未来可通过 BRAIN API 拉取"已提交列表"来同步。
            sql += "AND (e.submitted IS NULL OR e.submitted = 0) "

        if experiment_id:
            sql += "AND e.experiment_id = ? "
            params.append(experiment_id)

        sql += "ORDER BY e.score DESC"

        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ============================================================
# 过滤
# ============================================================

def _filter_by_checks(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    只保留 raw_json 里 BRAIN checks 全部 PASS 或 PENDING 的记录。

    无 checks 数据的记录被排除（保守）。
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        raw = r.get("raw_json")
        if all_checks_pass(raw, ignore_pending=True):
            out.append(r)
    return out


def _dedup_by_structure(
    rows: List[Dict[str, Any]],
    max_per_structure: int = 1,
) -> List[Dict[str, Any]]:
    """
    结构去重：每个结构签名只保留 top-N（默认 top-1）。

    输入需已按 score DESC 排序。

    使用 _quick_sig（正则，数字→N），不依赖 parse_expression，
    因此对含基本面字段的表达式也能正确去重。
    """
    seen: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for r in rows:
        expr = str(r.get("expression") or "")
        sig = _quick_sig(expr)

        if not sig:
            # 空表达式 → 视为独立结构，直接保留
            out.append(r)
            continue

        count = seen.get(sig, 0)
        if count >= max_per_structure:
            continue
        seen[sig] = count + 1
        out.append(r)
    return out


# ============================================================
# 导出
# ============================================================

def _export_csv(
    rows: List[Dict[str, Any]],
    experiment_id: Optional[str],
) -> str:
    """
    导出候选到 CSV。

    路径：experiments/<exp_id>/auto_submit_candidates.csv
    exp_id 缺省时用 "all"。
    """
    exp_dir = os.path.join(
        "experiments", experiment_id or "all",
    )
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "auto_submit_candidates.csv")

    fieldnames = [
        "alpha_id", "sharpe", "fitness", "turnover",
        "score", "expression",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "alpha_id": r.get("alpha_id") or "",
                "sharpe": r.get("sharpe"),
                "fitness": r.get("fitness"),
                "turnover": r.get("turnover"),
                "score": r.get("score"),
                "expression": r.get("expression") or "",
            })
    return path


# ============================================================
# 提交（可选）
# ============================================================

def _try_submit(alpha_id: str) -> bool:
    """
    尝试提交单个 alpha 到 BRAIN。

    TODO: 依赖 brain.client.submit_alpha 的端点正确性
          （社区参考，非官方文档）。使用前请手动验证。
    失败返回 False（不抛异常）。
    """
    try:
        from brain.client import BrainClient
        from config.settings import (
            AppSettings,
            BrainEndpoints,
            PollingSettings,
        )
        from brain.authentication import create_authenticated_session
    except ImportError:
        logger.warning("无法导入 brain 模块，跳过提交")
        return False

    try:
        settings = AppSettings.from_env()
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
            max_retries=settings.polling.http_retry,
        )
        client = BrainClient(
            session, settings.endpoints, settings.polling,
        )
        return client.submit_alpha(alpha_id)
    except Exception:
        logger.exception("提交失败: %s", alpha_id)
        return False


# ============================================================
# 主入口
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-submit framework (dry-run by default). "
            "Filters eligible alphas and exports a CSV."
        ),
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "SQLite database path "
            f"(default: ${ENV_DB_PATH} or {DEFAULT_DB_PATH})"
        ),
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Limit to a specific experiment_id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of candidates to export/submit",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually submit (requires dry_run=False in settings). "
            "Default: export only."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    settings = AutoSubmitSettings()

    db_path = _resolve_db_path(args.db)
    if not os.path.isfile(db_path):
        print(f"database not found: {db_path}", file=sys.stderr)
        return 1

    # -------- 1. 查询 --------
    try:
        rows = _query_candidates(
            db_path, settings, experiment_id=args.experiment_id,
        )
    except sqlite3.Error as e:
        print(f"query failed: {e}", file=sys.stderr)
        return 1

    print(f"找到候选 {len(rows)} 条")

    # -------- 2. checks 过滤 --------
    rows = _filter_by_checks(rows)
    print(f"通过 BRAIN checks 过滤后 {len(rows)} 条")

    # -------- 2b. 排除 BRAIN 已知死角（纯基本面） --------
    before = len(rows)
    rows = [
        r for r in rows
        if not _is_saturated_regex(str(r.get("expression") or ""))
    ]
    print(f"排除 BRAIN 死角后 {len(rows)} 条（过滤 {before - len(rows)} 条）")

    # -------- 3. 结构去重 --------
    if settings.structure_dedup:
        rows = _dedup_by_structure(
            rows, max_per_structure=settings.max_per_structure,
        )
        print(f"结构去重后 {len(rows)} 条")

    # -------- 4. 截断 --------
    limit = (
        args.limit
        if args.limit is not None and args.limit > 0
        else settings.max_daily_submissions
    )
    rows = rows[:limit]
    print(f"截断到 {len(rows)} 条")

    if not rows:
        print("(no candidates)")
        return 0

    # -------- 5. 打印 --------
    for r in rows:
        expr = str(r.get("expression") or "")[:80]
        print(
            f"  {str(r.get('alpha_id') or ''):<12} "
            f"sharpe={float(r.get('sharpe') or 0.0):+.4f} "
            f"fitness={float(r.get('fitness') or 0.0):+.4f} "
            f"turnover={float(r.get('turnover') or 0.0):.4f} | "
            f"{expr}"
        )

    # -------- 6. 提交 or 导出 --------
    do_execute = args.execute and not settings.dry_run

    if do_execute:
        print("执行模式：真正提交")
        for r in rows:
            alpha_id = str(r.get("alpha_id") or "")
            if not alpha_id:
                continue
            ok = _try_submit(alpha_id)
            print(f"  submit {alpha_id}: {'OK' if ok else 'FAILED'}")
    else:
        if args.execute and settings.dry_run:
            print("演练模式（dry_run=True），未提交")
        else:
            print("演练模式，未提交")

        path = _export_csv(rows, args.experiment_id)
        print(f"已导出: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
