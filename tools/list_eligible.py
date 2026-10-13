# -*- coding: utf-8 -*-

"""
列出符合门槛且未提交的 alpha（辅助，用于人工挑选）。

筛选条件（硬编码，按需求）：
    sharpe   >= 1.25
    fitness  >= 1.0
    turnover >= 0.01
    turnover <= 0.70
    submitted IS NULL OR submitted = 0
    self_corr IS NULL OR self_corr = 'PASS'   （默认；--show-all 关闭）

用法
----
    python tools/list_eligible.py
    python tools/list_eligible.py --db data/wq.db
    python tools/list_eligible.py --json
    python tools/list_eligible.py --limit 50
    python tools/list_eligible.py --show-all

注意
----
- 只读，不写数据库。
- 直接查询 evaluations JOIN expressions，不复用
  WQDatabase.get_unsubmitted_eligible（其 turnover 只有上界，
  本脚本同时要求下界 >= 0.01）。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List


# 硬编码门槛（按需求）
MIN_SHARPE = 1.25
MIN_FITNESS = 1.0
MIN_TURNOVER = 0.01
MAX_TURNOVER = 0.70

DEFAULT_DB_PATH = "data/wq.db"
ENV_DB_PATH = "WQ_DB_PATH"


def _resolve_db_path(cli_db: str | None) -> str:
    """优先级：--db > WQ_DB_PATH > 默认。"""
    if cli_db:
        return cli_db
    env = os.environ.get(ENV_DB_PATH)
    if env:
        return env
    return DEFAULT_DB_PATH


def _query(
    db_path: str,
    limit: int | None,
    show_all: bool = False,
) -> List[Dict[str, Any]]:
    """查询符合门槛且未提交的 alpha，按 score DESC 排序。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT "
            "e.alpha_id, x.expression, "
            "e.sharpe, e.fitness, e.turnover, e.score, "
            "e.classification, e.experiment_id, e.evaluated_at, "
            "e.self_corr "
            "FROM evaluations e "
            "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
            "WHERE e.sharpe >= ? AND e.fitness >= ? "
            "AND e.turnover >= ? AND e.turnover <= ? "
            "AND (e.submitted IS NULL OR e.submitted = 0) "
        )
        params: List[Any] = [
            MIN_SHARPE, MIN_FITNESS, MIN_TURNOVER, MAX_TURNOVER,
        ]
        if not show_all:
            sql += (
                "AND (e.self_corr IS NULL OR e.self_corr = 'PASS') "
            )
        sql += "ORDER BY e.score DESC"
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(no eligible alphas)")
        return
    header = (
        f"{'alpha_id':<12} {'experiment':<20} {'sharpe':>7} "
        f"{'fitness':>8} {'turnover':>9} {'self_corr':>10}  expression"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        expr = (r.get("expression") or "")[:80]
        exp_id = str(r.get("experiment_id") or "")[:20]
        sc = r.get("self_corr")
        sc_str = str(sc) if sc is not None else "-"
        print(
            f"{str(r.get('alpha_id') or ''):<12} "
            f"{exp_id:<20} "
            f"{float(r.get('sharpe') or 0.0):>7.4f} "
            f"{float(r.get('fitness') or 0.0):>8.4f} "
            f"{float(r.get('turnover') or 0.0):>9.4f} "
            f"{sc_str:>10}  "
            f"{expr}"
        )
    print(f"\nTotal: {len(rows)}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List alphas meeting sharpe/fitness/turnover thresholds.",
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
        "--json",
        action="store_true",
        help="Output JSON instead of a table",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to return",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help=(
            "Show all rows regardless of self_corr "
            "(default: only NULL or PASS)"
        ),
    )
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db)
    if not os.path.isfile(db_path):
        print(f"database not found: {db_path}", file=sys.stderr)
        return 1

    try:
        rows = _query(db_path, args.limit, show_all=args.show_all)
    except sqlite3.Error as e:
        print(f"query failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
