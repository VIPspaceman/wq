# -*- coding: utf-8 -*-

"""
手动标记某个 alpha 为"已提交"（主路，最可靠）。

用法
----
    python tools/mark_submitted.py <alpha_id>
    python tools/mark_submitted.py <alpha_id> --note "已提交到 BRAIN"
    python tools/mark_submitted.py <alpha_id> --db data/wq.db

行为
----
- 直接 UPDATE evaluations SET submitted=1, submitted_at=NOW()
  WHERE alpha_id=?（不改 schema，submitted/submitted_at 列已存在）。
- 打印受影响行数 + 该 alpha 的表达式（前 120 字）。
- 未找到 alpha_id → 打印 "未找到 alpha_id: X"，退出码 1。
- 已标记过 → 打印 "已经提交过"，但仍 UPDATE（幂等）。
- 只读/写本地 SQLite，不调用 BRAIN API。

数据库路径
----------
优先级：--db 参数 > 环境变量 WQ_DB_PATH > 默认 "data/wq.db"。

TODO
----
- --note 目前仅打印，不落库（约束：不改 evaluations schema）。
  若未来允许改 schema，可新增 note 列。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from typing import List, Optional


DEFAULT_DB_PATH = "data/wq.db"
ENV_DB_PATH = "WQ_DB_PATH"


def _resolve_db_path(cli_db: Optional[str]) -> str:
    """优先级：--db > WQ_DB_PATH > 默认。"""
    if cli_db:
        return cli_db
    env = os.environ.get(ENV_DB_PATH)
    if env:
        return env
    return DEFAULT_DB_PATH


def _fetch_expression(
    conn: sqlite3.Connection,
    alpha_id: str,
) -> str:
    """按 alpha_id 取表达式（JOIN expressions）。失败返回空串。"""
    try:
        row = conn.execute(
            "SELECT x.expression "
            "FROM evaluations e "
            "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
            "WHERE e.alpha_id = ?",
            (alpha_id,),
        ).fetchone()
        if row is None:
            return ""
        return str(row[0] or "")
    except sqlite3.Error:
        return ""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mark an alpha as submitted in the local DB.",
    )
    parser.add_argument(
        "alpha_id",
        help="BRAIN alpha ID to mark as submitted",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note (currently printed only, not stored)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "SQLite database path "
            f"(default: ${ENV_DB_PATH} or {DEFAULT_DB_PATH})"
        ),
    )
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db)
    if not os.path.isfile(db_path):
        print(f"database not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    try:
        # 先查是否已提交（用于打印"已经提交过"）
        already = False
        try:
            row = conn.execute(
                "SELECT submitted FROM evaluations WHERE alpha_id = ?",
                (args.alpha_id,),
            ).fetchone()
        except sqlite3.Error as e:
            print(f"query failed: {e}", file=sys.stderr)
            return 1

        if row is None:
            print(f"未找到 alpha_id: {args.alpha_id}", file=sys.stderr)
            return 1

        already = bool(row[0])

        now = datetime.now().isoformat()
        with conn:
            cur = conn.execute(
                "UPDATE evaluations "
                "SET submitted = 1, submitted_at = ? "
                "WHERE alpha_id = ?",
                (now, args.alpha_id),
            )
        affected = cur.rowcount or 0

        if affected == 0:
            print(f"未找到 alpha_id: {args.alpha_id}", file=sys.stderr)
            return 1

        if already:
            print("已经提交过")

        expr = _fetch_expression(conn, args.alpha_id)
        print(f"affected rows: {affected}")
        print(f"expression: {expr[:120]}")

        if args.note:
            # TODO: --note 目前仅打印，不落库（不改 schema）。
            print(f"note (not stored): {args.note}")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
