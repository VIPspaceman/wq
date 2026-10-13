# -*- coding: utf-8 -*-

"""
提交前检查：用 BRAIN GET /alphas/{alpha_id}/check 补全本地检查。

背景
----
本地只有 3 项门槛（sharpe / fitness / turnover），
BRAIN 的 /check 端点能返回 7 项官方 checks。
本工具在提交前调用该端点，只保留"7 项全 PASS"的候选。

注意
----
- SELF_CORRELATION 通常返回 PENDING（提交时才由 BRAIN 判定），
  本工具将其视为"忽略"，不计入 FAIL。
- 本工具 **不提交** 任何 alpha，只做只读检查。

用法
----
    python -m tools.check_before_submit --limit 20
    python -m tools.check_before_submit --dry-run
    python -m tools.check_before_submit --db data/wq.db --output out.csv

约束
----
- 不新增依赖（httpx / dotenv / sqlite3 均为已有）
- 每条之间 sleep 1 秒，避免撞 daemon 的并发槽
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from brain.authentication import create_authenticated_session
from config.settings import AppSettings
from evaluation.official_checks import parse_official_checks


logger = logging.getLogger(__name__)


# ============================================================
# 常量
# ============================================================

DEFAULT_DB = "data/wq.db"
DEFAULT_LIMIT = 20
SLEEP_BETWEEN = 5.0

# 429 退避
RATE_LIMIT_MAX_RETRY = 3
RATE_LIMIT_BACKOFFS = [5.0, 10.0, 20.0]
RETRY_AFTER_MAX_SECONDS = 60.0

# 空 body 重试（/check 是异步端点，self-corr 可能未算完）
EMPTY_BODY_MAX_RETRY = 3
EMPTY_BODY_WAIT = 10.0

# 本地门槛（与 AutoSubmitSettings 默认值对齐）
MIN_SHARPE = 1.25
MIN_FITNESS = 1.0
MIN_TURNOVER = 0.01
MAX_TURNOVER = 0.70

# 视为"忽略"的 check（提交时才结算）
IGNORE_CHECKS = {"SELF_CORRELATION"}

# 期望的 7 项 check 名称（仅用于展示列顺序，缺失则留空）
EXPECTED_CHECKS = [
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "SELF_CORRELATION",
]


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_before_submit",
        description="提交前用 BRAIN /check 端点补全本地检查",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"最多检查 N 条（默认 {DEFAULT_LIMIT}）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只从 DB 读并输出候选，不调 API",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"数据库路径（默认 {DEFAULT_DB}）",
    )
    parser.add_argument(
        "--output", default=None,
        help="输出 CSV 路径（默认自动生成到最新实验目录）",
    )
    return parser.parse_args(argv)


# ============================================================
# DB 读取（容错列探测）
# ============================================================

def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        rows = conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [str(r[1]) for r in rows]


def _pick_column(
    columns: List[str], candidates: List[str],
) -> Optional[str]:
    """从 columns 中挑第一个存在的候选列名（大小写不敏感）。"""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _find_alpha_table(conn: sqlite3.Connection) -> Optional[str]:
    """找到存 alpha 记录的表名。"""
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table'"
        ).fetchall()
    except sqlite3.Error:
        return None
    names = [str(r[0]) for r in rows]
    for cand in ("alphas", "alpha", "records", "alpha_records"):
        for n in names:
            if n.lower() == cand:
                return n
    # 兜底：返回第一个含 alpha_id 列的表
    for n in names:
        cols = _table_columns(conn, n)
        if _pick_column(cols, ["alpha_id"]) is not None:
            return n
    return None


def _load_candidates(
    db_path: str, limit: int,
) -> List[Dict[str, Any]]:
    """
    从 DB 读取"未提交 + 本地通过门槛"的候选。

    列名通过 PRAGMA 探测，兼容不同 schema。
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        table = _find_alpha_table(conn)
        if table is None:
            raise RuntimeError("找不到 alpha 记录表")

        cols = _table_columns(conn, table)
        col_alpha_id = _pick_column(cols, ["alpha_id"])
        col_expr = _pick_column(cols, ["expression", "expr"])
        col_sharpe = _pick_column(cols, ["sharpe"])
        col_fitness = _pick_column(cols, ["fitness"])
        col_turnover = _pick_column(cols, ["turnover"])
        col_submitted = _pick_column(
            cols, ["submitted", "is_submitted"],
        )

        if col_alpha_id is None:
            raise RuntimeError(
                f"表 {table} 缺少 alpha_id 列"
            )

        select_cols = [col_alpha_id]
        for c in (col_expr, col_sharpe, col_fitness, col_turnover):
            select_cols.append(c if c else "NULL")

        where_parts: List[str] = []
        params: List[Any] = []

        if col_submitted is not None:
            where_parts.append(
                f"({col_submitted} IS NULL OR {col_submitted} = 0)"
            )
        if col_sharpe is not None:
            where_parts.append(f"{col_sharpe} >= ?")
            params.append(MIN_SHARPE)
        if col_fitness is not None:
            where_parts.append(f"{col_fitness} >= ?")
            params.append(MIN_FITNESS)
        if col_turnover is not None:
            where_parts.append(f"{col_turnover} >= ?")
            params.append(MIN_TURNOVER)
            where_parts.append(f"{col_turnover} <= ?")
            params.append(MAX_TURNOVER)

        sql = f"SELECT {', '.join(select_cols)} FROM {table}"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += " LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "alpha_id": str(r[0] or ""),
            "expression": str(r[1] or ""),
            "sharpe": _as_float(r[2]),
            "fitness": _as_float(r[3]),
            "turnover": _as_float(r[4]),
        })
    return out


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# 输出路径
# ============================================================

def _resolve_output_path(cli_output: Optional[str]) -> str:
    """默认输出到最新实验目录。"""
    if cli_output:
        return cli_output

    base = "experiments"
    if not os.path.isdir(base):
        return "check_candidates.csv"

    exp_ids = [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ]
    if not exp_ids:
        return "check_candidates.csv"

    exp_ids.sort()
    latest = exp_ids[-1]
    return os.path.join(base, latest, "check_candidates.csv")


# ============================================================
# 检查逻辑
# ============================================================

def _is_pass(results: Dict[str, str]) -> bool:
    """
    所有返回的 check 都必须是 PASS 才返回 True。

    唯一例外：SELF_CORRELATION=PENDING 时忽略（提交时才结算）。
    SELF_CORRELATION=FAIL 仍判 FAIL。
    无 checks 数据 → False。
    """
    if not results:
        return False
    for name, result in results.items():
        if result == "PASS":
            continue
        if name in IGNORE_CHECKS and result == "PENDING":
            continue
        return False
    return True


def _fail_reasons(results: Dict[str, str]) -> List[str]:
    """
    返回所有非 PASS 的 check 名称。

    唯一例外：SELF_CORRELATION=PENDING 时忽略。
    """
    out: List[str] = []
    for name, result in results.items():
        if result == "PASS":
            continue
        if name in IGNORE_CHECKS and result == "PENDING":
            continue
        out.append(name)
    return out


def _retry_after_seconds(resp: httpx.Response) -> Optional[float]:
    """从 Retry-After 头解析等待秒数（仅数字形式）。"""
    raw = resp.headers.get("Retry-After") or resp.headers.get(
        "retry-after"
    )
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return min(value, RETRY_AFTER_MAX_SECONDS)


def _check_one(
    client: httpx.Client,
    base_url: str,
    alpha_id: str,
) -> Dict[str, Any]:
    """
    调 GET /alphas/{alpha_id}/check。

    返回结构化结果：
        {"status": "ok",           "checks": {name: result}}
        {"status": "rate_limited", "detail": str}
        {"status": "parse_error",  "detail": str}
        {"status": "http_error",   "detail": str}
        {"status": "network_error","detail": str}

    429 时按 Retry-After 或指数退避重试，最多 RATE_LIMIT_MAX_RETRY 次。

    /check 是异步端点：首次调用可能返回 200 + 空 body
    （self-corr 尚未算完）。此时等待 EMPTY_BODY_WAIT 秒后重试，
    最多 EMPTY_BODY_MAX_RETRY 次；三次都空才判 parse_error。
    """
    url = f"{base_url}/alphas/{alpha_id}/check"

    rate_limit_attempt = 0
    empty_body_attempt = 0

    while True:
        try:
            resp = client.get(url)
        except httpx.HTTPError as e:
            logger.warning("check 请求异常 %s: %r", alpha_id, e)
            return {"status": "network_error", "detail": repr(e)}

        # -------- 429：退避重试 --------
        if resp.status_code == 429:
            if rate_limit_attempt >= RATE_LIMIT_MAX_RETRY:
                logger.warning(
                    "check 429 重试耗尽 %s", alpha_id,
                )
                return {
                    "status": "rate_limited",
                    "detail": "429 retries exhausted",
                }
            wait = _retry_after_seconds(resp)
            if wait is None:
                wait = RATE_LIMIT_BACKOFFS[
                    min(
                        rate_limit_attempt,
                        len(RATE_LIMIT_BACKOFFS) - 1,
                    )
                ]
            logger.warning(
                "check 429 %s，等待 %.1fs 后重试 (%d/%d)",
                alpha_id, wait,
                rate_limit_attempt + 1, RATE_LIMIT_MAX_RETRY,
            )
            time.sleep(wait)
            rate_limit_attempt += 1
            continue

        # -------- 非 200 --------
        if resp.status_code != 200:
            logger.warning(
                "check HTTP %d for %s", resp.status_code, alpha_id,
            )
            return {
                "status": "http_error",
                "detail": f"HTTP {resp.status_code}",
            }

        # -------- 解析 JSON --------
        body_text = resp.text or ""
        data = None
        if body_text.strip():
            try:
                data = resp.json()
            except ValueError:
                data = None

        # -------- 空 body / 解析失败：异步未算完，重试 --------
        if data is None:
            if empty_body_attempt >= EMPTY_BODY_MAX_RETRY:
                snippet = body_text[:200]
                logger.warning(
                    "check 空 body 重试耗尽 %s | body[:200]=%r",
                    alpha_id, snippet,
                )
                return {
                    "status": "parse_error",
                    "detail": snippet,
                }
            empty_body_attempt += 1
            logger.warning(
                "[重试 %d/%d] %s 空 body，%.0f 秒后重试",
                empty_body_attempt, EMPTY_BODY_MAX_RETRY,
                alpha_id, EMPTY_BODY_WAIT,
            )
            time.sleep(EMPTY_BODY_WAIT)
            continue

        return {
            "status": "ok",
            "checks": parse_official_checks(data),
        }


# ============================================================
# CSV 输出
# ============================================================

def _write_csv(
    path: str,
    rows: List[Dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fieldnames = [
        "alpha_id", "sharpe", "fitness", "turnover",
        "verdict",
    ] + EXPECTED_CHECKS + ["expression"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            checks = r.get("checks") or {}
            row = {
                "alpha_id": r.get("alpha_id", ""),
                "sharpe": r.get("sharpe", 0.0),
                "fitness": r.get("fitness", 0.0),
                "turnover": r.get("turnover", 0.0),
                "verdict": r.get("verdict", ""),
                "expression": (r.get("expression", "") or "")[:80],
            }
            for name in EXPECTED_CHECKS:
                row[name] = checks.get(name, "")
            writer.writerow(row)


# ============================================================
# main
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args(argv)

    # -------- 1. 读候选 --------
    try:
        candidates = _load_candidates(args.db, args.limit)
    except Exception as e:
        print(f"❌ 读取候选失败: {e}", file=sys.stderr)
        return 2

    if not candidates:
        print("没有符合条件的候选（未提交 + 本地门槛通过）")
        return 0

    print()
    print(f"=== 检查 {len(candidates)} 条候选 ===")

    # -------- 2. dry-run：只输出候选 --------
    if args.dry_run:
        for i, c in enumerate(candidates, start=1):
            print(
                f"[{i}/{len(candidates)}] {c['alpha_id']}  "
                f"sharpe={c['sharpe']:+.4f} "
                f"fitness={c['fitness']:+.4f} "
                f"turnover={c['turnover']:.4f}  "
                f"{c['expression'][:80]}"
            )
        out_path = _resolve_output_path(args.output)
        _write_csv(out_path, candidates)
        print()
        print(f"(dry-run) 候选已写入: {out_path}")
        return 0

    # -------- 3. 登录 --------
    try:
        settings = AppSettings.from_env()
    except Exception as e:
        print(f"❌ 加载配置失败: {e}", file=sys.stderr)
        return 2

    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
    except Exception as e:
        print(f"❌ BRAIN 登录失败: {e}", file=sys.stderr)
        return 2

    base_url = settings.endpoints.base_url

    # -------- 4. 逐条检查 --------
    results_rows: List[Dict[str, Any]] = []
    passed = 0
    rejected = 0
    reason_counts: Dict[str, int] = {}

    with httpx.Client(
        cookies=session.cookies,
        timeout=settings.polling.request_timeout,
    ) as client:
        for i, c in enumerate(candidates, start=1):
            alpha_id = c["alpha_id"]
            outcome = _check_one(client, base_url, alpha_id)
            status = outcome.get("status")

            if status == "ok":
                checks = outcome.get("checks") or {}
                if _is_pass(checks):
                    verdict = "PASS"
                    reasons = []
                    passed += 1
                else:
                    verdict = "FAIL"
                    reasons = _fail_reasons(checks)
                    rejected += 1
                    for r in reasons:
                        reason_counts[r] = (
                            reason_counts.get(r, 0) + 1
                        )
            else:
                # rate_limited / parse_error / http_error / network_error
                checks = {}
                verdict = "ERROR"
                reasons = []
                reason_counts[status] = (
                    reason_counts.get(status, 0) + 1
                )

            reason_str = (
                f" ({', '.join(reasons)})" if reasons else ""
            )
            print(
                f"[{i}/{len(candidates)}] {alpha_id}  "
                f"→ {verdict}{reason_str}"
            )

            row = dict(c)
            row["verdict"] = verdict
            row["checks"] = checks
            results_rows.append(row)

            if i < len(candidates):
                time.sleep(SLEEP_BETWEEN)

    # -------- 5. 写 CSV --------
    out_path = _resolve_output_path(args.output)
    _write_csv(out_path, results_rows)

    # -------- 6. 摘要 --------
    print()
    print("=== 摘要 ===")
    print(f"通过: {passed} 条")
    print(f"拒绝: {rejected} 条")
    if reason_counts:
        print("拒绝原因统计:")
        for name, cnt in sorted(
            reason_counts.items(),
            key=lambda kv: kv[1],
            reverse=True,
        ):
            print(f"  {name}: {cnt}")
    print()
    print(f"CSV 已写入: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
