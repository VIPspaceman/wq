# -*- coding: utf-8 -*-

"""
group_neutralize 变体测试：把 group_neutralize 包在 3 条已知 7-PASS
变体外面，提交 sim，测 self-corr 是否降。

背景
----
tools/test_group_neutralize.py 已验证 BRAIN 接受
group_neutralize(expr, group) 语法。

本脚本测试：包一层 group_neutralize 后，原来自-corr FAIL 的 alpha
是否能通过。

输入（3 条 7-PASS 的 alpha_id）
-------------------------------
    MP3vgnnn  (parent=O086Z7pp, close→high 变体)
    GrOjKoZZ  (parent=wpbZnVmY, close→open 变体)
    Vk0QL6a5  (parent=j28Arpej, close→low 变体)

从 DB 读它们的完整 expression（含 winsorize/ts_backfill 包装）。

输出
----
每条生成 2 个变体（包在最外层）：
    group_neutralize(<原表达式>, sector)
    group_neutralize(<原表达式>, industry)

共 6 条 sim。

流程
----
    1. 登录 + 加载远程算子池
    2. 从 DB 拉 3 条 expression
    3. 构造 6 条新表达式
    4. 用 build_executor + Executor.run([...]) 提交 6 条
    5. 结果写入 DB（experiment_id="exp_group_neut_test1"）
    6. 对 7 项 PASS 的调 /check 拿 self-corr

用法
----
    python tools/test_group_neutralize_variants.py
    python tools/test_group_neutralize_variants.py --db data/wq.db
    python tools/test_group_neutralize_variants.py --dry-run

约束
----
- 只提交 6 条 sim（固定，不自动扩量）
- 整脚本最多 15 分钟（超时保护）
- 不改其他文件
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 让脚本在 tools/ 目录下也能 import 项目根下的包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


logger = logging.getLogger(__name__)


# ============================================================
# 常量
# ============================================================

DEFAULT_DB = "data/wq.db"
EXPERIMENT_ID = "exp_group_neut_test1"

# 3 条已知 7-PASS 的 alpha_id
TARGET_ALPHA_IDS: Tuple[str, ...] = (
    "MP3vgnnn",
    "GrOjKoZZ",
    "Vk0QL6a5",
)

# 每条生成 2 个变体
GROUPS: Tuple[str, ...] = ("sector", "industry")

# 整脚本超时（秒）
TOTAL_TIMEOUT_SECONDS = 15 * 60

# self-corr 检查间隔
SELF_CORR_SLEEP = 5.0

# 视为"忽略"的 check（提交时才结算）
IGNORE_CHECKS = {"SELF_CORRELATION"}

# sim 阶段 7 项（不含 SELF_CORRELATION）
SIM_STAGE_CHECKS = (
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "MATCHES_COMPETITION",
)


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_group_neutralize_variants",
        description=(
            "把 group_neutralize 包在 3 条已知 7-PASS 变体外面，"
            "提交 sim，测 self-corr 是否降"
        ),
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"数据库路径（默认 {DEFAULT_DB}）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只构造表达式，不提交",
    )
    return parser.parse_args(argv)


# ============================================================
# 登录 + 加载远程算子池
# ============================================================

def _login(settings):
    """登录 BRAIN，返回已认证 Session。"""
    from brain.authentication import create_authenticated_session

    return create_authenticated_session(
        settings.credentials,
        settings.endpoints,
        request_timeout=settings.polling.request_timeout,
    )


def _load_remote_operators(settings, session) -> bool:
    """加载远程算子池（合并进本地 catalog）。"""
    from alpha.operators import load_from_api

    try:
        catalog = load_from_api(
            session, settings.endpoints, settings.polling,
        )
    except Exception:
        logger.exception("加载远程算子池异常")
        return False

    if catalog is None:
        logger.warning("远程算子池为空")
        return False

    logger.info("远程算子池已加载: %s", catalog.summary())
    return True


# ============================================================
# 从 DB 读 expression
# ============================================================

def _load_expressions(
    db_path: str, alpha_ids: Tuple[str, ...],
) -> Dict[str, str]:
    """
    从 DB 读 alpha_id → expression 映射。

    用 evaluations JOIN expressions。
    找不到的 alpha_id 不出现在返回 dict 里。
    """
    import sqlite3

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in alpha_ids)
        sql = (
            "SELECT e.alpha_id, x.expression "
            "FROM evaluations e "
            "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
            f"WHERE e.alpha_id IN ({placeholders})"
        )
        rows = conn.execute(sql, list(alpha_ids)).fetchall()
    finally:
        conn.close()

    out: Dict[str, str] = {}
    for r in rows:
        aid = str(r["alpha_id"] or "")
        expr = str(r["expression"] or "")
        if aid and expr:
            out[aid] = expr
    return out


# ============================================================
# 构造变体
# ============================================================

def _build_variants(
    expr_by_alpha: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    构造 6 条变体。

    返回每项：
        {
            "parent_alpha_id": str,
            "parent_expression": str,
            "group": str,
            "expression": str,
        }
    """
    out: List[Dict[str, Any]] = []
    for aid in TARGET_ALPHA_IDS:
        parent_expr = expr_by_alpha.get(aid)
        if not parent_expr:
            logger.warning("DB 里找不到 alpha_id=%s 的 expression", aid)
            continue
        for group in GROUPS:
            out.append({
                "parent_alpha_id": aid,
                "parent_expression": parent_expr,
                "group": group,
                "expression": (
                    f"group_neutralize({parent_expr},{group})"
                ),
            })
    return out


# ============================================================
# 提交
# ============================================================

def _run_submit(
    settings,
    variants: List[Dict[str, Any]],
    db_path: str,
) -> int:
    """
    提交 6 条变体，写 DB，查 self-corr，打印统计。

    返回 0 成功，非 0 失败。
    """
    from execution.task import Task, TaskState
    from search.miner import build_executor
    from storage.database import WQDatabase
    from evaluation.record import build_record
    from alpha.parser import parse_expression
    from core.signature import structure_signature

    # -------- 1. 打开 DB --------
    try:
        db = WQDatabase(db_path)
        db.init_schema()
        device_id = db.get_or_create_device("group-neut-test")
    except Exception as e:
        print(f"❌ 打开 DB 失败: {e}", file=sys.stderr)
        return 2

    # -------- 2. 构建 executor --------
    try:
        executor = build_executor(settings)
    except Exception as e:
        print(f"❌ 构建 executor 失败: {e}", file=sys.stderr)
        db.close()
        return 2

    # -------- 3. 变体 → Task --------
    variant_by_expr: Dict[str, Dict[str, Any]] = {}
    tasks: List[Task] = []
    for v in variants:
        t = Task(expression=v["expression"], source="GROUP_NEUT")
        tasks.append(t)
        variant_by_expr[v["expression"]] = v

    print()
    print(f"提交 {len(tasks)} 条变体 ...")

    # -------- 4. 提交（带超时保护） --------
    error_holder: Dict[str, Any] = {}

    def _run() -> None:
        try:
            executor.run(tasks)
        except Exception as e:
            error_holder["error"] = repr(e)

    t = threading.Thread(target=_run, daemon=True)
    start = time.time()
    t.start()
    t.join(timeout=TOTAL_TIMEOUT_SECONDS)
    elapsed = time.time() - start

    if t.is_alive():
        print(
            f"⚠️ 提交超时（>{TOTAL_TIMEOUT_SECONDS}s），"
            f"已耗时 {elapsed:.0f}s"
        )
        print("（后台线程仍在运行，进程退出时会终止）")
        db.close()
        return 3

    if "error" in error_holder:
        print(f"❌ 提交异常: {error_holder['error']}")

    print(f"提交耗时: {elapsed:.1f}s")

    # -------- 5. 写 DB --------
    completed = [t for t in tasks if t.state == TaskState.COMPLETED]
    failed = [t for t in tasks if t.state == TaskState.FAILED]

    written = 0
    for task in completed:
        v = variant_by_expr.get(task.expression)
        if v is None or task.result is None:
            continue
        try:
            node = parse_expression(task.expression)
            if node is None:
                continue
            record = build_record(node, task.result)

            # ---- 打 tags ----
            record.add_tag("GROUP_NEUT")
            record.add_tag(f"group_neutralize:{v['group']}")
            record.add_tag(f"variant_of:{v['parent_alpha_id']}")

            # ---- 父种子结构签名 ----
            parent_sig = None
            try:
                pnode = parse_expression(v["parent_expression"])
                if pnode is not None:
                    parent_sig = structure_signature(pnode)
            except Exception:
                parent_sig = None

            ok_db = db.save_record(
                record=record.to_dict(),
                experiment_id=EXPERIMENT_ID,
                device_id=device_id,
                generation=0,
                parent_structure=parent_sig,
            )
            if ok_db:
                written += 1
        except Exception:
            logger.exception(
                "写 DB 失败: %s", task.expression[:60]
            )

    # -------- 6. self-corr 检查 --------
    self_corr_stats = _check_self_corr(
        settings, completed, variant_by_expr, db,
    )

    # -------- 7. 打印每条结果 --------
    print()
    print("=" * 70)
    print("Group Neutralize Variants Results")
    print("=" * 70)

    for task in tasks:
        v = variant_by_expr.get(task.expression)
        if v is None:
            continue
        print()
        print(
            f"=== group_neutralize("
            f"{v['parent_alpha_id']}, {v['group']}) ==="
        )

        if task.state != TaskState.COMPLETED or task.result is None:
            print(f"  state: {task.state.value}")
            if task.error:
                print(f"  error: {task.error}")
            continue

        result = task.result
        is_data = result.get("is", {}) or {}
        alpha_id = str(result.get("_alpha_id") or "")

        print(f"  Alpha ID={alpha_id}")
        print(
            f"  sharpe={is_data.get('sharpe')} "
            f"fitness={is_data.get('fitness')} "
            f"turnover={is_data.get('turnover')}"
        )

        # 7-check
        checks = _extract_checks_map(result)
        seven_ok, seven_reason = _seven_check_status(checks)
        if seven_ok:
            print("  7-check: PASS")
        else:
            print(f"  7-check: FAIL({seven_reason})")

        # self-corr
        sc = self_corr_stats.get(task.expression, "N/A")
        print(f"  self-corr: {sc}")

    # -------- 8. 统计 --------
    print()
    print("=" * 70)
    print("统计")
    print("=" * 70)
    print(f"提交 {len(tasks)} 条")
    print(f"Completed: {len(completed)}")
    print(f"Failed:    {len(failed)}")
    print(f"DB written: {written}")

    seven_pass = 0
    selfcorr_pass = 0
    by_group: Dict[str, Dict[str, int]] = {
        g: {"seven": 0, "selfcorr": 0, "submit": 0}
        for g in GROUPS
    }

    for task in tasks:
        v = variant_by_expr.get(task.expression)
        if v is None:
            continue
        g = v["group"]
        by_group[g]["submit"] += 1

        if task.result:
            checks = _extract_checks_map(task.result)
            ok, _ = _seven_check_status(checks)
            if ok:
                seven_pass += 1
                by_group[g]["seven"] += 1

        sc = self_corr_stats.get(task.expression)
        if sc == "PASS":
            selfcorr_pass += 1
            by_group[g]["selfcorr"] += 1

    print()
    print(f"7 项 PASS: {seven_pass} 条")
    print(f"self-corr PASS: {selfcorr_pass} 条")
    print()
    print("按 group 分组:")
    for g in GROUPS:
        s = by_group[g]
        print(
            f"  {g:10s}: submit={s['submit']} / "
            f"7check={s['seven']} / selfcorr={s['selfcorr']}"
        )

    db.close()
    print("=" * 70)
    return 0


# ============================================================
# self-corr 检查
# ============================================================

def _check_self_corr(
    settings,
    completed: List[Any],
    variant_by_expr: Dict[str, Dict[str, Any]],
    db: Any,
) -> Dict[str, str]:
    """
    对已完成的变体调 /check 拿 self-corr，并写回 DB。

    返回 {expression: self_corr_result}。
    """
    from brain.authentication import create_authenticated_session
    from brain.client import BrainClient
    from evaluation.official_checks import parse_official_checks

    out: Dict[str, str] = {}

    targets: List[Tuple[str, str]] = []
    for task in completed:
        if not task.result:
            continue
        alpha_id = str(task.result.get("_alpha_id") or "")
        if not alpha_id:
            continue
        targets.append((task.expression, alpha_id))

    if not targets:
        return out

    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
        client = BrainClient(
            session, settings.endpoints, settings.polling,
        )
    except Exception:
        logger.exception("self-corr 检查：登录失败")
        return out

    try:
        for i, (expr, alpha_id) in enumerate(targets, start=1):
            data = client.check_alpha(alpha_id)
            if not data:
                result = "ERROR"
            else:
                checks = parse_official_checks(data)
                result = checks.get("SELF_CORRELATION", "UNKNOWN")
            out[expr] = result

            try:
                db.update_self_corr(alpha_id, result)
            except Exception:
                logger.exception(
                    "update_self_corr 失败: %s", alpha_id
                )

            print(
                f"  self-corr [{i}/{len(targets)}] "
                f"{alpha_id} → {result}"
            )
            if i < len(targets):
                time.sleep(SELF_CORR_SLEEP)
    finally:
        try:
            session.close()
        except Exception:
            pass

    return out


# ============================================================
# checks 辅助
# ============================================================

def _extract_checks_map(raw_json: Any) -> Dict[str, str]:
    """
    从 result / raw_json 中提取 {check_name: result}。

    支持 {"is": {...}} / {"is_raw": {...}} / {"checks": [...]}。
    不抛异常；无数据返回 {}。
    """
    if raw_json is None:
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


def _seven_check_status(
    checks: Dict[str, str],
) -> Tuple[bool, str]:
    """
    判断 sim 阶段 7 项是否全 PASS。

    返回 (ok, reason)。
    reason 为失败项名称（逗号分隔），ok=True 时为空串。
    """
    if not checks:
        return False, "no_checks"

    failed: List[str] = []
    for name in SIM_STAGE_CHECKS:
        result = checks.get(name)
        if result is None:
            failed.append(f"{name}:MISSING")
            continue
        if result != "PASS":
            failed.append(f"{name}:{result}")

    if failed:
        return False, ",".join(failed)
    return True, ""


# ============================================================
# main
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args(argv)

    print()
    print("=" * 70)
    print("group_neutralize 变体测试")
    print("=" * 70)
    print(f"  目标 alpha_id: {', '.join(TARGET_ALPHA_IDS)}")
    print(f"  groups:        {', '.join(GROUPS)}")
    print(f"  预期提交:      {len(TARGET_ALPHA_IDS) * len(GROUPS)} 条")
    print(f"  experiment_id: {EXPERIMENT_ID}")
    print(f"  模式:          {'dry-run' if args.dry_run else '提交'}")
    print()

    # -------- 1. 加载配置 --------
    try:
        from config.settings import AppSettings
        settings = AppSettings.from_env()
    except Exception as e:
        print(f"❌ 加载配置失败: {e}", file=sys.stderr)
        return 2

    # -------- 2. 登录 --------
    try:
        session = _login(settings)
    except Exception as e:
        print(f"❌ BRAIN 登录失败: {e}", file=sys.stderr)
        return 2

    # -------- 3. 加载远程算子池 --------
    _load_remote_operators(settings, session)

    # -------- 4. 从 DB 读 expression --------
    try:
        expr_by_alpha = _load_expressions(
            args.db, TARGET_ALPHA_IDS,
        )
    except Exception as e:
        print(f"❌ 读取 DB 失败: {e}", file=sys.stderr)
        try:
            session.close()
        except Exception:
            pass
        return 2

    missing = [
        aid for aid in TARGET_ALPHA_IDS
        if aid not in expr_by_alpha
    ]
    if missing:
        print(
            f"⚠️ DB 里找不到以下 alpha_id 的 expression: "
            f"{', '.join(missing)}"
        )

    if not expr_by_alpha:
        print("❌ 没有任何可用的 expression", file=sys.stderr)
        try:
            session.close()
        except Exception:
            pass
        return 1

    # -------- 5. 构造变体 --------
    variants = _build_variants(expr_by_alpha)

    print(f"构造 {len(variants)} 条变体:")
    for v in variants:
        print(
            f"  [{v['parent_alpha_id']} / {v['group']}] "
            f"{v['expression'][:100]}"
        )
    print()

    # -------- 6. dry-run --------
    if args.dry_run:
        print("(dry-run) 未提交。")
        try:
            session.close()
        except Exception:
            pass
        return 0

    # -------- 7. 提交 --------
    try:
        rc = _run_submit(settings, variants, args.db)
    finally:
        try:
            session.close()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
