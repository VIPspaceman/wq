# -*- coding: utf-8 -*-

"""
analyst4 单信号验证：用 5 个 analyst4 字段生成 10 条单信号。

背景
----
账号有 analyst4 权限（50 字段）。本脚本验证这些字段能否单独
产出 7-PASS alpha。

字段（人工挑选，coverage 高 + 经济含义明确）
--------------------------------------------
    1. anl4_afv4_eps_high    (EPS 最高估计, cov=1.00)
    2. anl4_afv4_dts_spe     (EPS 估计标准差, cov=0.69)
    3. anl4_adxqfv110_pu     (上调估计数, cov=0.70)
    4. anl4_adxqfv110_down   (下调估计数, cov=0.70)
    5. anl4_af_cfps_value    (现金流实际值, cov=0.62)

流程
----
    1. 加载远程算子池
    2. 加载 analyst4 字段池（合并进本地 catalog）
    3. 构造 10 条表达式
    4. validate_node 校验
    5. 提交 10 条 sim
    6. 对 7-PASS 的调 /check 拿 self-corr
    7. 结果写 DB（experiment_id="exp_analyst4_test1"）
    8. 打印统计（按字段分组）

用法
----
    python tools/test_analyst4_signals.py --dry-run
    python tools/test_analyst4_signals.py
    python tools/test_analyst4_signals.py --limit 10

约束
----
- 只新建本文件
- 不改 alpha/fields.py（运行时加载 analyst4 字段池）
- 整脚本最多 30 分钟（超时保护）
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
DEFAULT_LIMIT = 10
EXPERIMENT_ID = "exp_multidataset_test1"

# dataset id
ANALYST4_DATASET_ID = "analyst4"
OPTION8_DATASET_ID = "option8"

# 整脚本超时（秒）
TOTAL_TIMEOUT_SECONDS = 30 * 60

# self-corr 检查
SELF_CORR_SLEEP = 5.0
SELF_CORR_MAX = 50

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

# 视为"忽略"的 check（提交时才结算）
IGNORE_CHECKS = {"SELF_CORRELATION"}


# ============================================================
# 信号列表
# ============================================================
#
# 7 条固定信号，分三类：
#
#   ANALYST4_FLIP（1 条）：
#       分析师字段 × pv1，整体取负（方向翻转）。
#
#   ANALYST4_IMPL（3 条）：
#       分析师字段 × pv1 实现类信号（波动率 / 收益差分）。
#
#   OPTION8_MOM（3 条）：
#       option8 字段（历史波动率）× pv1 动量类信号。
#
# 每项：(label, expression)
#   label 用于分组统计。

_SIGNALS: Tuple[Tuple[str, str], ...] = (
    # ---- ANALYST4_FLIP ----
    (
        "ANALYST4_FLIP",
        (
            "(-1*(rank(winsorize(ts_backfill(anl4_afv4_dts_spe,120)))"
            "*rank(ts_av_diff(close,5))))"
        ),
    ),
    # ---- ANALYST4_IMPL ----
    (
        "ANALYST4_IMPL",
        (
            "(rank(winsorize(ts_backfill(anl4_af_cfps_value,120)))"
            "*rank(ts_std_dev(returns,20)))"
        ),
    ),
    (
        "ANALYST4_IMPL",
        (
            "(rank(winsorize(ts_backfill(actual_eps_value_quarterly,120)))"
            "*rank(ts_std_dev(returns,20)))"
        ),
    ),
    (
        "ANALYST4_IMPL",
        (
            "(rank(winsorize(ts_backfill(anl4_af_cfps_value,120)))"
            "*rank(ts_delta(returns,10)))"
        ),
    ),
    # ---- OPTION8_MOM ----
    (
        "OPTION8_MOM",
        (
            "(rank(winsorize(ts_backfill(historical_volatility_10,120)))"
            "*rank(ts_av_diff(close,20)))"
        ),
    ),
    (
        "OPTION8_MOM",
        (
            "(rank(winsorize(ts_backfill(historical_volatility_120,120)))"
            "*(-1*rank(ts_av_diff(close,5))))"
        ),
    ),
    (
        "OPTION8_MOM",
        (
            "(rank(winsorize(ts_backfill(historical_volatility_10,120)))"
            "*rank(ts_mean(returns,10)))"
        ),
    ),
)


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_analyst4_signals",
        description="用 5 个 analyst4 字段生成 10 条单信号",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只生成不提交",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"最多提交 N 条（默认 {DEFAULT_LIMIT}）",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"数据库路径（默认 {DEFAULT_DB}）",
    )
    return parser.parse_args(argv)


# ============================================================
# 登录 + 加载远程 catalog
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


def _load_dataset_fields(
    settings, session, dataset_id: str,
) -> bool:
    """
    加载指定 dataset 的字段池（合并进本地 catalog）。

    ⚠️ 关键：必须在解析表达式之前调用，否则含该 dataset 字段的
    表达式会被 _check_fields 拒绝（本地 hardcoded_catalog 不含
    这些字段）。

    用 merge_with_local=True 保留本地 close/volume 等字段，
    以及之前已加载的其它 dataset 字段。

    返回 True 表示加载成功。
    """
    from alpha.fields import load_from_api

    try:
        catalog = load_from_api(
            session,
            settings.endpoints,
            settings.polling,
            region=settings.simulation.region,
            delay=settings.simulation.delay,
            universe=settings.simulation.universe,
            instrument_type=settings.simulation.instrument_type,
            dataset_id=dataset_id,
            max_fields=500,
            max_pages=1,
            merge_with_local=True,
        )
    except Exception:
        logger.exception("加载 %s 字段池异常", dataset_id)
        return False

    if catalog is None:
        logger.warning("%s 字段池为空", dataset_id)
        return False

    logger.info("%s 字段池已加载: %s", dataset_id, catalog.summary())
    return True


# ============================================================
# 表达式构造 + 校验
# ============================================================

def _build_signals() -> List[Dict[str, Any]]:
    """
    构造 10 条信号。

    返回每项：
        {
            "label": str,        # A_LONG / B_CROSS
            "template_idx": int, # 在 _SIGNALS 中的序号
            "expression": str,
        }
    """
    out: List[Dict[str, Any]] = []
    for idx, (label, expr) in enumerate(_SIGNALS):
        out.append({
            "label": label,
            "template_idx": idx,
            "expression": expr,
        })
    return out


def _validate_signals(
    signals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    用 validate_expression 过滤信号。

    解析失败 / 校验失败 → 剔除。
    """
    from alpha.validator import validate_expression

    out: List[Dict[str, Any]] = []
    reject_reasons: Dict[str, int] = {}

    for s in signals:
        expr = s["expression"]
        ok, reason = validate_expression(expr)
        if not ok:
            logger.warning(
                "validate 拒绝 [%s] %s | %s",
                s["label"], reason, expr[:80],
            )
            key = (reason or "unknown").split(":")[0].strip()
            reject_reasons[key] = reject_reasons.get(key, 0) + 1
            continue
        out.append(s)

    logger.info(
        "validate_expression 通过 %d / %d",
        len(out), len(signals),
    )
    if reject_reasons:
        logger.info("拒绝原因统计: %s", reject_reasons)
    return out


# ============================================================
# 提交
# ============================================================

def _run_submit(
    settings,
    signals: List[Dict[str, Any]],
    db_path: str,
) -> int:
    """
    提交信号，写 DB，查 self-corr，打印统计。

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
        device_id = db.get_or_create_device("analyst4-test")
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

    # -------- 3. 信号 → Task --------
    signal_by_expr: Dict[str, Dict[str, Any]] = {}
    tasks: List[Task] = []
    for s in signals:
        t = Task(expression=s["expression"], source="ANALYST4")
        tasks.append(t)
        signal_by_expr[s["expression"]] = s

    print()
    print(f"提交 {len(tasks)} 条信号 ...")

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
        s = signal_by_expr.get(task.expression)
        if s is None or task.result is None:
            continue
        try:
            node = parse_expression(task.expression)
            if node is None:
                continue
            record = build_record(node, task.result)

            # ---- 打 tags ----
            record.add_tag("ANALYST4")
            record.add_tag(f"analyst4_label:{s['label']}")
            record.add_tag(f"analyst4_tmpl:{s['template_idx']}")

            # ---- 结构签名（作为 parent_structure）----
            parent_sig = None
            try:
                parent_sig = structure_signature(node)
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
        settings, completed, signal_by_expr, db,
    )

    # -------- 7. 打印每条结果 --------
    print()
    print("=" * 70)
    print("Analyst4 Signal Results")
    print("=" * 70)

    for task in tasks:
        s = signal_by_expr.get(task.expression)
        if s is None:
            continue
        print()
        print(f"=== [{s['label']} / tmpl{s['template_idx']}] ===")
        print(f"  expr: {task.expression[:100]}")

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

        checks = _extract_checks_map(result)
        seven_ok, seven_reason = _seven_check_status(checks)
        if seven_ok:
            print("  7-check: PASS")
        else:
            print(f"  7-check: FAIL({seven_reason})")

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
    by_label: Dict[str, Dict[str, int]] = {}

    for task in tasks:
        s = signal_by_expr.get(task.expression)
        if s is None:
            continue
        label = s["label"]
        if label not in by_label:
            by_label[label] = {"submit": 0, "seven": 0, "selfcorr": 0}
        by_label[label]["submit"] += 1

        if task.result:
            checks = _extract_checks_map(task.result)
            ok, _ = _seven_check_status(checks)
            if ok:
                seven_pass += 1
                by_label[label]["seven"] += 1

        sc = self_corr_stats.get(task.expression)
        if sc == "PASS":
            selfcorr_pass += 1
            by_label[label]["selfcorr"] += 1

    print()
    print(f"7 项 PASS: {seven_pass} 条")
    print(f"self-corr PASS: {selfcorr_pass} 条")
    print()
    print("按类别分组:")
    for label in sorted(by_label.keys()):
        s = by_label[label]
        print(
            f"  {label:10s}: submit={s['submit']} / "
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
    signal_by_expr: Dict[str, Dict[str, Any]],
    db: Any,
) -> Dict[str, str]:
    """
    对已完成的信号调 /check 拿 self-corr，并写回 DB。

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

    targets = targets[:SELF_CORR_MAX]

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
    print("analyst4 单信号验证")
    print("=" * 70)
    print(f"  dataset_id:    {ANALYST4_DATASET_ID}")
    print(f"  experiment_id: {EXPERIMENT_ID}")
    print(f"  limit:         {args.limit}")
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

    # -------- 4. 加载字段池（关键！） --------
    # 先 analyst4，再 option8；都 merge 到本地 catalog。
    for ds in (ANALYST4_DATASET_ID, OPTION8_DATASET_ID):
        fields_ok = _load_dataset_fields(settings, session, ds)
        if not fields_ok:
            print(
                f"⚠️ {ds} 字段池加载失败，"
                f"含该 dataset 字段的表达式可能被 validate 拒绝",
                file=sys.stderr,
            )

    # -------- 5. 构造信号 --------
    signals = _build_signals()
    print(f"构造 {len(signals)} 条信号")

    # -------- 6. 校验 --------
    signals = _validate_signals(signals)

    if not signals:
        print("❌ 所有信号都被 validate 拒绝", file=sys.stderr)
        try:
            session.close()
        except Exception:
            pass
        return 1

    # -------- 6b. 截断 --------
    if len(signals) > args.limit:
        print(f"截断到 {args.limit} 条")
        signals = signals[: args.limit]

    print()
    print(f"最终信号数: {len(signals)}")
    for i, s in enumerate(signals, start=1):
        print(
            f"  {i:02d}. [{s['label']}] {s['expression'][:100]}"
        )
    print()

    # -------- 7. dry-run --------
    if args.dry_run:
        print("(dry-run) 未提交。")
        try:
            session.close()
        except Exception:
            pass
        return 0

    # -------- 8. 提交 --------
    try:
        rc = _run_submit(settings, signals, args.db)
    finally:
        try:
            session.close()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
