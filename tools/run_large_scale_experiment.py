# -*- coding: utf-8 -*-

"""
大样本跨族组合实验（200 条）。

目标
----
统计"新 dataset × pv1"组合的真实产出率。

样本量 200 条，约 8-12 小时，可以挂 daemon 跑。

实验设计（24 个实验单元 × 8 条 ≈ 200 条）
------------------------------------------
维度 1：新字段（6 个）
    anl4_af_cfps_value / actual_eps_value_quarterly /
    anl4_afv4_dts_spe / anl4_afv4_eps_high /
    historical_volatility_10 / historical_volatility_120

维度 2：pv1 分量（4 个）
    ts_av_diff(close, 5) / ts_av_diff(close, 20) /
    ts_std_dev(returns, 20) / ts_mean(returns, 10)

维度 3：组合形式（2 个）
    MUL: (rank(A) * rank(B))
    ADD: (rank(A) + (-1*rank(B)))

维度 4：符号（2 个）
    原向：A × B
    翻转：-1 × (A × B)

组合生成
--------
6 字段 × 4 pv1 × 2 形式 × 2 符号 = 96 组合
再从 96 组合里随机抽 200 条（允许重复不同窗口）

实际执行
--------
- 生成 200 条唯一表达式（用 expression_hash 去重）
- 如果 96 组合不够 200，加窗口扰动（5/10/20/60）扩展

表达式模板
----------
所有字段预处理统一为 winsorize(ts_backfill(field, 120))
所有组合外层带括号

示例：
    (rank(winsorize(ts_backfill(anl4_af_cfps_value,120)))
     * rank(ts_av_diff(close,5)))
    (-1*(rank(winsorize(ts_backfill(historical_volatility_10,120)))
     * rank(ts_std_dev(returns,20))))
    (rank(winsorize(ts_backfill(anl4_afv4_dts_spe,120)))
     + (-1*rank(ts_av_diff(close,20))))

流程
----
    1. 加载远程算子池
    2. 加载 analyst4 + option8 字段池（合并模式，不要覆盖）
    3. 生成 200 条唯一表达式
    4. validate_node 过滤（去掉语法错/深度超限）
    5. 提交 sim（复用 executor）
    6. 对 7-PASS 的调 /check 拿 self-corr
    7. 结果写 DB（experiment_id = "exp_largescale_test1"）
    8. 统计输出：
       - 提交数 / 7-PASS 数 / self-corr PASS 数
       - 按新字段分组
       - 按组合形式分组
       - 按符号分组

用法
----
    python tools/run_large_scale_experiment.py --dry-run
    python tools/run_large_scale_experiment.py
    python tools/run_large_scale_experiment.py --limit 200
    python tools/run_large_scale_experiment.py --max-minutes 720

约束
----
- 只新建本文件
- 复用现有 executor / DB 接口 / check_alpha
- 加超时保护
- 整脚本最多 max-minutes 分钟（默认 720 = 12 小时）
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 让脚本在 tools/ 目录下也能 import 项目根下的包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


logger = logging.getLogger(__name__)


# ============================================================
# 常量
# ============================================================

DEFAULT_DB = "data/wq.db"
DEFAULT_LIMIT = 200
DEFAULT_MAX_MINUTES = 720          # 12 小时
DEFAULT_SEED = 20261012
EXPERIMENT_ID = "exp_largescale_test1"

# dataset id
ANALYST4_DATASET_ID = "analyst4"
OPTION8_DATASET_ID = "option8"

# self-corr 检查
SELF_CORR_SLEEP = 5.0
SELF_CORR_MAX = 200

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
# 实验维度
# ============================================================

# 维度 1：新字段（6 个）
NEW_FIELDS: Tuple[str, ...] = (
    "anl4_af_cfps_value",
    "actual_eps_value_quarterly",
    "anl4_afv4_dts_spe",
    "anl4_afv4_eps_high",
    "historical_volatility_10",
    "historical_volatility_120",
)

# 维度 2：pv1 分量（4 个）
# 每项：(label, expression)
PV1_COMPONENTS: Tuple[Tuple[str, str], ...] = (
    ("av_diff_close_5", "ts_av_diff(close,5)"),
    ("av_diff_close_20", "ts_av_diff(close,20)"),
    ("std_dev_returns_20", "ts_std_dev(returns,20)"),
    ("mean_returns_10", "ts_mean(returns,10)"),
)

# 维度 3：组合形式（2 个）
FORM_MUL = "MUL"
FORM_ADD = "ADD"

# 维度 4：符号（2 个）
SIGN_NORMAL = "NORMAL"
SIGN_FLIP = "FLIP"

# 字段预处理窗口候选（用于扩展）
BACKFILL_WINDOWS: Tuple[int, ...] = (120, 250)

# pv1 窗口扰动候选（用于扩展）
PV1_WINDOW_VARIANTS: Tuple[int, ...] = (5, 10, 20, 60)


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_large_scale_experiment",
        description="大样本跨族组合实验（200 条）",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"生成 N 条唯一表达式（默认 {DEFAULT_LIMIT}）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只生成不提交",
    )
    parser.add_argument(
        "--max-minutes", type=int, default=DEFAULT_MAX_MINUTES,
        help=(
            f"总耗时上限（分钟，默认 {DEFAULT_MAX_MINUTES} "
            f"= 12 小时）"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"随机种子（默认 {DEFAULT_SEED}）",
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
# 表达式构造
# ============================================================

def _wrap_field(field_name: str, backfill_window: int = 120) -> str:
    """字段预处理：winsorize(ts_backfill(field, W))。"""
    return (
        f"winsorize(ts_backfill({field_name},{backfill_window}))"
    )


def _build_expression(
    field_name: str,
    pv1_expr: str,
    form: str,
    sign: str,
    *,
    backfill_window: int = 120,
) -> str:
    """
    构造组合表达式。

    MUL  : (rank(A) * rank(B))
    ADD  : (rank(A) + (-1*rank(B)))
    FLIP : -1 * (上述)
    """
    a = f"rank({_wrap_field(field_name, backfill_window)})"
    b = f"rank({pv1_expr})"

    if form == FORM_MUL:
        core = f"({a}*{b})"
    elif form == FORM_ADD:
        core = f"({a}+(-1*{b}))"
    else:
        raise ValueError(f"Unknown form: {form}")

    if sign == SIGN_FLIP:
        return f"(-1*{core})"
    return core


def _build_base_combinations() -> List[Dict[str, Any]]:
    """
    构造 96 个基础组合（6 × 4 × 2 × 2）。

    返回每项：
        {
            "field": str,
            "pv1_label": str,
            "pv1_expr": str,
            "form": str,
            "sign": str,
            "expression": str,
        }
    """
    out: List[Dict[str, Any]] = []
    for field_name in NEW_FIELDS:
        for pv1_label, pv1_expr in PV1_COMPONENTS:
            for form in (FORM_MUL, FORM_ADD):
                for sign in (SIGN_NORMAL, SIGN_FLIP):
                    expr = _build_expression(
                        field_name, pv1_expr, form, sign,
                    )
                    out.append({
                        "field": field_name,
                        "pv1_label": pv1_label,
                        "pv1_expr": pv1_expr,
                        "form": form,
                        "sign": sign,
                        "expression": expr,
                    })
    return out


def _build_extended_combinations(
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    构造扩展组合（窗口扰动）。

    用于在 96 基础组合不足 200 时补充。

    扰动维度：
        - backfill_window: 120 / 250
        - pv1 窗口: 5 / 10 / 20 / 60（替换 pv1_expr 里的窗口）
    """
    out: List[Dict[str, Any]] = []

    for field_name in NEW_FIELDS:
        for pv1_label, pv1_expr in PV1_COMPONENTS:
            for form in (FORM_MUL, FORM_ADD):
                for sign in (SIGN_NORMAL, SIGN_FLIP):
                    for bf_w in BACKFILL_WINDOWS:
                        for pv1_w in PV1_WINDOW_VARIANTS:
                            # 替换 pv1_expr 里的窗口
                            new_pv1 = _replace_window(
                                pv1_expr, pv1_w,
                            )
                            if new_pv1 is None:
                                continue
                            expr = _build_expression(
                                field_name, new_pv1, form, sign,
                                backfill_window=bf_w,
                            )
                            out.append({
                                "field": field_name,
                                "pv1_label": (
                                    f"{pv1_label}_w{pv1_w}"
                                ),
                                "pv1_expr": new_pv1,
                                "form": form,
                                "sign": sign,
                                "expression": expr,
                            })

    rng.shuffle(out)
    return out


def _replace_window(expr: str, new_window: int) -> Optional[str]:
    """
    把 ts_xxx(x, N) 里的 N 替换为 new_window。

    只处理单层 ts 调用（pv1 分量都是单层）。
    """
    import re

    m = re.match(r"^([a-z_]+)\((.+),(\d+)\)$", expr)
    if not m:
        return None
    op, inner, _old = m.groups()
    return f"{op}({inner},{new_window})"


def _generate_unique_expressions(
    limit: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    生成 limit 条唯一表达式。

    策略：
        1. 先取 96 基础组合
        2. 不足时用扩展组合补充
        3. expression_hash 去重
    """
    from core.hashing import expression_hash

    base = _build_base_combinations()
    rng.shuffle(base)

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def _try_add(c: Dict[str, Any]) -> bool:
        h = expression_hash(c["expression"])
        if h in seen:
            return False
        seen.add(h)
        c["expr_hash"] = h
        out.append(c)
        return True

    # ---- 1. 基础组合 ----
    for c in base:
        if len(out) >= limit:
            break
        _try_add(c)

    # ---- 2. 扩展组合 ----
    if len(out) < limit:
        extended = _build_extended_combinations(rng)
        for c in extended:
            if len(out) >= limit:
                break
            _try_add(c)

    logger.info(
        "生成 %d 条唯一表达式（目标 %d）", len(out), limit,
    )
    return out


# ============================================================
# 校验
# ============================================================

# 组合表达式嵌套两个分量的完整 AST，深度/节点数必然超过
# validate_node 的默认限制（5/15）。因此这里放宽到 30/60。
COMBINE_MAX_DEPTH = 30
COMBINE_MAX_NODES = 60


def _validate_combinations(
    combos: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    用 validate_node 过滤组合。

    解析失败 / 校验失败 → 剔除。
    """
    from alpha.parser import parse_expression
    from alpha.validator import validate_node

    out: List[Dict[str, Any]] = []
    reject_reasons: Dict[str, int] = {}

    for c in combos:
        expr = c["expression"]
        node = parse_expression(expr)
        if node is None:
            logger.debug("validate: 解析失败 %s", expr[:80])
            reject_reasons["parse_failed"] = (
                reject_reasons.get("parse_failed", 0) + 1
            )
            continue
        ok, reason = validate_node(
            node,
            max_depth=COMBINE_MAX_DEPTH,
            max_nodes=COMBINE_MAX_NODES,
        )
        if not ok:
            logger.debug(
                "validate: 拒绝 %s | %s", reason, expr[:80],
            )
            key = (reason or "unknown").split(":")[0].strip()
            reject_reasons[key] = reject_reasons.get(key, 0) + 1
            continue
        out.append(c)

    logger.info(
        "validate_node 通过 %d / %d", len(out), len(combos),
    )
    if reject_reasons:
        logger.info("拒绝原因统计: %s", reject_reasons)
    return out


# ============================================================
# 提交
# ============================================================

def _run_submit(
    settings,
    combos: List[Dict[str, Any]],
    db_path: str,
    max_minutes: int,
) -> int:
    """
    提交组合，写 DB，查 self-corr，打印统计。

    返回 0 成功，非 0 失败。
    """
    from execution.task import Task, TaskState
    from search.miner import build_executor
    from storage.database import WQDatabase
    from evaluation.record import build_record
    from alpha.parser import parse_expression
    from core.signature import structure_signature

    total_timeout = max_minutes * 60

    # -------- 1. 打开 DB --------
    try:
        db = WQDatabase(db_path)
        db.init_schema()
        device_id = db.get_or_create_device("largescale-tool")
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

    # -------- 3. 组合 → Task --------
    combo_by_expr: Dict[str, Dict[str, Any]] = {}
    tasks: List[Task] = []
    for c in combos:
        t = Task(expression=c["expression"], source="LARGESCALE")
        tasks.append(t)
        combo_by_expr[c["expression"]] = c

    print()
    print(f"提交 {len(tasks)} 条组合 ...")

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
    t.join(timeout=total_timeout)
    elapsed = time.time() - start

    if t.is_alive():
        print(
            f"⚠️ 提交超时（>{total_timeout}s），"
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
        c = combo_by_expr.get(task.expression)
        if c is None or task.result is None:
            continue
        try:
            node = parse_expression(task.expression)
            if node is None:
                continue
            record = build_record(node, task.result)

            # ---- 打 tags ----
            record.add_tag("LARGESCALE")
            record.add_tag(f"ls_field:{c['field']}")
            record.add_tag(f"ls_pv1:{c['pv1_label']}")
            record.add_tag(f"ls_form:{c['form']}")
            record.add_tag(f"ls_sign:{c['sign']}")

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
        settings, completed, combo_by_expr, db,
    )

    # -------- 7. 打印每条结果 --------
    print()
    print("=" * 70)
    print("Large Scale Experiment Results")
    print("=" * 70)

    for task in tasks:
        c = combo_by_expr.get(task.expression)
        if c is None:
            continue
        print()
        print(
            f"=== [{c['field']} × {c['pv1_label']} / "
            f"{c['form']} / {c['sign']}] ==="
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
    by_field: Dict[str, Dict[str, int]] = {}
    by_form: Dict[str, Dict[str, int]] = {}
    by_sign: Dict[str, Dict[str, int]] = {}

    def _bump(d: Dict[str, Dict[str, int]], key: str) -> None:
        if key not in d:
            d[key] = {"submit": 0, "seven": 0, "selfcorr": 0}
        d[key]["submit"] += 1

    for task in tasks:
        c = combo_by_expr.get(task.expression)
        if c is None:
            continue

        _bump(by_field, c["field"])
        _bump(by_form, c["form"])
        _bump(by_sign, c["sign"])

        if task.result:
            checks = _extract_checks_map(task.result)
            ok, _ = _seven_check_status(checks)
            if ok:
                seven_pass += 1
                by_field[c["field"]]["seven"] += 1
                by_form[c["form"]]["seven"] += 1
                by_sign[c["sign"]]["seven"] += 1

        sc = self_corr_stats.get(task.expression)
        if sc == "PASS":
            selfcorr_pass += 1
            by_field[c["field"]]["selfcorr"] += 1
            by_form[c["form"]]["selfcorr"] += 1
            by_sign[c["sign"]]["selfcorr"] += 1

    print()
    print(f"7 项 PASS: {seven_pass} 条")
    print(f"self-corr PASS: {selfcorr_pass} 条")
    print()

    def _print_group(
        title: str, d: Dict[str, Dict[str, int]],
    ) -> None:
        print(f"按{title}分组:")
        for key in sorted(d.keys()):
            s = d[key]
            print(
                f"  {key:32s}: submit={s['submit']} / "
                f"7check={s['seven']} / selfcorr={s['selfcorr']}"
            )
        print()

    _print_group("新字段", by_field)
    _print_group("组合形式", by_form)
    _print_group("符号", by_sign)

    db.close()
    print("=" * 70)
    return 0


# ============================================================
# self-corr 检查
# ============================================================

def _check_self_corr(
    settings,
    completed: List[Any],
    combo_by_expr: Dict[str, Dict[str, Any]],
    db: Any,
) -> Dict[str, str]:
    """
    对已完成的组合调 /check 拿 self-corr，并写回 DB。

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
    print("大样本跨族组合实验")
    print("=" * 70)
    print(f"  limit:         {args.limit}")
    print(f"  seed:          {args.seed}")
    print(f"  db:            {args.db}")
    print(f"  experiment_id: {EXPERIMENT_ID}")
    print(f"  max_minutes:   {args.max_minutes}")
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

    # -------- 5. 生成表达式 --------
    rng = random.Random(args.seed)
    combos = _generate_unique_expressions(args.limit, rng)

    if not combos:
        print("❌ 没有生成任何组合", file=sys.stderr)
        try:
            session.close()
        except Exception:
            pass
        return 1

    # -------- 6. 校验 --------
    combos = _validate_combinations(combos)

    if not combos:
        print("❌ 所有组合都被 validate_node 拒绝", file=sys.stderr)
        try:
            session.close()
        except Exception:
            pass
        return 1

    print()
    print(f"最终组合数: {len(combos)}")
    print()

    # -------- 6b. 逐条打印（前 20 条） --------
    for i, c in enumerate(combos[:20], start=1):
        print(
            f"  {i:03d}. [{c['field']} × {c['pv1_label']} / "
            f"{c['form']} / {c['sign']}]"
        )
        print(f"       {c['expression'][:120]}")
    if len(combos) > 20:
        print(f"  ... 还有 {len(combos) - 20} 条")
    print()

    # -------- 6c. 按维度统计 --------
    field_counts: Dict[str, int] = {}
    form_counts: Dict[str, int] = {}
    sign_counts: Dict[str, int] = {}
    for c in combos:
        field_counts[c["field"]] = (
            field_counts.get(c["field"], 0) + 1
        )
        form_counts[c["form"]] = form_counts.get(c["form"], 0) + 1
        sign_counts[c["sign"]] = sign_counts.get(c["sign"], 0) + 1

    print("按新字段统计:")
    for k, v in sorted(field_counts.items()):
        print(f"  {k:32s}: {v} 条")
    print("按组合形式统计:")
    for k, v in sorted(form_counts.items()):
        print(f"  {k:32s}: {v} 条")
    print("按符号统计:")
    for k, v in sorted(sign_counts.items()):
        print(f"  {k:32s}: {v} 条")
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
        rc = _run_submit(
            settings, combos, args.db, args.max_minutes,
        )
    finally:
        try:
            session.close()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
