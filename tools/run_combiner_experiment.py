# -*- coding: utf-8 -*-

"""
跨家族两两组合实验。

目标
----
从 gold + silver 种子里两两配对，生成复合结构，提交 sim，
测 self-corr 是否比单结构低。

种子池
------
    seed_tier IN ('gold', 'silver')
    - gold   : 已提交成功
    - silver : 7 项 PASS，self-corr 未查

组合规则
--------
1. 两两配对（不做三元）
2. 强制跨 family：
   - 提取每条种子的"主字段集"
   - 两条种子的主字段集不能有交集
   - 例：close 系 × volume 系 ✓；close 系 × close 系 ✗
3. 组合形式（按权重采样）：
   - rank(A) + (-1*rank(B))   权重 0.7
   - rank(A) * rank(B)        权重 0.3
4. 去重：
   - expression_hash 去重
   - structure_signature 去重

流程
----
    1. 加载远程算子池
    2. 从 DB 拉 gold + silver 种子（含 expression）
    3. 生成 N 个唯一组合（按规则）
    4. 本地 validate_node 过滤
    5. 提交 sim（复用 executor）
    6. 对 7 项 PASS 的调 /check 拿 self-corr
    7. 结果写 DB（experiment_id="exp_combiner_test1"）
    8. 打印统计

用法
----
    python tools/run_combiner_experiment.py
    python tools/run_combiner_experiment.py --limit 30
    python tools/run_combiner_experiment.py --dry-run
    python tools/run_combiner_experiment.py --experiment-id exp_combiner_test1

约束
----
- 只新建本文件
- 复用现有 executor + DB 接口
- 整脚本最多 30 分钟（超时保护）
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
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
# 2026-10-12 修正：批量生成 100 条候选，输出 30-50 条合格
DEFAULT_LIMIT = 100
DEFAULT_MAX_OUTPUT = 50
DEFAULT_SEED = 20261012
DEFAULT_EXPERIMENT_ID = "exp_combiner_test1"

# 整脚本超时（秒）
TOTAL_TIMEOUT_SECONDS = 30 * 60

# self-corr 检查
SELF_CORR_SLEEP = 5.0
SELF_CORR_MAX = 50

# 组合形式
FORM_ADD = "ADD"      # rank(A) + (-1*rank(B))
FORM_MUL = "MUL"      # rank(A) * rank(B)

# 权重（按顺序采样）
#
# 2026-10-12 修正
# ----------------
# 上一轮 ADD 形式全部质量下降（sharpe 掉），MUL 是唯一 7-PASS。
# 因此默认只用 MUL；ADD 保留但权重降到 0.2。
FORM_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    (FORM_MUL, 0.8),
    (FORM_ADD, 0.2),
)

# 字段族判定（用于跨 family 强约束）
_PRICE_FIELDS = frozenset({
    "open", "high", "low", "close", "vwap", "price",
})
_VOLUME_FIELDS = frozenset({
    "volume", "adv20", "adv", "dollar_volume",
})
_RETURN_FIELDS = frozenset({
    "returns", "ret",
})
_FUNDAMENTAL_FIELDS = frozenset({
    "assets", "capex", "cash", "cashflow", "cashflow_op",
    "cogs", "current_ratio", "debt", "ebit", "ebitda",
    "employee", "enterprise_value", "eps", "equity",
    "bookvalue_ps",
})

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
# 字段提取（正则，不依赖 parse_expression）
# ============================================================
#
# 背景
# ----
# parse_expression 用本地 hardcoded catalog（只有 pv1 6 个字段 +
# fundamental6 15 个字段），DB 里的表达式若含其它字段会 parse 失败。
# 因此这里用正则提取"主字段集"。

# 算子名（用于从字段候选中排除）
_OPERATOR_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

# 字段名（小写字母开头的标识符）
_FIELD_RE = re.compile(r"(?<![A-Za-z0-9_])([a-z][a-z0-9_]*)(?![A-Za-z0-9_])")

# 已知算子名（用于排除）
_KNOWN_OPERATORS = frozenset({
    "rank", "zscore", "scale", "quantile", "normalize",
    "winsorize", "densify", "abs", "log", "sqrt", "inverse",
    "reverse", "signed_power", "sign",
    "ts_delta", "ts_mean", "ts_std_dev", "ts_sum", "ts_rank",
    "ts_av_diff", "ts_decay_linear", "ts_product", "ts_scale",
    "ts_backfill", "ts_delay", "ts_zscore", "ts_quantile",
    "ts_arg_max", "ts_arg_min", "ts_count_nans", "ts_step",
    "ts_corr", "ts_cov",
    "group_neutralize", "group_rank", "group_zscore",
    "if_else", "trade_when", "bucket",
})


def _extract_fields(expr: str) -> Set[str]:
    """
    从表达式里提取字段名（排除算子名）。

    用词边界匹配，避免 'cash' 匹配 'cashflow'。
    """
    out: Set[str] = set()
    for m in _FIELD_RE.findall(expr or ""):
        if m in _KNOWN_OPERATORS:
            continue
        # 排除纯数字（正则已保证首字符是字母，这里再保险）
        if m.isdigit():
            continue
        out.add(m)
    return out


def _main_field_set(expr: str) -> Set[str]:
    """
    提取"主字段集"：表达式里出现的所有字段名。

    用于跨 family 判定：两条种子的主字段集不能有交集。
    """
    return _extract_fields(expr)


def _field_family(name: str) -> str:
    """判定字段属于哪个族。"""
    n = (name or "").lower()
    if n in _PRICE_FIELDS:
        return "price"
    if n in _VOLUME_FIELDS:
        return "volume"
    if n in _RETURN_FIELDS:
        return "returns"
    if n in _FUNDAMENTAL_FIELDS:
        return "fundamental"
    return "other"


def _field_families(fields: Set[str]) -> Set[str]:
    """把字段集映射成字段族集合。"""
    return {_field_family(f) for f in fields}


def _family_key(fields: Set[str]) -> str:
    """
    把字段族集合转成排序后的字符串（用于统计）。

    例：{"price", "volume"} → "price×volume"
    """
    fams = sorted(_field_families(fields))
    return "×".join(fams) if fams else "(none)"


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_combiner_experiment",
        description="跨家族两两组合实验",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"生成 N 条组合（默认 {DEFAULT_LIMIT}）",
    )
    parser.add_argument(
        "--max-output", type=int, default=50,
        help="最多输出多少条合格候选（默认 50）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只生成不提交",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"随机种子（默认 {DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"数据库路径（默认 {DEFAULT_DB}）",
    )
    parser.add_argument(
        "--experiment-id", default=None,
        help=(
            "实验 ID（默认 "
            f"{DEFAULT_EXPERIMENT_ID}）"
        ),
    )
    return parser.parse_args(argv)


# ============================================================
# 远程算子池
# ============================================================

def _load_remote_catalog() -> bool:
    """
    加载远程算子池（合并进本地 catalog）。

    必须在解析种子之前调用，否则含远程算子的种子会解析失败。
    失败时返回 False（不中断，降级用本地 catalog）。
    """
    try:
        from config.settings import AppSettings
        from brain.authentication import create_authenticated_session
        from alpha.operators import load_from_api
    except ImportError:
        logger.warning("无法导入远程 catalog 依赖，使用本地 catalog")
        return False

    try:
        settings = AppSettings.from_env()
    except Exception as e:
        logger.warning("加载配置失败，使用本地 catalog: %s", e)
        return False

    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
    except Exception as e:
        logger.warning("BRAIN 登录失败，使用本地 catalog: %s", e)
        return False

    try:
        catalog = load_from_api(
            session,
            settings.endpoints,
            settings.polling,
        )
    except Exception as e:
        logger.warning("加载远程算子池异常，使用本地 catalog: %s", e)
        return False

    if catalog is None:
        logger.warning("远程算子池为空，使用本地 catalog")
        return False

    logger.info("远程算子池已加载: %s", catalog.summary())

    # 关闭临时 session（load_from_api 内部已用完）
    try:
        session.close()
    except Exception:
        pass

    return True


# ============================================================
# 种子加载
# ============================================================

def _load_seeds(
    db_path: str,
    *,
    tiers: Tuple[str, ...] = ("silver",),
) -> List[Dict[str, Any]]:
    """
    从 DB 拉指定 tier 的种子。

    2026-10-12 修正
    ----------------
    默认只加载 silver（31 条）。
    gold 已在 BRAIN 库里，用作分量会把 self-corr 下限锁死
    （上一轮 e7QYo1mN 用 gold #2 作分量，self-corr 被锁在 0.80）。
    因此 gold 只作为"参考规格"，不参与组合。

    返回每项：
        {
            "alpha_id": str,
            "expression": str,
            "seed_tier": str,
            "fields": set[str],
            "families": set[str],
        }
    """
    import sqlite3

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    if not tiers:
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in tiers)
        rows = conn.execute(
            "SELECT "
            "e.alpha_id, x.expression, e.seed_tier "
            "FROM evaluations e "
            "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
            f"WHERE e.seed_tier IN ({placeholders}) "
            "AND x.expression IS NOT NULL "
            "AND x.expression != '' "
            "ORDER BY e.sharpe DESC",
            list(tiers),
        ).fetchall()
    finally:
        conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        expr = str(r["expression"] or "")
        if not expr:
            continue
        fields = _main_field_set(expr)
        if not fields:
            continue
        out.append({
            "alpha_id": str(r["alpha_id"] or ""),
            "expression": expr,
            "seed_tier": str(r["seed_tier"] or ""),
            "fields": fields,
            "families": _field_families(fields),
        })
    return out


# ============================================================
# 组合生成
# ============================================================

def _pick_form(rng: random.Random) -> str:
    """按权重采样组合形式。"""
    r = rng.random()
    acc = 0.0
    for form, w in FORM_WEIGHTS:
        acc += w
        if r <= acc:
            return form
    return FORM_WEIGHTS[-1][0]


def _build_combined_expr(
    expr_a: str,
    expr_b: str,
    form: str,
) -> str:
    """
    构造组合表达式。

    ADD : rank(A) + (-1*rank(B))
    MUL : rank(A) * rank(B)
    """
    if form == FORM_ADD:
        return f"(rank({expr_a})+(-1*rank({expr_b})))"
    if form == FORM_MUL:
        return f"(rank({expr_a})*rank({expr_b}))"
    raise ValueError(f"Unknown form: {form}")


def _generate_combinations(
    seeds: List[Dict[str, Any]],
    *,
    limit: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    生成组合。

    规则（2026-10-12 修正）：
        - 两两配对（不做三元）
        - 强制跨 family：两分量字段族不能相同
          （用字段族判定，比"字段集无交集"更宽松，
           允许 price×volume 这类跨族组合）
        - 按权重采样组合形式（默认 MUL 0.8 / ADD 0.2）
        - expression_hash + structure_signature 双重去重

    返回每项：
        {
            "expression": str,
            "form": str,
            "seed_a": dict,
            "seed_b": dict,
            "expr_hash": str,
            "structure_sig": str,
            "family_key": str,   # 如 "price×volume"
        }
    """
    from core.hashing import expression_hash
    from core.signature import structure_signature
    from alpha.parser import parse_expression

    if len(seeds) < 2:
        logger.warning("种子数 < 2，无法组合")
        return []

    # ---- 1. 枚举所有跨 family 配对 ----
    # 强约束：两分量字段族不能相同
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    n = len(seeds)
    for i in range(n):
        for j in range(i + 1, n):
            a = seeds[i]
            b = seeds[j]
            fam_a = a.get("families") or _field_families(a["fields"])
            fam_b = b.get("families") or _field_families(b["fields"])
            # 字段族完全相同 → 跳过（如 price×price）
            if fam_a == fam_b:
                continue
            # 字段集有交集 → 跳过（避免同字段重复）
            if a["fields"] & b["fields"]:
                continue
            pairs.append((a, b))

    if not pairs:
        logger.warning("没有跨 family 的配对")
        return []

    logger.info(
        "跨 family 配对数: %d（种子 %d 条）", len(pairs), n,
    )

    # ---- 2. 打乱配对顺序 ----
    rng.shuffle(pairs)

    # ---- 3. 生成组合 ----
    out: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()
    seen_sigs: Set[str] = set()

    for a, b in pairs:
        if len(out) >= limit:
            break

        # 随机决定 A/B 顺序（避免总是 a 在前）
        if rng.random() < 0.5:
            a, b = b, a

        # 采样形式
        form = _pick_form(rng)

        expr = _build_combined_expr(
            a["expression"], b["expression"], form,
        )

        # ---- expression_hash 去重 ----
        h = expression_hash(expr)
        if h in seen_hashes:
            continue

        # ---- 解析 + structure_signature 去重 ----
        node = parse_expression(expr)
        if node is None:
            logger.debug("解析失败，跳过: %s", expr[:80])
            continue

        sig = structure_signature(node)
        if sig and sig in seen_sigs:
            continue

        seen_hashes.add(h)
        if sig:
            seen_sigs.add(sig)

        # ---- 字段族组合 key ----
        fam_a = a.get("families") or _field_families(a["fields"])
        fam_b = b.get("families") or _field_families(b["fields"])
        family_key = "×".join(sorted(fam_a | fam_b))

        out.append({
            "expression": expr,
            "form": form,
            "seed_a": a,
            "seed_b": b,
            "expr_hash": h,
            "structure_sig": sig,
            "family_key": family_key,
        })

    logger.info(
        "生成 %d 条唯一组合（去重后）", len(out),
    )
    return out


# ============================================================
# 本地校验
# ============================================================

# 组合表达式嵌套两个种子的完整 AST（每个种子本身含
# winsorize(ts_backfill(...)) 包装），深度/节点数必然超过
# validate_node 的默认限制（5/15）。因此这里放宽到 30/60。
COMBINE_MAX_DEPTH = 30
COMBINE_MAX_NODES = 60


def _validate_combinations(
    combos: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    用 validate_node 过滤组合。

    解析失败 / 校验失败 → 剔除。

    注意：组合表达式嵌套两个种子的完整 AST，
    深度/节点数必然超过默认限制（5/15），
    因此这里放宽到 COMBINE_MAX_DEPTH / COMBINE_MAX_NODES。
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
    experiment_id: str,
    db_path: str,
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

    # -------- 1. 打开 DB --------
    try:
        db = WQDatabase(db_path)
        db.init_schema()
        device_id = db.get_or_create_device("combiner-tool")
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
        t = Task(expression=c["expression"], source="COMBINED")
        tasks.append(t)
        combo_by_expr[c["expression"]] = c

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
        c = combo_by_expr.get(task.expression)
        if c is None or task.result is None:
            continue
        try:
            node = parse_expression(task.expression)
            if node is None:
                continue
            record = build_record(node, task.result)

            # ---- 打 tags ----
            record.add_tag("COMBINED")
            record.add_tag("combiner_pair")
            record.add_tag(f"combiner_form:{c['form']}")
            if c["seed_a"].get("alpha_id"):
                record.add_tag(
                    f"combiner_a:{c['seed_a']['alpha_id']}"
                )
            if c["seed_b"].get("alpha_id"):
                record.add_tag(
                    f"combiner_b:{c['seed_b']['alpha_id']}"
                )

            # ---- 父种子结构签名（用 seed_a 的）----
            parent_sig = None
            try:
                pnode = parse_expression(
                    c["seed_a"]["expression"]
                )
                if pnode is not None:
                    parent_sig = structure_signature(pnode)
            except Exception:
                parent_sig = None

            ok_db = db.save_record(
                record=record.to_dict(),
                experiment_id=experiment_id,
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
    print("Combiner Experiment Results")
    print("=" * 70)

    for task in tasks:
        c = combo_by_expr.get(task.expression)
        if c is None:
            continue
        print()
        print(
            f"=== [{c['form']}] "
            f"{c['seed_a']['alpha_id']} × "
            f"{c['seed_b']['alpha_id']} ==="
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
    by_form: Dict[str, Dict[str, int]] = {
        FORM_ADD: {"submit": 0, "seven": 0, "selfcorr": 0},
        FORM_MUL: {"submit": 0, "seven": 0, "selfcorr": 0},
    }
    by_family: Dict[str, Dict[str, int]] = {}

    for task in tasks:
        c = combo_by_expr.get(task.expression)
        if c is None:
            continue
        form = c["form"]
        if form not in by_form:
            by_form[form] = {"submit": 0, "seven": 0, "selfcorr": 0}
        by_form[form]["submit"] += 1

        fam_key = c.get("family_key") or "(unknown)"
        if fam_key not in by_family:
            by_family[fam_key] = {
                "submit": 0, "seven": 0, "selfcorr": 0,
            }
        by_family[fam_key]["submit"] += 1

        if task.result:
            checks = _extract_checks_map(task.result)
            ok, _ = _seven_check_status(checks)
            if ok:
                seven_pass += 1
                by_form[form]["seven"] += 1
                by_family[fam_key]["seven"] += 1

        sc = self_corr_stats.get(task.expression)
        if sc == "PASS":
            selfcorr_pass += 1
            by_form[form]["selfcorr"] += 1
            by_family[fam_key]["selfcorr"] += 1

    print()
    print(f"7 项 PASS: {seven_pass} 条")
    print(f"self-corr PASS: {selfcorr_pass} 条")
    print()
    print("按组合形式分组:")
    for form in sorted(by_form.keys()):
        s = by_form[form]
        print(
            f"  {form:6s}: submit={s['submit']} / "
            f"7check={s['seven']} / selfcorr={s['selfcorr']}"
        )

    if by_family:
        print()
        print("按字段族组合分组:")
        for fam_key in sorted(
            by_family.keys(),
            key=lambda k: by_family[k]["submit"],
            reverse=True,
        ):
            s = by_family[fam_key]
            print(
                f"  {fam_key:24s}: submit={s['submit']} / "
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

    experiment_id = args.experiment_id or DEFAULT_EXPERIMENT_ID

    print()
    print("=" * 70)
    print("跨家族两两组合实验")
    print("=" * 70)
    print(f"  limit:         {args.limit}")
    print(f"  seed:          {args.seed}")
    print(f"  db:            {args.db}")
    print(f"  experiment_id: {experiment_id}")
    print(f"  模式:          {'dry-run' if args.dry_run else '提交'}")
    print()

    # -------- 1. 加载远程算子池 --------
    _load_remote_catalog()

    # -------- 2. 从 DB 拉种子（只用 silver） --------
    try:
        seeds = _load_seeds(args.db, tiers=("silver",))
    except Exception as e:
        print(f"❌ 读取种子失败: {e}", file=sys.stderr)
        return 2

    if len(seeds) < 2:
        print(
            f"❌ silver 种子数不足（{len(seeds)} 条），无法组合",
            file=sys.stderr,
        )
        return 1

    print(f"加载 silver 种子: {len(seeds)} 条")

    # -------- 2b. 打印 gold 参考规格（不参与组合） --------
    try:
        gold_seeds = _load_seeds(args.db, tiers=("gold",))
    except Exception:
        gold_seeds = []

    if gold_seeds:
        print()
        print("Gold 参考规格（不参与组合）:")
        for g in gold_seeds:
            fam_key = _family_key(g["fields"])
            print(
                f"  {g['alpha_id']:10s} [{fam_key}] "
                f"{g['expression'][:80]}"
            )
        print()

    # -------- 3. 生成组合 --------
    rng = random.Random(args.seed)
    combos = _generate_combinations(
        seeds, limit=args.limit, rng=rng,
    )

    if not combos:
        print("❌ 没有生成任何组合", file=sys.stderr)
        return 1

    # -------- 4. 本地校验 --------
    combos = _validate_combinations(combos)

    if not combos:
        print("❌ 所有组合都被 validate_node 拒绝", file=sys.stderr)
        return 1

    # -------- 4b. 截断到 max-output --------
    if len(combos) > args.max_output:
        print(
            f"合格候选 {len(combos)} 条，截断到 "
            f"{args.max_output} 条"
        )
        combos = combos[: args.max_output]

    print()
    print(f"最终组合数: {len(combos)}")
    print()

    # -------- 4c. 逐条打印（含字段族） --------
    for i, c in enumerate(combos, start=1):
        fam_a = c["seed_a"].get("families") or _field_families(
            c["seed_a"]["fields"]
        )
        fam_b = c["seed_b"].get("families") or _field_families(
            c["seed_b"]["fields"]
        )
        fam_a_str = "×".join(sorted(fam_a)) or "(none)"
        fam_b_str = "×".join(sorted(fam_b)) or "(none)"
        print(
            f"  {i:02d}. [{c['form']}] "
            f"{c['seed_a']['alpha_id']} [{fam_a_str}] × "
            f"{c['seed_b']['alpha_id']} [{fam_b_str}]"
        )
        print(f"      {c['expression'][:120]}")

    # -------- 4d. 按字段族组合统计 --------
    fam_counts: Dict[str, int] = {}
    for c in combos:
        key = c.get("family_key") or "(unknown)"
        fam_counts[key] = fam_counts.get(key, 0) + 1
    if fam_counts:
        print()
        print("按字段族组合统计:")
        for k, v in sorted(
            fam_counts.items(), key=lambda kv: kv[1], reverse=True,
        ):
            print(f"  {k}: {v} 条")
    print()

    # -------- 5. dry-run --------
    if args.dry_run:
        print("(dry-run) 未提交。")
        return 0

    # -------- 6. 提交 --------
    try:
        from config.settings import AppSettings
        settings = AppSettings.from_env()
    except Exception as e:
        print(f"❌ 加载配置失败: {e}", file=sys.stderr)
        return 2

    return _run_submit(
        settings, combos, experiment_id, args.db,
    )


if __name__ == "__main__":
    sys.exit(main())
