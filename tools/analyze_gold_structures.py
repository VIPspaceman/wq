# -*- coding: utf-8 -*-

"""
分析 5 条 gold 的复合结构，提取组合设计规格。

输入
----
5 条 gold（从 DB 的 evaluations + expressions 表拉）：
    GrOxlAbQ / j28Arpej / wpbZnVmY / O086Z7pp / 1YZPE8A6

分析内容
--------
对每条 gold：
    1. 解析表达式为 AST
    2. 识别顶层结构：
       - 是单结构 还是 复合结构（A + B / A * B / A - B）
       - 如果是复合：A 侧和 B 侧分别是什么
    3. 统计每个 gold 的：
       - 分量数（1 / 2 / 3）
       - 组合形式（ADD / MUL / 其他）
       - 每个分量的字段族（price / volume / returns / fundamental）
       - 每个分量的外层算子（rank / zscore / quantile / scale）
       - 每个分量的 sharpe（如果能拆出来）
    4. 提取共同特征

输出
----
GOLD_STRUCTURE_ANALYSIS.txt

用法
----
    python tools/analyze_gold_structures.py
    python tools/analyze_gold_structures.py --db data/wq.db
    python tools/analyze_gold_structures.py --output GOLD_STRUCTURE_ANALYSIS.txt

约束
----
- 只读 DB
- 只新建本文件 + 输出 txt
- 不改其他文件
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
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
DEFAULT_OUTPUT = "GOLD_STRUCTURE_ANALYSIS.txt"

# 5 条 gold 的 alpha_id（按用户给定顺序）
GOLD_ALPHA_IDS: Tuple[str, ...] = (
    "GrOxlAbQ",
    "j28Arpej",
    "wpbZnVmY",
    "O086Z7pp",
    "1YZPE8A6",
)

# 字段族判定
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

# 外层算子（用于识别分量外层）
_OUTER_OPS = frozenset({
    "rank", "zscore", "scale", "quantile", "normalize",
    "winsorize", "densify",
})

# 组合形式
FORM_ADD = "ADD"
FORM_SUB = "SUB"
FORM_MUL = "MUL"
FORM_DIV = "DIV"
FORM_SINGLE = "SINGLE"
FORM_OTHER = "OTHER"


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_gold_structures",
        description="分析 5 条 gold 的复合结构",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"数据库路径（默认 {DEFAULT_DB}）",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"输出路径（默认 {DEFAULT_OUTPUT}）",
    )
    return parser.parse_args(argv)


# ============================================================
# 远程算子池
# ============================================================

def _load_remote_catalog() -> bool:
    """
    加载远程算子池（合并进本地 catalog）。

    必须在解析 gold 表达式之前调用，否则含远程算子
    （ts_av_diff / ts_decay_linear 等）的表达式会解析失败。
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
            session, settings.endpoints, settings.polling,
        )
    except Exception as e:
        logger.warning("加载远程算子池异常，使用本地 catalog: %s", e)
        return False

    if catalog is None:
        logger.warning("远程算子池为空，使用本地 catalog")
        return False

    logger.info("远程算子池已加载: %s", catalog.summary())
    try:
        session.close()
    except Exception:
        pass
    return True


# ============================================================
# DB 读取
# ============================================================

def _load_gold_rows(
    db_path: str, alpha_ids: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    """
    从 DB 拉指定 alpha_id 的 gold 记录。

    返回每项：
        {
            "alpha_id": str,
            "expression": str,
            "sharpe": float,
            "fitness": float,
            "turnover": float,
            "seed_tier": str,
        }
    """
    import sqlite3

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in alpha_ids)
        sql = (
            "SELECT "
            "e.alpha_id, x.expression, "
            "e.sharpe, e.fitness, e.turnover, e.seed_tier "
            "FROM evaluations e "
            "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
            f"WHERE e.alpha_id IN ({placeholders})"
        )
        rows = conn.execute(sql, list(alpha_ids)).fetchall()
    finally:
        conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "alpha_id": str(r["alpha_id"] or ""),
            "expression": str(r["expression"] or ""),
            "sharpe": _as_float(r["sharpe"]),
            "fitness": _as_float(r["fitness"]),
            "turnover": _as_float(r["turnover"]),
            "seed_tier": str(r["seed_tier"] or ""),
        })
    return out


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# 字段族判定
# ============================================================

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


def _collect_fields(node: Any) -> Set[str]:
    """递归收集 AST 里所有 field 名。"""
    if node is None:
        return set()
    out: Set[str] = set()
    if node.kind == "field":
        out.add(node.value)
    elif node.kind in ("unary", "ts"):
        out |= _collect_fields(node.child)
    elif node.kind in ("ts_binary", "binary"):
        out |= _collect_fields(node.left)
        out |= _collect_fields(node.right)
    return out


def _field_families(node: Any) -> Set[str]:
    """收集 AST 里所有字段的族。"""
    return {_field_family(f) for f in _collect_fields(node)}


# ============================================================
# 顶层结构识别
# ============================================================

def _strip_outer_wrappers(node: Any) -> Any:
    """
    剥掉最外层的 rank / zscore / scale / quantile / winsorize 等
    截面算子，返回内层节点。

    用于识别"顶层组合形式"。
    """
    cur = node
    while (
        cur is not None
        and cur.kind == "unary"
        and cur.value in _OUTER_OPS
        and cur.child is not None
    ):
        cur = cur.child
    return cur


def _identify_top_structure(node: Any) -> Dict[str, Any]:
    """
    识别顶层结构。

    返回：
        {
            "form": "ADD" / "SUB" / "MUL" / "DIV" / "SINGLE" / "OTHER",
            "outer_op": str | None,   # 最外层截面算子
            "left": Node | None,
            "right": Node | None,
            "components": [Node, ...],  # 分量列表
        }
    """
    if node is None:
        return {
            "form": FORM_OTHER,
            "outer_op": None,
            "left": None,
            "right": None,
            "components": [],
        }

    # 记录最外层截面算子
    outer_op = None
    if node.kind == "unary" and node.value in _OUTER_OPS:
        outer_op = node.value

    # 剥掉外层截面算子
    inner = _strip_outer_wrappers(node)

    if inner is None:
        return {
            "form": FORM_OTHER,
            "outer_op": outer_op,
            "left": None,
            "right": None,
            "components": [],
        }

    # 单结构
    if inner.kind != "binary":
        return {
            "form": FORM_SINGLE,
            "outer_op": outer_op,
            "left": None,
            "right": None,
            "components": [inner],
        }

    # 复合结构
    op = inner.value
    form_map = {
        "+": FORM_ADD,
        "-": FORM_SUB,
        "*": FORM_MUL,
        "/": FORM_DIV,
    }
    form = form_map.get(op, FORM_OTHER)

    # 收集分量（递归展开同 op 的链式组合）
    components = _flatten_binary(inner, op)

    return {
        "form": form,
        "outer_op": outer_op,
        "left": inner.left,
        "right": inner.right,
        "components": components,
    }


def _flatten_binary(node: Any, op: str) -> List[Any]:
    """
    展开同 op 的链式二元组合。

    例：((A+B)+C) 且 op="+" → [A, B, C]
    """
    if node is None:
        return []
    if node.kind == "binary" and node.value == op:
        return (
            _flatten_binary(node.left, op)
            + _flatten_binary(node.right, op)
        )
    return [node]


# ============================================================
# 分量分析
# ============================================================

def _analyze_component(node: Any) -> Dict[str, Any]:
    """
    分析单个分量。

    返回：
        {
            "expr": str,
            "outer_op": str | None,
            "field_families": set[str],
            "fields": set[str],
            "is_negated": bool,   # 是否被 -1* 包裹
        }
    """
    from alpha.ast import node_to_expr

    if node is None:
        return {
            "expr": "",
            "outer_op": None,
            "field_families": set(),
            "fields": set(),
            "is_negated": False,
        }

    # 检测 -1* 包裹
    is_negated = False
    cur = node
    if (
        cur.kind == "binary"
        and cur.value == "*"
        and cur.left is not None
        and cur.left.kind == "constant"
        and cur.left.value in ("-1", "-1.0")
    ):
        is_negated = True
        cur = cur.right

    # 外层算子
    outer_op = None
    if cur is not None and cur.kind == "unary" and cur.value in _OUTER_OPS:
        outer_op = cur.value

    try:
        expr = node_to_expr(node)
    except Exception:
        expr = ""

    return {
        "expr": expr,
        "outer_op": outer_op,
        "field_families": _field_families(node),
        "fields": _collect_fields(node),
        "is_negated": is_negated,
    }


# ============================================================
# 报告生成
# ============================================================

def _build_report(
    gold_rows: List[Dict[str, Any]],
) -> str:
    """生成分析报告。"""
    from alpha.parser import parse_expression

    lines: List[str] = []
    lines.append("=" * 40)
    lines.append("5 条 Gold 的复合结构分析")
    lines.append("=" * 40)
    lines.append("")

    # ---- 逐条分析 ----
    lines.append("【逐条分析】")
    lines.append("")

    analyses: List[Dict[str, Any]] = []

    for idx, row in enumerate(gold_rows, start=1):
        aid = row["alpha_id"]
        expr = row["expression"]

        lines.append(f"Gold {idx}: {aid}")
        lines.append(f"  表达式: {expr}")

        node = parse_expression(expr)
        if node is None:
            lines.append("  ⚠️ 解析失败（可能含未加载的算子/字段）")
            lines.append("")
            analyses.append({
                "alpha_id": aid,
                "parse_ok": False,
                "form": FORM_OTHER,
                "components": [],
            })
            continue

        struct = _identify_top_structure(node)
        form = struct["form"]
        outer_op = struct["outer_op"]
        components = struct["components"]

        # 顶层描述
        if form == FORM_SINGLE:
            lines.append("  顶层: 单结构（无复合）")
        else:
            lines.append(f"  顶层: {form} 复合")

        if outer_op:
            lines.append(f"  最外层截面算子: {outer_op}")

        # 分量分析
        comp_analyses: List[Dict[str, Any]] = []
        for ci, comp in enumerate(components, start=1):
            ca = _analyze_component(comp)
            comp_analyses.append(ca)

            label = chr(ord("A") + ci - 1)  # A / B / C
            fam_str = " × ".join(sorted(ca["field_families"])) or "(无)"
            neg_str = " [取负]" if ca["is_negated"] else ""
            op_str = ca["outer_op"] or "(无外层)"

            lines.append(
                f"  {label} 侧: {ca['expr']}"
            )
            lines.append(
                f"      [字段族: {fam_str}] "
                f"[外层: {op_str}]{neg_str}"
            )

        # 组合形式描述
        if form == FORM_ADD:
            lines.append("  组合形式: rank(A + (-1)*B)")
        elif form == FORM_MUL:
            lines.append("  组合形式: rank(A) * rank(B)")
        elif form == FORM_SUB:
            lines.append("  组合形式: A - B")
        elif form == FORM_DIV:
            lines.append("  组合形式: A / B")
        elif form == FORM_SINGLE:
            lines.append("  组合形式: 单结构")
        else:
            lines.append(f"  组合形式: {form}")

        # 字段族组合
        all_fams: Set[str] = set()
        for ca in comp_analyses:
            all_fams |= ca["field_families"]
        lines.append(
            f"  字段族: {' × '.join(sorted(all_fams)) or '(无)'}"
        )

        # 指标
        lines.append(
            f"  指标: sharpe={row['sharpe']:+.4f} "
            f"fitness={row['fitness']:+.4f} "
            f"turnover={row['turnover']:.4f}"
        )
        lines.append("")

        analyses.append({
            "alpha_id": aid,
            "parse_ok": True,
            "form": form,
            "outer_op": outer_op,
            "components": comp_analyses,
            "field_families": all_fams,
            "sharpe": row["sharpe"],
        })

    # ---- 共同特征 ----
    lines.append("【共同特征】")
    lines.append("")

    parsed = [a for a in analyses if a.get("parse_ok")]
    total = len(analyses)

    # 复合 vs 单结构
    compound = [a for a in parsed if a["form"] != FORM_SINGLE]
    single = [a for a in parsed if a["form"] == FORM_SINGLE]
    lines.append(
        f"- {len(compound)}/{total} 是复合结构"
        f"（{len(single)}/{total} 是单结构）"
    )

    # 组合形式分布
    form_counts: Dict[str, int] = {}
    for a in compound:
        form_counts[a["form"]] = form_counts.get(a["form"], 0) + 1
    if form_counts:
        form_str = ", ".join(
            f"{k}={v}" for k, v in sorted(form_counts.items())
        )
        lines.append(f"- 复合形式分布: {form_str}")

    # 分量数分布
    comp_counts: Dict[int, int] = {}
    for a in compound:
        n = len(a["components"])
        comp_counts[n] = comp_counts.get(n, 0) + 1
    if comp_counts:
        comp_str = ", ".join(
            f"{k}分量={v}" for k, v in sorted(comp_counts.items())
        )
        lines.append(f"- 分量数分布: {comp_str}")

    # 字段族组合
    fam_pairs: Dict[str, int] = {}
    for a in compound:
        fams = sorted(a.get("field_families", set()))
        if len(fams) >= 2:
            key = " × ".join(fams)
            fam_pairs[key] = fam_pairs.get(key, 0) + 1
    if fam_pairs:
        lines.append("- 分量字段族组合:")
        for k, v in sorted(
            fam_pairs.items(), key=lambda kv: kv[1], reverse=True,
        ):
            lines.append(f"    {k}: {v} 条")

    # 外层算子
    outer_counts: Dict[str, int] = {}
    for a in parsed:
        for ca in a.get("components", []):
            op = ca.get("outer_op")
            if op:
                outer_counts[op] = outer_counts.get(op, 0) + 1
    if outer_counts:
        outer_str = ", ".join(
            f"{k}={v}" for k, v in sorted(
                outer_counts.items(), key=lambda kv: kv[1], reverse=True,
            )
        )
        lines.append(f"- 分量外层算子: {outer_str}")

    lines.append("")

    # ---- 组合设计规则 ----
    lines.append("【组合设计规则】")
    lines.append("")
    lines.append("1. 分量数量：2（不做三元）")
    lines.append(
        "2. 分量字段族：必须不同"
        "（price × volume / price × fundamental）"
    )
    lines.append("3. 分量外层算子：至少一个用 rank / quantile")
    lines.append(
        "4. 组合形式：ADD（rank(A + (-1)*rank(B))）"
        "或 MUL（rank(A) * rank(B)）"
    )
    lines.append(
        "5. **禁止使用已提交的 gold 作为分量**"
        "（它们已在 BRAIN 库里）"
    )
    lines.append("6. 分量 standalone sharpe 必须 ≥ 1.2")
    lines.append("")

    return "\n".join(lines)


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
    print("分析 5 条 gold 的复合结构")
    print("=" * 70)
    print(f"  db:     {args.db}")
    print(f"  output: {args.output}")
    print()

    # -------- 1. 加载远程算子池 --------
    _load_remote_catalog()

    # -------- 2. 从 DB 拉 gold --------
    try:
        gold_rows = _load_gold_rows(args.db, GOLD_ALPHA_IDS)
    except Exception as e:
        print(f"❌ 读取 DB 失败: {e}", file=sys.stderr)
        return 2

    if not gold_rows:
        print("❌ DB 里找不到任何指定的 gold alpha_id", file=sys.stderr)
        return 1

    found_ids = {r["alpha_id"] for r in gold_rows}
    missing = [aid for aid in GOLD_ALPHA_IDS if aid not in found_ids]
    if missing:
        print(f"⚠️ DB 里找不到: {', '.join(missing)}")

    print(f"加载 {len(gold_rows)} 条 gold")

    # -------- 3. 生成报告 --------
    report = _build_report(gold_rows)
    print()
    print(report)

    # -------- 4. 写文件 --------
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已写入: {args.output}")
    except Exception as e:
        print(f"❌ 写报告失败: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
