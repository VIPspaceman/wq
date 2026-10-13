# -*- coding: utf-8 -*-

"""
Phase 1 失败分析：分析铁种子（self_corr=FAIL）的崩溃模式。

按 parent_structure 分组，输出：
    1. 每组的代表表达式
    2. 每组的共同字段、算子、窗口
    3. 每组包含多少条变体

写到 IRON_PATTERNS.txt。

用法
----
    python tools/analyze_crash_patterns.py
    python tools/analyze_crash_patterns.py --db data/wq.db
    python tools/analyze_crash_patterns.py --output IRON_PATTERNS.txt
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 让脚本在 tools/ 目录下也能 import 项目根下的包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from storage.database import WQDatabase


DEFAULT_DB = "data/wq.db"
DEFAULT_OUTPUT = "IRON_PATTERNS.txt"

# 常见算子名（用于从表达式中提取）
_OPERATOR_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
# 窗口数字（如 ts_delta(close, 5) 中的 5）
_WINDOW_RE = re.compile(r",\s*(\d+)\s*\)")
# 字段名（小写字母开头的标识符，排除算子）
_FIELD_RE = re.compile(r"\b([a-z][a-z0-9_]*)\b")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_crash_patterns",
        description="分析铁种子的崩溃模式",
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _extract_operators(expr: str) -> Set[str]:
    return set(_OPERATOR_RE.findall(expr))


def _extract_windows(expr: str) -> Set[str]:
    return set(_WINDOW_RE.findall(expr))


def _extract_fields(expr: str, operators: Set[str]) -> Set[str]:
    """提取字段名（排除算子名）。"""
    out: Set[str] = set()
    for m in _FIELD_RE.findall(expr):
        if m in operators:
            continue
        out.add(m)
    return out


def _load_iron_rows(db: WQDatabase) -> List[Dict[str, Any]]:
    """从 DB 读所有 iron 种子（含表达式）。"""
    try:
        with db._lock:
            rows = db._conn.execute(
                "SELECT "
                "e.alpha_id, e.parent_structure, x.expression "
                "FROM evaluations e "
                "LEFT JOIN expressions x ON e.expr_hash = x.expr_hash "
                "WHERE e.seed_tier = 'iron'"
            ).fetchall()
    except Exception:
        return []
    return [dict(r) for r in rows]


def _build_report(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("IRON PATTERNS (Phase 1)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"iron 总数: {len(rows)}")

    # 按 parent_structure 分组
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    no_sig: List[Dict[str, Any]] = []
    for r in rows:
        sig = r.get("parent_structure")
        if not sig:
            no_sig.append(r)
            continue
        groups[sig].append(r)

    lines.append(f"去重后结构组数: {len(groups)}")
    lines.append(f"无结构签名条数: {len(no_sig)}")
    lines.append("")

    # 按组大小降序
    sorted_groups = sorted(
        groups.items(), key=lambda kv: len(kv[1]), reverse=True,
    )

    for idx, (sig, members) in enumerate(sorted_groups, start=1):
        lines.append("-" * 60)
        lines.append(f"[组 {idx}] structure={sig[:16]}... 变体数={len(members)}")
        lines.append("")

        # 代表表达式（第一条）
        rep_expr = str(members[0].get("expression") or "")
        lines.append(f"  代表表达式: {rep_expr[:120]}")
        lines.append("")

        # 统计共同字段 / 算子 / 窗口
        field_counter: Counter = Counter()
        op_counter: Counter = Counter()
        win_counter: Counter = Counter()

        for m in members:
            expr = str(m.get("expression") or "")
            ops = _extract_operators(expr)
            fields = _extract_fields(expr, ops)
            wins = _extract_windows(expr)
            for f in fields:
                field_counter[f] += 1
            for o in ops:
                op_counter[o] += 1
            for w in wins:
                win_counter[w] += 1

        n = len(members)

        def _common(counter: Counter) -> List[Tuple[str, int]]:
            return [
                (k, v) for k, v in counter.most_common()
                if v == n
            ]

        common_fields = _common(field_counter)
        common_ops = _common(op_counter)
        common_wins = _common(win_counter)

        lines.append(
            f"  共同字段: "
            f"{', '.join(k for k, _ in common_fields) or '(无)'}"
        )
        lines.append(
            f"  共同算子: "
            f"{', '.join(k for k, _ in common_ops) or '(无)'}"
        )
        lines.append(
            f"  共同窗口: "
            f"{', '.join(k for k, _ in common_wins) or '(无)'}"
        )
        lines.append("")

        # 变体列表（最多 10 条）
        lines.append("  变体（最多 10 条）:")
        for m in members[:10]:
            expr = str(m.get("expression") or "")
            lines.append(f"    - {expr[:100]}")
        if len(members) > 10:
            lines.append(f"    ... 还有 {len(members) - 10} 条")
        lines.append("")

    if no_sig:
        lines.append("-" * 60)
        lines.append(f"[无结构签名] {len(no_sig)} 条")
        for m in no_sig[:10]:
            expr = str(m.get("expression") or "")
            lines.append(f"    - {expr[:100]}")
        lines.append("")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if not os.path.exists(args.db):
        print(f"❌ 数据库不存在: {args.db}", file=sys.stderr)
        return 2

    db = WQDatabase(args.db)
    try:
        rows = _load_iron_rows(db)
    finally:
        db.close()

    if not rows:
        print("没有 iron 种子（seed_tier='iron'）")
        print("提示：先运行 update_seed_tiers() 或 baseline_stats.py")
        return 0

    report = _build_report(rows)
    print(report)

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
