# -*- coding: utf-8 -*-

"""
Phase 1 基线统计。

输出纯文本报告到 BASELINE_REPORT.txt：
    1. 金 / 银 / 铁各多少条
    2. 铁种子按 parent_structure 去重后有多少组
    3. 当前系统整体：总 sim 数、7 项 PASS 数、self-corr PASS 数
    4. 三者的比率（7项通过率、self-corr 通过率、有效产出率）

用法
----
    python tools/baseline_stats.py
    python tools/baseline_stats.py --db data/wq.db
    python tools/baseline_stats.py --output BASELINE_REPORT.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让脚本在 tools/ 目录下也能 import 项目根下的包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from storage.database import WQDatabase


DEFAULT_DB = "data/wq.db"
DEFAULT_OUTPUT = "BASELINE_REPORT.txt"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="baseline_stats",
        description="Phase 1 基线统计",
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _pct(num: int, den: int) -> str:
    if den <= 0:
        return "N/A"
    return f"{num / den * 100.0:.2f}%"


def _build_report(stats: Dict[str, Any]) -> str:
    total = int(stats.get("total_evaluations", 0))
    seven = int(stats.get("seven_pass", 0))
    sc_pass = int(stats.get("self_corr_pass", 0))
    gold = int(stats.get("gold", 0))
    silver = int(stats.get("silver", 0))
    iron = int(stats.get("iron", 0))
    iron_uniq = int(stats.get("iron_unique_structures", 0))

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("BASELINE REPORT (Phase 1)")
    lines.append("=" * 60)
    lines.append("")

    lines.append("[1] 种子分层")
    lines.append(f"  gold   (self_corr=PASS) : {gold}")
    lines.append(f"  silver (self_corr=NULL) : {silver}")
    lines.append(f"  iron   (self_corr=FAIL) : {iron}")
    lines.append("")

    lines.append("[2] 铁种子结构去重")
    lines.append(f"  iron 总数              : {iron}")
    lines.append(f"  iron 去重后结构组数    : {iron_uniq}")
    if iron > 0:
        lines.append(
            f"  平均每组变体数         : "
            f"{iron / iron_uniq:.2f}" if iron_uniq > 0 else "  N/A"
        )
    lines.append("")

    lines.append("[3] 系统整体")
    lines.append(f"  总 sim 数              : {total}")
    lines.append(f"  7 项 PASS 数           : {seven}")
    lines.append(f"  self-corr PASS 数      : {sc_pass}")
    lines.append("")

    lines.append("[4] 比率")
    lines.append(f"  7 项通过率             : {_pct(seven, total)}")
    lines.append(f"  self-corr 通过率       : {_pct(sc_pass, seven)}")
    lines.append(f"  有效产出率             : {_pct(sc_pass, total)}")
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if not os.path.exists(args.db):
        print(f"❌ 数据库不存在: {args.db}", file=sys.stderr)
        return 2

    db = WQDatabase(args.db)
    try:
        stats = db.get_seed_stats()
    finally:
        db.close()

    report = _build_report(stats)
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
