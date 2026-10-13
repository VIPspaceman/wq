# -*- coding: utf-8 -*-

"""
analyst4 字段审计：拉取 analyst4 全字段并按语义分类。

流程
----
    1. 登录 BRAIN
    2. 用 brain.capabilities.fetch_data_fields 拉 analyst4 全部字段
       （dataset_id='analyst4', max_pages=1）
    3. 每个字段输出 id / description / coverage
    4. 按字段名关键词分类：
       ESTIMATE / ACTUAL / REVISION / DISPERSION / SURPRISE / OTHER
    5. 输出到 ANALYST4_FIELDS.txt

用法
----
    python tools/analyze_analyst4_fields.py
    python tools/analyze_analyst4_fields.py --output ANALYST4_FIELDS.txt

约束
----
- 只读，不提交 sim
- 只新建本文件 + 输出 txt
- 不改其他文件
"""

from __future__ import annotations

import argparse
import logging
import sys
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

DEFAULT_OUTPUT = "ANALYST4_FIELDS.txt"
DATASET_ID = "analyst4"

# 分类关键词（小写匹配）
_CATEGORY_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("ESTIMATE", ("estimate", "forecast", "expected")),
    ("ACTUAL", ("actual",)),
    ("REVISION", ("revision", "change", "upgrade", "downgrade")),
    ("DISPERSION", ("dispersion", "std", "spread")),
    ("SURPRISE", ("surprise",)),
)

CATEGORY_ORDER = (
    "ESTIMATE",
    "ACTUAL",
    "REVISION",
    "DISPERSION",
    "SURPRISE",
    "OTHER",
)


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_analyst4_fields",
        description="拉取 analyst4 全字段并按语义分类",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"输出路径（默认 {DEFAULT_OUTPUT}）",
    )
    return parser.parse_args(argv)


# ============================================================
# 登录 + 拉字段
# ============================================================

def _login(settings):
    """登录 BRAIN，返回已认证 Session。"""
    from brain.authentication import create_authenticated_session

    return create_authenticated_session(
        settings.credentials,
        settings.endpoints,
        request_timeout=settings.polling.request_timeout,
    )


def _fetch_analyst4_fields(settings, session) -> List[Dict[str, Any]]:
    """拉取 analyst4 全部字段（max_pages=1）。"""
    from brain.capabilities import fetch_data_fields

    return fetch_data_fields(
        session,
        settings.endpoints,
        settings.polling,
        region=settings.simulation.region,
        delay=settings.simulation.delay,
        universe=settings.simulation.universe,
        instrument_type=settings.simulation.instrument_type,
        dataset_id=DATASET_ID,
        max_pages=1,
    )


# ============================================================
# 分类
# ============================================================

def _classify(field_id: str) -> str:
    """按字段名关键词分类。"""
    n = (field_id or "").lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(kw in n for kw in keywords):
            return category
    return "OTHER"


def _extract_id(item: Dict[str, Any]) -> str:
    for key in ("id", "name", "field"):
        v = item.get(key)
        if v:
            return str(v).strip()
    return ""


def _extract_description(item: Dict[str, Any]) -> str:
    return str(item.get("description", "") or "")


def _extract_coverage(item: Dict[str, Any]) -> Optional[float]:
    v = item.get("coverage")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ============================================================
# 报告生成
# ============================================================

def _build_report(
    fields: List[Dict[str, Any]],
) -> str:
    """生成审计报告。"""
    # ---- 分类 ----
    buckets: Dict[str, List[Dict[str, Any]]] = {
        c: [] for c in CATEGORY_ORDER
    }
    for item in fields:
        fid = _extract_id(item)
        if not fid:
            continue
        cat = _classify(fid)
        buckets[cat].append(item)

    lines: List[str] = []
    lines.append("=" * 40)
    lines.append("analyst4 字段审计")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"总字段数: {len(fields)}")
    lines.append("")

    # ---- 分类明细 ----
    for cat in CATEGORY_ORDER:
        items = buckets[cat]
        lines.append(f"【{cat} 类】({len(items)} 个)")
        for item in items:
            fid = _extract_id(item)
            desc = _extract_description(item)
            cov = _extract_coverage(item)

            line = f"- {fid}"
            extras: List[str] = []
            if desc:
                extras.append(f"desc={desc[:80]}")
            if cov is not None:
                extras.append(f"coverage={cov:.4f}")
            if extras:
                line += "  [" + " | ".join(extras) + "]"
            lines.append(line)
        lines.append("")

    # ---- 高价值字段推荐 ----
    lines.append("=" * 40)
    lines.append("【高价值字段推荐】")
    lines.append("基于分类，推荐以下 5 个字段用于首日验证：")
    lines.append("")

    recommendations = _pick_recommendations(buckets)
    if not recommendations:
        lines.append("(无可用字段)")
    else:
        for i, (fid, cat, reason) in enumerate(
            recommendations, start=1,
        ):
            lines.append(f"{i}. {fid}  [{cat}]  {reason}")
    lines.append("")

    return "\n".join(lines)


def _pick_recommendations(
    buckets: Dict[str, List[Dict[str, Any]]],
) -> List[Tuple[str, str, str]]:
    """
    从分类结果里挑 5 个高价值字段。

    优先级：ESTIMATE > SURPRISE > REVISION > DISPERSION > ACTUAL
    每个分类最多取 2 个，凑满 5 个为止。
    """
    priority = (
        ("ESTIMATE", "分析师预期，最直接的预测信号"),
        ("SURPRISE", "实际 vs 预期差异，事件驱动信号"),
        ("REVISION", "预期修正，动量类信号"),
        ("DISPERSION", "分析师分歧度，不确定性信号"),
        ("ACTUAL", "实际值，作为基准/对照"),
    )

    out: List[Tuple[str, str, str]] = []
    for cat, reason in priority:
        items = buckets.get(cat) or []
        for item in items[:2]:
            fid = _extract_id(item)
            if not fid:
                continue
            out.append((fid, cat, reason))
            if len(out) >= 5:
                return out
    return out


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
    print("analyst4 字段审计")
    print("=" * 70)
    print(f"  dataset_id: {DATASET_ID}")
    print(f"  output:     {args.output}")
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

    # -------- 3. 拉字段 --------
    try:
        fields = _fetch_analyst4_fields(settings, session)
    except Exception as e:
        print(f"❌ 拉取 analyst4 字段失败: {e}", file=sys.stderr)
        try:
            session.close()
        except Exception:
            pass
        return 2
    finally:
        try:
            session.close()
        except Exception:
            pass

    if not fields:
        print("❌ analyst4 字段为空（或 API 返回失败）")
        return 1

    print(f"获取 {len(fields)} 个字段")

    # -------- 4. 生成报告 --------
    report = _build_report(fields)
    print()
    print(report)

    # -------- 5. 写文件 --------
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
