# scripts/probe_data_fields.py
# -*- coding: utf-8 -*-

"""
Data Fields Discovery 探测脚本（不属于正式工程模块）。

用途
----
    1. 登录 BRAIN
    2. 调用 GET /data-fields（自动分页，limit=50/页）
    3. 打印统计 + 字段样例
    4. 把原始返回写入 data/data_fields.json

用法
----
    cd /Users/vip/Code/Python/worldquant
    export WQ_BRAIN_USERNAME='...'
    export WQ_BRAIN_PASSWORD='...'
    python scripts/probe_data_fields.py

输出
----
    data/data_fields.json   原始 API 返回（所有页合并）
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging  # noqa: E402
from collections import Counter  # noqa: E402

from config.logging import setup_logging  # noqa: E402
from config.settings import AppSettings  # noqa: E402
from brain import (  # noqa: E402
    create_authenticated_session,
    fetch_data_fields,
)


# 探测页数上限（避免一次拉太多）
MAX_PAGES = int(os.environ.get("WQ_PROBE_MAX_PAGES", "40"))


def main() -> None:
    setup_logging(level=logging.INFO)

    settings = AppSettings.from_env()

    print()
    print("=" * 80)
    print("Data Fields Discovery: BRAIN /data-fields")
    print("=" * 80)
    print(f"  region         : {settings.simulation.region}")
    print(f"  universe       : {settings.simulation.universe}")
    print(f"  delay          : {settings.simulation.delay}")
    print(f"  instrumentType : {settings.simulation.instrument_type}")
    print(f"  max_pages      : {MAX_PAGES} (每页 50 条)")
    print("=" * 80)

    session = create_authenticated_session(
        settings.credentials,
        settings.endpoints,
        request_timeout=settings.polling.request_timeout,
    )

    # 拉取（自动分页）
    fields = fetch_data_fields(
        session,
        settings.endpoints,
        settings.polling,
        region=settings.simulation.region,
        delay=settings.simulation.delay,
        universe=settings.simulation.universe,
        instrument_type=settings.simulation.instrument_type,
        max_pages=MAX_PAGES,
    )

    if not fields:
        print()
        print("❌ 获取数据字段失败或为空")
        sys.exit(1)

    # 落盘原始返回
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "data_fields.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 原始返回已写入: {out_path}")

    # ---------- 统计 ----------
    print()
    print("=" * 80)
    print(f"字段总数: {len(fields)}")
    print("=" * 80)

    # 观察字段 schema
    print()
    print("第一个 item 的 keys:")
    print(" ", list(fields[0].keys()))

    print()
    print("第一个 item 完整内容:")
    print(" ", json.dumps(fields[0], ensure_ascii=False, indent=4))

    # 分布统计
    for key in ("type", "fieldType", "dataset", "category"):
        if key in fields[0]:
            c = Counter(str(f.get(key, "?")) for f in fields)
            print()
            print(f"字段 {key!r} 分布 (Top 10):")
            for k, v in c.most_common(10):
                print(f"  {str(k):40s}: {v}")

    # 名字样例
    print()
    print("字段名样例 (前 40 个):")
    for f in fields[:40]:
        name = f.get("name") or f.get("id") or "?"
        ftype = f.get("type") or f.get("fieldType") or "?"
        dataset = f.get("dataset")
        if isinstance(dataset, dict):
            dataset = dataset.get("id", "?")
        print(
            f"  {str(name):45s} "
            f"type={str(ftype):10s} "
            f"dataset={dataset}"
        )

    print()
    print("=" * 80)
    print("完成 — 请把上面输出贴给开发者（用于设计类型映射）")
    print("=" * 80)


if __name__ == "__main__":
    main()