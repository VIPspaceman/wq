# scripts/probe_data_fields.py
# -*- coding: utf-8 -*-

"""
Data Fields Discovery 探测脚本。

用法
----
    # 拉全平台（默认 dataset 不限，最多 2000 条）
    python scripts/probe_data_fields.py

    # 只拉 PV 数据集（推荐！快速、字段少）
    WQ_PROBE_DATASET=pv1 python scripts/probe_data_fields.py

    # 拉基本面数据
    WQ_PROBE_DATASET=fundamental6 python scripts/probe_data_fields.py

    # 拉更多页
    WQ_PROBE_MAX_PAGES=100 WQ_PROBE_DATASET=pv1 python scripts/probe_data_fields.py

输出
----
    data/data_fields.json   原始返回（所有页合并）
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


MAX_PAGES = int(os.environ.get("WQ_PROBE_MAX_PAGES", "40"))
DATASET_ID = os.environ.get("WQ_PROBE_DATASET", "").strip() or None


def main() -> None:
    setup_logging(level=logging.INFO)
    settings = AppSettings.from_env()

    print()
    print("=" * 80)
    print("Data Fields Discovery")
    print("=" * 80)
    print(f"  region    : {settings.simulation.region}")
    print(f"  universe  : {settings.simulation.universe}")
    print(f"  delay     : {settings.simulation.delay}")
    print(f"  dataset   : {DATASET_ID or '(不限)'}")
    print(f"  max_pages : {MAX_PAGES}")
    print("=" * 80)

    session = create_authenticated_session(
        settings.credentials,
        settings.endpoints,
        request_timeout=settings.polling.request_timeout,
    )

    fields = fetch_data_fields(
        session,
        settings.endpoints,
        settings.polling,
        region=settings.simulation.region,
        delay=settings.simulation.delay,
        universe=settings.simulation.universe,
        instrument_type=settings.simulation.instrument_type,
        dataset_id=DATASET_ID,
        max_pages=MAX_PAGES,
    )

    if not fields:
        print("\n❌ 获取失败或为空")
        sys.exit(1)

    # 落盘
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "data_fields.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 原始返回已写入: {out_path}")

    # 统计
    print()
    print("=" * 80)
    print(f"字段总数: {len(fields)}")
    print("=" * 80)

    # type 分布
    type_counter = Counter(
        str(f.get("type", "?")).upper() for f in fields
    )
    print()
    print("type 分布:")
    for k, v in type_counter.most_common():
        print(f"  {k:15s}: {v}")

    # dataset 分布
    def _ds_id(f):
        ds = f.get("dataset")
        if isinstance(ds, dict):
            return ds.get("id", "?")
        return str(ds or "?")

    ds_counter = Counter(_ds_id(f) for f in fields)
    print()
    print("dataset 分布 (Top 10):")
    for k, v in ds_counter.most_common(10):
        print(f"  {k:30s}: {v}")

    # MATRIX 字段样例
    matrix = [f for f in fields if str(f.get("type", "")).upper() == "MATRIX"]
    print()
    print(f"MATRIX 字段样例 (前 30 个, 共 {len(matrix)}):")
    for f in matrix[:30]:
        fid = f.get("id", "?")
        cov = f.get("coverage", 0)
        ds = _ds_id(f)
        print(f"  {fid:50s} cov={cov:.2f} ds={ds}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()