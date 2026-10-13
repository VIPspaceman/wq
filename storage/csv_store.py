# storage/csv_store.py
# -*- coding: utf-8 -*-

"""
CsvAlphaRepository。

Step 10d 更新
-------------
- CSV 新增 `tags` 列（list → 逗号分隔字符串）
- 原子写：先写 <path>.tmp，再 os.replace()
- 编码 utf-8-sig，兼容 Excel
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, List

from evaluation.record import AlphaRecord
from storage.repository import AlphaRepository


# CSV 列（与 V3 save_csv 对齐 + tags 新增）
CSV_FIELDS = [
    "expression",
    "alpha_id",
    "sharpe",
    "fitness",
    "turnover",
    "score",
    "classification",
    "family",
    "complexity",
    "depth",
    "timestamp",
    "tags",          # ★ Step 10d 新增
]


class CsvAlphaRepository(AlphaRepository):
    def __init__(self, path: str) -> None:
        self._path = path

    def save_all(self, records: Iterable[AlphaRecord]) -> None:
        dir_ = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(dir_, exist_ok=True)
        tmp = self._path + ".tmp"

        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in records:
                d = r.to_dict()
                row = {k: d.get(k, "") for k in CSV_FIELDS}
                # tags：list → 逗号分隔字符串
                tags = row.get("tags")
                if isinstance(tags, list):
                    row["tags"] = ",".join(str(t) for t in tags)
                writer.writerow(row)

        os.replace(tmp, self._path)

    def load_all(self) -> List[AlphaRecord]:
        if not os.path.exists(self._path):
            return []
        result: List[AlphaRecord] = []
        with open(self._path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # from_dict 会自动处理 tags 字符串（Step 10d）
                result.append(AlphaRecord.from_dict(dict(row)))
        return result