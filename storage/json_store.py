# storage/json_store.py
# -*- coding: utf-8 -*-

"""
JsonAlphaRepository。

写入策略
--------
原子写：先写 <path>.tmp，再 os.replace()。
避免崩溃时留下半个文件。
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List

from evaluation.record import AlphaRecord
from storage.repository import AlphaRepository


def atomic_write_json(path: str, data) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class JsonAlphaRepository(AlphaRepository):
    def __init__(self, path: str) -> None:
        self._path = path

    def save_all(self, records: Iterable[AlphaRecord]) -> None:
        data = [r.to_dict() for r in records]
        atomic_write_json(self._path, data)

    def load_all(self) -> List[AlphaRecord]:
        if not os.path.exists(self._path):
            return []
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [AlphaRecord.from_dict(x) for x in data]