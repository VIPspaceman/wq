# storage/repository.py
# -*- coding: utf-8 -*-

"""
AlphaRepository 抽象。

职责边界
--------
只定义接口；具体实现见 json_store.py / csv_store.py。

未来可扩展：
    - SqliteAlphaRepository
    - PostgresAlphaRepository
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from evaluation.record import AlphaRecord


class AlphaRepository(ABC):
    @abstractmethod
    def save_all(self, records: Iterable[AlphaRecord]) -> None:
        """整体覆盖式保存。"""

    @abstractmethod
    def load_all(self) -> List[AlphaRecord]:
        """读取全部记录；文件不存在返回空列表。"""