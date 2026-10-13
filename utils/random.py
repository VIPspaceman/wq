# utils/random.py
# -*- coding: utf-8 -*-

"""
随机数种子管理。

职责边界
--------
只负责：设置 / 查询随机种子（用于实验可复现）。

不含：
    - 随机数业务逻辑
    - 表达式的随机生成（在 alpha/generator.py）
"""

from __future__ import annotations

import random as _random
from typing import Any


def set_seed(seed: int) -> None:
    """设置全局随机种子（影响 alpha/generator 与 evolution/*）。"""
    _random.seed(seed)


def get_seed() -> Any:
    """
    读取当前全局 seed 状态（用于日志/实验记录）。

    注意：返回的是 random.getstate() 的内容，
    仅用于记录，不用于恢复流程。
    """
    return _random.getstate()
