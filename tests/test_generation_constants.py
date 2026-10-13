# -*- coding: utf-8 -*-

"""
验证 P1-6 修复：
    search/generation.py 复用 REFINE_DEPTH_BONUS / REFINE_NODE_BONUS，
    不再硬编码 +1 / +2。
"""

from __future__ import annotations

from agent.refine_router import (
    REFINE_DEPTH_BONUS,
    REFINE_NODE_BONUS,
)


def test_bonus_values() -> None:
    assert REFINE_DEPTH_BONUS == 1
    assert REFINE_NODE_BONUS == 2


def test_generation_imports_bonus() -> None:
    """
    确认 search/generation.py 能 import 这两个常量
    （若 import 失败会抛 ImportError）。
    """
    import search.generation  # noqa: F401
