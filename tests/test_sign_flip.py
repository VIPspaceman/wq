# -*- coding: utf-8 -*-

"""
search/sign_flip.py 单元测试。

重点验证 P0-5 修复：
    带空格的 (-1 * x) 也能被识别为"已取反"。
"""

from __future__ import annotations

import pytest

from evaluation.classifier import Classification
from evaluation.record import AlphaRecord
from search.sign_flip import build_sign_flip_tasks


def _mk_record(
    expr: str,
    sharpe: float,
    classification: Classification = Classification.NEGATIVE,
) -> AlphaRecord:
    return AlphaRecord(
        expression=expr,
        alpha_id="A1",
        sharpe=sharpe,
        fitness=0.5,
        turnover=0.5,
        score=0.0,
        classification=classification,
        family="OTHER",
        complexity=3,
        depth=2,
    )


# ============================================================
# 基本生成
# ============================================================

def test_basic_flip() -> None:
    records = [_mk_record("rank(ts_delta(close,5))", -0.88)]
    hashes: set = set()
    tasks = build_sign_flip_tasks(records, hashes)
    assert len(tasks) == 1
    assert tasks[0].expression == "(-1*rank(ts_delta(close,5)))"
    assert tasks[0].source == "SIGN_FLIPPED"


def test_skip_positive() -> None:
    records = [_mk_record(
        "rank(ts_delta(close,5))", 0.88,
        classification=Classification.CANDIDATE,
    )]
    tasks = build_sign_flip_tasks(records, set())
    assert tasks == []


def test_skip_sharpe_too_high() -> None:
    records = [_mk_record("rank(ts_delta(close,5))", -0.3)]
    tasks = build_sign_flip_tasks(records, set())
    assert tasks == []


# ============================================================
# P0-5 修复验证：带空格的 (-1 * x) 也应被跳过
# ============================================================

@pytest.mark.parametrize("expr", [
    "(-1*rank(ts_delta(close,5)))",
    "(-1 * rank(ts_delta(close,5)))",
    "(-1  *  rank(ts_delta(close,5)))",
])
def test_skip_already_flipped(expr: str) -> None:
    """
    已取反的表达式（无论有无空格）都应被跳过，避免双取反。
    """
    records = [_mk_record(expr, -0.88)]
    tasks = build_sign_flip_tasks(records, set())
    assert tasks == []


# ============================================================
# 去重
# ============================================================

def test_dedup_by_hash() -> None:
    records = [_mk_record("rank(ts_delta(close,5))", -0.88)]
    hashes: set = set()
    tasks1 = build_sign_flip_tasks(records, hashes)
    assert len(tasks1) == 1

    # 第二次调用应因 hash 已存在而跳过
    tasks2 = build_sign_flip_tasks(records, hashes)
    assert tasks2 == []


# ============================================================
# max_flips 上限
# ============================================================

def test_max_flips_limit() -> None:
    records = [
        _mk_record(f"rank(ts_delta(close,{i}))", -0.9 - i * 0.01)
        for i in range(3, 10)
    ]
    tasks = build_sign_flip_tasks(records, set(), max_flips=2)
    assert len(tasks) == 2
