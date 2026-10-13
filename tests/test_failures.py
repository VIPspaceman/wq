# -*- coding: utf-8 -*-

"""
agent/failures.py 单元测试。

覆盖：
    - can_attempt / attempts / record 基本语义
    - 熔断上限
    - outcomes 上限（P2 修复验证）
    - reset / clear / size
"""

from __future__ import annotations

import pytest

from agent.failures import (
    RefineFailureTracker,
    RefineOutcome,
)


# ============================================================
# 基本语义
# ============================================================

def test_can_attempt_initially_true() -> None:
    t = RefineFailureTracker(max_attempts=3)
    assert t.can_attempt("rank(close)")
    assert t.attempts("rank(close)") == 0


def test_record_increments_attempts() -> None:
    t = RefineFailureTracker(max_attempts=3)
    t.record("rank(close)", RefineOutcome.SUCCESS)
    assert t.attempts("rank(close)") == 1
    t.record("rank(close)", RefineOutcome.SUCCESS)
    assert t.attempts("rank(close)") == 2


def test_exhausted_after_max() -> None:
    t = RefineFailureTracker(max_attempts=2)
    t.record("rank(close)", RefineOutcome.SUCCESS)
    t.record("rank(close)", RefineOutcome.SUCCESS)
    assert not t.can_attempt("rank(close)")


def test_different_expressions_independent() -> None:
    t = RefineFailureTracker(max_attempts=1)
    t.record("rank(close)", RefineOutcome.SUCCESS)
    assert not t.can_attempt("rank(close)")
    assert t.can_attempt("rank(volume)")


def test_invalid_max_attempts() -> None:
    with pytest.raises(ValueError):
        RefineFailureTracker(max_attempts=0)


# ============================================================
# outcomes 上限（P2 修复验证）
# ============================================================

def test_outcomes_bounded() -> None:
    """
    outcomes 列表不应无限增长。
    """
    t = RefineFailureTracker(max_attempts=1000)
    for _ in range(100):
        t.record("rank(close)", RefineOutcome.SUCCESS)

    # 内部记录
    h = None
    from utils.hashing import expression_hash
    h = expression_hash("rank(close)")
    rec = t._records[h]
    assert len(rec.outcomes) <= 20
    assert rec.attempts == 100


# ============================================================
# reset / clear / size
# ============================================================

def test_reset() -> None:
    t = RefineFailureTracker(max_attempts=3)
    t.record("rank(close)", RefineOutcome.SUCCESS)
    t.reset("rank(close)")
    assert t.attempts("rank(close)") == 0


def test_clear() -> None:
    t = RefineFailureTracker(max_attempts=3)
    t.record("rank(close)", RefineOutcome.SUCCESS)
    t.record("rank(volume)", RefineOutcome.SUCCESS)
    assert t.size() == 2
    t.clear()
    assert t.size() == 0


def test_size() -> None:
    t = RefineFailureTracker(max_attempts=3)
    assert t.size() == 0
    t.record("rank(close)", RefineOutcome.SUCCESS)
    assert t.size() == 1
