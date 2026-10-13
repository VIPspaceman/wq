# -*- coding: utf-8 -*-

"""
diagnostic/diagnoser.py 单元测试。

重点验证 P0-2 修复：
    负方向 CANDIDATE 不再自动加 LOW_FITNESS。
"""

from __future__ import annotations

import pytest

from diagnostic.diagnoser import diagnose
from diagnostic.types import DefectType


def _mk(sharpe: float, fitness: float = 0.5, turnover: float = 0.5):
    return {
        "is": {
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
        }
    }


# ============================================================
# 正方向
# ============================================================

@pytest.mark.parametrize("sharpe", [0.80, 0.95, 1.24])
def test_positive_candidate(sharpe: float) -> None:
    r = diagnose(_mk(sharpe), "rank(ts_delta(close,5))")
    assert r.is_candidate
    assert r.direction_hint == "positive"
    assert DefectType.LOW_SHARPE in r.defects


@pytest.mark.parametrize("sharpe", [0.10, 0.50, 0.79])
def test_positive_too_low(sharpe: float) -> None:
    r = diagnose(_mk(sharpe), "rank(ts_delta(close,5))")
    assert not r.is_candidate


# ============================================================
# 负方向
# ============================================================

@pytest.mark.parametrize("sharpe", [-0.50, -0.88, -1.11, -2.50])
def test_negative_candidate(sharpe: float) -> None:
    r = diagnose(_mk(sharpe), "rank(ts_delta(close,5))")
    assert r.is_candidate
    assert r.direction_hint == "negative"
    assert DefectType.LOW_SHARPE in r.defects


@pytest.mark.parametrize("sharpe", [-0.10, -0.30, -0.49])
def test_negative_too_small(sharpe: float) -> None:
    r = diagnose(_mk(sharpe), "rank(ts_delta(close,5))")
    assert not r.is_candidate


# ============================================================
# P0-2 修复验证：负方向不自动加 LOW_FITNESS
# ============================================================

def test_negative_no_auto_low_fitness() -> None:
    """
    负 sharpe 时 fitness 通常也为负（数学相关），
    不应自动标记 LOW_FITNESS。
    """
    r = diagnose(
        _mk(-0.88, fitness=-0.5),
        "rank(ts_delta(close,5))",
    )
    assert r.is_candidate
    assert r.direction_hint == "negative"
    assert DefectType.LOW_SHARPE in r.defects
    # ★ 关键断言：不应自动加 LOW_FITNESS
    assert DefectType.LOW_FITNESS not in r.defects


def test_negative_extreme_low_fitness_marked() -> None:
    """
    仅当 fitness 显著低于 -pass_fitness 时才标记 LOW_FITNESS。
    """
    r = diagnose(
        _mk(-0.88, fitness=-1.5),
        "rank(ts_delta(close,5))",
    )
    assert r.is_candidate
    assert DefectType.LOW_FITNESS in r.defects


# ============================================================
# PASS
# ============================================================

def test_pass_unaffected() -> None:
    r = diagnose(
        _mk(1.50, fitness=1.20, turnover=0.30),
        "rank(ts_delta(close,5))",
    )
    assert r.is_pass
    assert not r.is_candidate
    assert r.direction_hint == ""
    assert not r.needs_refine


# ============================================================
# WARNING / ERROR
# ============================================================

def test_warning_units_error() -> None:
    r = diagnose(
        {
            "status": "WARNING",
            "message": 'Incompatible unit for input of "sqrt"',
        },
        "sqrt(close)",
    )
    assert r.is_reject
    assert DefectType.UNITS_ERROR in r.defects
