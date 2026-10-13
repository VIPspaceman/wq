# -*- coding: utf-8 -*-

"""
search/combiner.py 单元测试。

重点验证 P2 修复：
    _try_combine 返回 Optional[Task] 而非 List[Task]。
"""

from __future__ import annotations

from evaluation.classifier import Classification
from evaluation.record import AlphaRecord
from search.combiner import build_combine_tasks, _try_combine


def _mk_record(
    expr: str,
    sharpe: float = 0.8,
    score: float = 1.0,
    family: str = "OTHER",
    tags=None,
) -> AlphaRecord:
    return AlphaRecord(
        expression=expr,
        alpha_id="A1",
        sharpe=sharpe,
        fitness=0.5,
        turnover=0.5,
        score=score,
        classification=Classification.CANDIDATE,
        family=family,
        complexity=3,
        depth=2,
        tags=list(tags or []),
    )


# ============================================================
# _try_combine 返回类型
# ============================================================

def test_try_combine_returns_task() -> None:
    records = [
        _mk_record("rank(ts_delta(close,5))", family="MOMENTUM"),
        _mk_record("rank(ts_std_dev(returns,10))", family="VOLATILITY"),
    ]
    task = _try_combine(
        records, set(), 10, 30, label="test",
    )
    assert task is not None
    assert task.source == "COMBINED"
    assert task.expression.startswith("rank(")


def test_try_combine_returns_none_on_dup() -> None:
    records = [
        _mk_record("rank(ts_delta(close,5))", family="MOMENTUM"),
        _mk_record("rank(ts_std_dev(returns,10))", family="VOLATILITY"),
    ]
    hashes: set = set()
    task1 = _try_combine(records, hashes, 10, 30, label="t1")
    assert task1 is not None

    # 第二次应因 hash 已存在返回 None
    task2 = _try_combine(records, hashes, 10, 30, label="t2")
    assert task2 is None


def test_try_combine_returns_none_on_too_few() -> None:
    records = [_mk_record("rank(close)")]
    task = _try_combine(records, set(), 10, 30, label="test")
    assert task is None


# ============================================================
# build_combine_tasks
# ============================================================

def test_build_combine_basic() -> None:
    records = [
        _mk_record("rank(ts_delta(close,5))", family="MOMENTUM"),
        _mk_record("rank(ts_std_dev(returns,10))", family="VOLATILITY"),
        _mk_record("rank(volume/ts_mean(volume,20))", family="VOLUME"),
    ]
    tasks = build_combine_tasks(
        records, set(), max_combines=1, combine_size=3,
    )
    assert len(tasks) == 1
    assert tasks[0].source == "COMBINED"


def test_build_combine_not_enough() -> None:
    records = [
        _mk_record("rank(ts_delta(close,5))", family="MOMENTUM"),
    ]
    tasks = build_combine_tasks(
        records, set(), max_combines=1, combine_size=3,
    )
    assert tasks == []


def test_build_combine_skip_low_sharpe() -> None:
    records = [
        _mk_record("rank(ts_delta(close,5))", sharpe=0.1, family="MOMENTUM"),
        _mk_record("rank(ts_std_dev(returns,10))", sharpe=0.1, family="VOLATILITY"),
        _mk_record("rank(volume/ts_mean(volume,20))", sharpe=0.1, family="VOLUME"),
    ]
    tasks = build_combine_tasks(
        records, set(), max_combines=1, combine_size=3,
    )
    assert tasks == []


def test_build_combine_skip_already_combined() -> None:
    records = [
        _mk_record("rank(ts_delta(close,5))", family="MOMENTUM",
                   tags=["COMBINED"]),
        _mk_record("rank(ts_std_dev(returns,10))", family="VOLATILITY"),
        _mk_record("rank(volume/ts_mean(volume,20))", family="VOLUME"),
    ]
    tasks = build_combine_tasks(
        records, set(), max_combines=1, combine_size=3,
    )
    # 只剩 2 个候选，不足 combine_size=3
    assert tasks == []


# ============================================================
# 轮转起点（P2 修复验证）
# ============================================================

def test_build_combine_rotation_diversity() -> None:
    """
    多次尝试应从不同起点开始，生成不同的组合。
    """
    records = [
        _mk_record(f"rank(ts_delta(close,{i}))", family=f"F{i}")
        for i in range(3, 9)
    ]
    tasks = build_combine_tasks(
        records, set(), max_combines=3, combine_size=3,
    )
    # 至少生成 2 个不同的组合
    exprs = {t.expression for t in tasks}
    assert len(exprs) >= 2
