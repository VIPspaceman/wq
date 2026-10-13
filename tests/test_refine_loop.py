# -*- coding: utf-8 -*-

"""
search/refine_loop.py 单元测试。

重点验证 P1-7 修复：
    RefineOutcomeSummary.failed → backtest_failed 改名。
"""

from __future__ import annotations

from search.refine_loop import RefineOutcomeSummary


def test_summary_default_fields() -> None:
    s = RefineOutcomeSummary()
    assert s.submitted == 0
    assert s.succeeded == 0
    assert s.backtest_failed == 0
    assert s.exhausted == 0
    assert s.llm_failed == 0


def test_summary_no_old_failed_field() -> None:
    """
    确认旧字段名 `failed` 已不存在（避免误用）。
    """
    s = RefineOutcomeSummary()
    assert not hasattr(s, "failed")


def test_summary_backtest_failed_increment() -> None:
    s = RefineOutcomeSummary()
    s.backtest_failed += 1
    assert s.backtest_failed == 1
