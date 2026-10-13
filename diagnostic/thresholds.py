# diagnostic/thresholds.py
# -*- coding: utf-8 -*-

"""
诊断阈值。

⚠️ 重要说明
-----------
本模块的阈值是 **诊断层专用**，与 `config/thresholds.py`
（内部评分/分层用）是 **两套独立的东西**：

    config/thresholds.py
        └─ CANDIDATE / STRONG / FINAL 三层分级
        └─ 本项目内部 heuristic

    diagnostic/thresholds.py
        └─ PASS / CANDIDATE / REJECT 判定
        └─ 触发 AI 修复的判定
        └─ 依据：BRAIN 官方 checks 的 limit 值

Step 10-Y 更新
--------------
新增 `negative_candidate_sharpe`：
    - 当 sharpe 为负且 |sharpe| >= 该值时，也触发 CANDIDATE
    - 目的：让"信号方向错了但结构可能有效"的 alpha 进入 AI 修复
    - LLM 会收到"取反"提示

触发规则（新）
--------------
    正方向：  0.80 <= sharpe < 1.25   →  CANDIDATE
    负方向：  sharpe <= -0.50          →  CANDIDATE（无上限）
    PASS：    sharpe >= 1.25 且 fitness >= 1.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticThreshold:
    """
    Attributes
    ----------
    pass_sharpe                  : PASS 判定下限
    pass_fitness                 : PASS 判定下限
    candidate_sharpe             : 正方向 CANDIDATE 下限
    negative_candidate_sharpe    : 负方向 CANDIDATE 下限（|sharpe| 门槛）
    high_turnover                : Turnover 上限
    low_fitness_warn             : Fitness "偏低"告警线
    """
    pass_sharpe: float = 1.25
    pass_fitness: float = 1.0
    candidate_sharpe: float = 0.80
    negative_candidate_sharpe: float = 0.50
    high_turnover: float = 0.70
    low_fitness_warn: float = 1.0
    # FIX-6b：turnover 下限（低于此值视为异常，不通过）
    min_turnover: float = 0.01


# 默认阈值
DEFAULT_THRESHOLD = DiagnosticThreshold()


# 保守模式（更少触发修复）
CONSERVATIVE_THRESHOLD = DiagnosticThreshold(
    candidate_sharpe=1.00,
    negative_candidate_sharpe=0.80,
)


# 激进模式（更多触发修复）
AGGRESSIVE_THRESHOLD = DiagnosticThreshold(
    candidate_sharpe=0.60,
    negative_candidate_sharpe=0.30,
)
