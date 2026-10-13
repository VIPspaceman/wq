# agent/__init__.py
# -*- coding: utf-8 -*-

"""
agent：LLM 驱动的 Alpha 重构层。

子模块：
    prompts        : 动态 prompt 构建（从 catalog 实时读取）
    cleaner        : LLM 输出清洗
    failures       : 熔断计数器
    refine_agent   : LLMRefiningAgent 主类
    refine_router  : 规则路由器（Step A）
"""

from agent.failures import (
    RefineFailureTracker,
    RefineOutcome,
)
from agent.refine_agent import LLMRefiningAgent, RefineResult
from agent.refine_router import (
    REFINE_DEPTH_BONUS,
    REFINE_NODE_BONUS,
    RuleRefineResult,
    try_rule_refine,
)

__all__ = [
    "LLMRefiningAgent",
    "RefineResult",
    "RefineFailureTracker",
    "RefineOutcome",
    # Step A
    "try_rule_refine",
    "RuleRefineResult",
    "REFINE_DEPTH_BONUS",
    "REFINE_NODE_BONUS",
]