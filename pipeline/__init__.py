# -*- coding: utf-8 -*-

"""
pipeline：候选过滤流水线。

职责边界
--------
把原先散落在 search/generation.py:run_generation 里的 5 层前置过滤，
统一为可组合的 Filter + Pipeline。

5 层过滤（顺序固定，不可调换）：
    1. HashFilter        — 表达式 hash 去重
    2. ValidateFilter    — validate_node（结构/算子/字段/类型/语义）
    3. SaturationFilter  — BRAIN 死角（纯基本面）
    4. StructureFilter   — 结构签名去重
    5. FieldQuotaFilter  — 单字段配额

设计
----
每个 Filter 实现：
    filter(node, context) -> Optional[str]
        返回 None  = 通过
        返回 str   = 拒绝原因（用于日志）

Pipeline 负责：
    - 按顺序调用 Filter
    - 维护 evaluated_hashes 副作用（拒绝时 add）
    - 维护 structure_seen / field_counts 副作用（通过时 add）
    - 汇总 stats（漏斗日志）

不含：
    - 表达式生成（alpha/）
    - simulation 提交（execution/）
    - 评估（evaluation/）
"""

from pipeline.filters import (
    FieldQuotaFilter,
    Filter,
    FilterContext,
    HashFilter,
    SaturationFilter,
    StructureFilter,
    ValidateFilter,
)
from pipeline.pipeline import Pipeline

__all__ = [
    "Filter",
    "FilterContext",
    "HashFilter",
    "ValidateFilter",
    "SaturationFilter",
    "StructureFilter",
    "FieldQuotaFilter",
    "Pipeline",
]
