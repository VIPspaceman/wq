# -*- coding: utf-8 -*-

"""
5 个前置过滤器。

行为约束
--------
本模块的判定逻辑 **逐字复制** 自 search/generation.py:run_generation
（Step 4 之前）。不优化、不合并、不改判定条件。

副作用约定
----------
Filter.filter() 本身 **不** 修改 context（纯函数）。
所有副作用（evaluated_hashes.add / structure_seen.add /
field_counts 累加）由 Pipeline 统一负责。

这样做的原因：
    - 原逻辑里，拒绝时 evaluated_hashes.add(h) 分散在 4 个分支
    - 提取为纯 Filter 后，Pipeline 在"拒绝"时统一 add
    - 行为一致（拒绝 → add），位置从 Filter 移到 Pipeline
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Set

from alpha.ast import Node, node_to_expr
from alpha.field_quota import get_main_field
from alpha.saturation import is_brain_saturated
from alpha.validator import validate_node
from core.hashing import expression_hash
from core.signature import structure_signature


# ============================================================
# 上下文
# ============================================================

@dataclass
class FilterContext:
    """
    过滤器共享上下文。

    Attributes
    ----------
    evaluated_hashes : 全局已评估 hash 集合（Pipeline 负责 add）
    structure_seen   : 全局已见结构签名集合（Pipeline 负责 add）
    field_counts     : 本代字段计数（Pipeline 负责累加）
    field_quota      : 单字段配额上限
    max_depth        : validate_node 深度上限
    max_nodes        : validate_node 节点上限
    enable_brain_saturation_filter : 是否启用饱和过滤
    """
    evaluated_hashes: Set[str] = field(default_factory=set)
    structure_seen: Set[str] = field(default_factory=set)
    field_counts: Counter = field(default_factory=Counter)
    field_quota: int = 1
    max_depth: int = 5
    max_nodes: int = 15
    enable_brain_saturation_filter: bool = True


# ============================================================
# Filter 基类
# ============================================================

class Filter:
    """
    过滤器基类。

    子类实现 filter(node, context) -> Optional[str]：
        None = 通过
        str  = 拒绝原因
    """

    #: 用于 stats 字典的 key（如 "hash_skipped"）
    stats_key: str = ""

    def filter(
        self, node: Node, context: FilterContext,
    ) -> Optional[str]:
        raise NotImplementedError


# ============================================================
# 1. HashFilter
# ============================================================

class HashFilter(Filter):
    """
    表达式 hash 去重。

    原逻辑：
        h = expression_hash(expr)
        if h in evaluated_hashes:
            skipped.append(expr)
            stats["hash_skipped"] += 1
            continue

    注意：拒绝时 **不** add（本来就在集合里）。
    """

    stats_key = "hash_skipped"

    def filter(
        self, node: Node, context: FilterContext,
    ) -> Optional[str]:
        expr = node_to_expr(node)
        h = expression_hash(expr)
        if h in context.evaluated_hashes:
            return "hash_duplicate"
        return None


# ============================================================
# 2. ValidateFilter
# ============================================================

class ValidateFilter(Filter):
    """
    validate_node（结构 / 算子 / 字段 / 类型 / 语义）。

    原逻辑：
        ok, _reason = validate_node(node, max_depth=..., max_nodes=...)
        if not ok:
            evaluated_hashes.add(h)
            skipped.append(expr)
            stats["validate_skipped"] += 1
            continue
    """

    stats_key = "validate_skipped"

    def filter(
        self, node: Node, context: FilterContext,
    ) -> Optional[str]:
        ok, reason = validate_node(
            node,
            max_depth=context.max_depth,
            max_nodes=context.max_nodes,
        )
        if not ok:
            return reason or "validate_failed"
        return None


# ============================================================
# 3. SaturationFilter
# ============================================================

class SaturationFilter(Filter):
    """
    BRAIN 死角（纯基本面）过滤。

    原逻辑：
        if enable_brain_saturation_filter and is_brain_saturated(node):
            skipped.append(expr)
            evaluated_hashes.add(h)
            sat_skipped += 1
            stats["saturation_skipped"] += 1
            continue
    """

    stats_key = "saturation_skipped"

    def filter(
        self, node: Node, context: FilterContext,
    ) -> Optional[str]:
        if not context.enable_brain_saturation_filter:
            return None
        if is_brain_saturated(node):
            return "brain_saturated"
        return None


# ============================================================
# 4. StructureFilter
# ============================================================

class StructureFilter(Filter):
    """
    结构签名去重。

    原逻辑：
        sig = structure_signature(node)
        if sig and sig in structure_seen:
            skipped.append(expr)
            evaluated_hashes.add(h)
            stats["structure_skipped"] += 1
            continue

    注意：通过时由 Pipeline 负责 structure_seen.add(sig)。
    """

    stats_key = "structure_skipped"

    def filter(
        self, node: Node, context: FilterContext,
    ) -> Optional[str]:
        sig = structure_signature(node)
        if sig and sig in context.structure_seen:
            return "structure_duplicate"
        return None


# ============================================================
# 5. FieldQuotaFilter
# ============================================================

class FieldQuotaFilter(Filter):
    """
    单字段配额过滤。

    原逻辑：
        main_field = get_main_field(node)
        if main_field and field_counts[main_field] >= field_quota:
            skipped.append(expr)
            evaluated_hashes.add(h)
            field_skipped += 1
            stats["field_quota_skipped"] += 1
            continue

    注意：通过时由 Pipeline 负责 field_counts[main_field] += 1。
    """

    stats_key = "field_quota_skipped"

    def filter(
        self, node: Node, context: FilterContext,
    ) -> Optional[str]:
        main_field = get_main_field(node)
        if main_field and context.field_counts[main_field] >= context.field_quota:
            return f"field_quota:{main_field}"
        return None
