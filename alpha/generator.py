# alpha/generator.py
# -*- coding: utf-8 -*-

"""
Alpha 生成（初始 / 语法驱动 / 智能模式）。

职责边界
--------
对齐 V3：
    - combine_nodes
    - generate_grammar_alpha
    - generate_smart_alpha
    - generate_initial_alpha

**方案 B 改动**
----------------
GRAMMAR_FAMILIES 的生成器现在可能返回 None
（当 catalog 里没有池成员时）。
generate_grammar_alpha / generate_smart_alpha 都加了 None 检查。

不含：
    - 变异 / 交叉（evolution/）
    - 种群管理（evolution/population.py）
    - 校验（validator.py 提供）
"""

from __future__ import annotations

import random
from typing import Optional

from alpha.ast import Node, binary, unary
from alpha.families import (
    GRAMMAR_FAMILIES,
    mean_reversion,
    price_momentum,
    price_momentum_ratio,
    price_vs_mean,
    volume_activity,
    volume_trend,
    volatility_signal,
)
from alpha.validator import validate_node


# ============================================================
# 跨家族组合
# ============================================================

def combine_nodes(a: Node, b: Node) -> Optional[Node]:
    op = random.choice(["*", "+", "-"])
    candidate = binary(op, a, b)

    if random.random() < 0.75:
        candidate = unary("rank", candidate)

    ok, _ = validate_node(candidate)
    if not ok:
        return None
    return candidate


# ============================================================
# 语法驱动生成
# ============================================================

def generate_grammar_alpha() -> Optional[Node]:
    family_names = list(GRAMMAR_FAMILIES.keys())
    family_a = random.choice(family_names)
    generator_a = random.choice(GRAMMAR_FAMILIES[family_a])
    a = generator_a()

    # 方案 B：随机版可能返回 None
    if a is None:
        return None

    # 60% 单家族
    if random.random() < 0.60:
        ok, _ = validate_node(a)
        if ok:
            a.family = family_a
            return a
        return None

    # 40% 跨家族组合
    family_b = random.choice([x for x in family_names if x != family_a])
    generator_b = random.choice(GRAMMAR_FAMILIES[family_b])
    b = generator_b()

    if b is None:
        return None

    candidate = combine_nodes(a, b)
    if candidate is None:
        return None

    candidate.family = f"{family_a}+{family_b}"
    return candidate


# ============================================================
# 智能模式
# ============================================================

def generate_smart_alpha() -> Optional[Node]:
    choice = random.randint(0, 9)

    if choice == 0:
        return price_momentum()
    if choice == 1:
        return price_momentum_ratio()
    if choice == 2:
        return price_vs_mean()
    if choice == 3:
        return mean_reversion()
    if choice == 4:
        return volume_activity()
    if choice == 5:
        return volume_trend()
    if choice == 6:
        return volatility_signal()
    if choice == 7:
        a = price_momentum()
        b = volume_activity()
        if a is None or b is None:
            return None
        return combine_nodes(a, b)
    if choice == 8:
        a = price_momentum()
        b = volatility_signal()
        if a is None or b is None:
            return None
        return combine_nodes(a, b)

    a = price_vs_mean()
    b = volume_activity()
    if a is None or b is None:
        return None
    return combine_nodes(a, b)


# ============================================================
# 初始 Alpha（对外统一入口）
# ============================================================

def generate_initial_alpha(attempts: int = 100) -> Optional[Node]:
    for _ in range(attempts):
        if random.random() < 0.65:
            node = generate_grammar_alpha()
        else:
            node = generate_smart_alpha()

        if node is None:
            continue

        ok, _ = validate_node(node)
        if not ok:
            continue

        return node

    return None
