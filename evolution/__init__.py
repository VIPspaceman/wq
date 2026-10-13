# evolution/__init__.py
# -*- coding: utf-8 -*-

"""
evolution：进化算子层。

子模块：
    mutation    : 变异
    crossover   : 交叉
    reverse     : 方向探索
    selection   : 精英选择
    diversity   : 家族多样性约束
    population  : 初始种群 / 下一代种群

⚠️ 本层不调用 BRAIN；只操作 alpha.ast.Node 和 evaluation.AlphaRecord。
"""

from evolution.crossover import crossover
from evolution.diversity import filter_by_family_diversity
from evolution.mutation import mutate
from evolution.population import (
    create_initial_population,
    create_next_generation,
)
from evolution.reverse import reverse_alpha
from evolution.selection import select_elites

__all__ = [
    "mutate",
    "crossover",
    "reverse_alpha",
    "select_elites",
    "filter_by_family_diversity",
    "create_initial_population",
    "create_next_generation",
]