# evolution/mutation.py
# -*- coding: utf-8 -*-

"""
变异算子。

职责边界
--------
对齐 V3 mutate()：
    - A. 子树替换为语法驱动 Alpha（35%）
    - B. 修改时间窗口（20%）
    - C. 修改字段（15%）
    - D. 修改运算符（15%）
    - E. 注入跨家族因子（15%）

运行时读取 operators
--------------------
本模块 **不** 在 import 时缓存 UNARY_OPERATORS / TS_OPERATORS /
BINARY_OPERATORS / WINDOWS；而是每次变异时通过
alpha.operators.default_catalog() 读取。
这样 set_default_catalog() 生效后，变异立刻用新算子集。

不含：
    - 交叉（crossover.py）
    - 种群管理（population.py）
    - 校验规则（validator.py）
"""

from __future__ import annotations

import random
from typing import Optional

from alpha.ast import (
    Node,
    clone_node,
    collect_nodes,
    field,
    replace_subtree,
)
from alpha.fields import default_catalog as _field_catalog
from alpha.generator import (
    combine_nodes,
    generate_grammar_alpha,
    generate_initial_alpha,
)
from alpha.operators import default_catalog
from alpha.validator import validate_node


def _is_coefficient_mul(node: Node) -> bool:
    """
    FIX-8：判断节点是否为"系数乘法"。

    条件：
        - node.kind == "binary"
        - node.value == "*"
        - node.left 或 node.right 是 constant("-1") / constant("-1.0")

    这类节点的 * 是取反语义（来自 rule_invert / sign_flip），
    不应被 mutation D 分支随机改写。
    """
    if node.kind != "binary" or node.value != "*":
        return False

    def _is_neg_one(n: Optional[Node]) -> bool:
        if n is None or n.kind != "constant":
            return False
        return n.value in ("-1", "-1.0")

    return _is_neg_one(node.left) or _is_neg_one(node.right)


def mutate(original: Node) -> Optional[Node]:
    """
    对 AST 施加一次变异；失败返回 None。
    """
    root = clone_node(original)
    nodes = collect_nodes(root)
    if not nodes:
        return None

    mutation_type = random.random()

    # -------- A. 子树替换 --------
    if mutation_type < 0.35:
        target = random.choice(nodes)
        replacement = generate_initial_alpha()
        if replacement is None:
            return None
        child = replace_subtree(root, target, replacement)

    # -------- B. 修改窗口 --------
    elif mutation_type < 0.55:
        ts_nodes = [x for x in nodes if x.kind == "ts"]
        if not ts_nodes:
            return None
        # 运行时读取 windows
        windows = default_catalog().windows()
        if not windows:
            return None
        target = random.choice(ts_nodes)
        target.window = random.choice(windows)
        child = root

    # -------- C. 修改字段 --------
    elif mutation_type < 0.70:
        field_nodes = [x for x in nodes if x.kind == "field"]
        if not field_nodes:
            return None
        # 运行时读取字段（set_default_catalog 后立即生效）
        field_names = _field_catalog().names()
        if not field_names:
            return None
        target = random.choice(field_nodes)
        new_field = field(random.choice(field_names))
        child = replace_subtree(root, target, new_field)

    # -------- D. 修改运算符 --------
    elif mutation_type < 0.85:
        catalog = default_catalog()

        # FIX-8：跳过"系数乘法"节点（(-1*x) 里的 * 是取反语义，
        # 不应被随机改写）。最多重试 5 次。
        target = None
        for _ in range(5):
            cand = random.choice(nodes)
            if _is_coefficient_mul(cand):
                continue
            target = cand
            break

        if target is None:
            return None

        if target.kind == "unary":
            ops = catalog.unary()
            if not ops:
                return None
            target.value = random.choice(ops)
        elif target.kind == "ts":
            ops = catalog.ts()
            if not ops:
                return None
            target.value = random.choice(ops)
        elif target.kind == "binary":
            ops = catalog.binary()
            if not ops:
                return None
            target.value = random.choice(ops)
        child = root

    # -------- E. 注入跨家族因子 --------
    else:
        extra = generate_grammar_alpha()
        if extra is None:
            return None
        child = combine_nodes(root, extra)
        if child is None:
            return None

    ok, _ = validate_node(child)
    if not ok:
        return None
    return child
