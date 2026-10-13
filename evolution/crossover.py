# evolution/crossover.py
# -*- coding: utf-8 -*-

"""
交叉算子。

职责边界
--------
对齐 V3 crossover()：
    - 克隆 a、b
    - 各自随机取一个节点
    - 用 b 的子树替换 a 的对应子树
    - 校验通过则返回，否则 None
"""

from __future__ import annotations

import random
from typing import Optional

from alpha.ast import (
    Node,
    clone_node,
    collect_nodes,
    replace_subtree,
)
from alpha.validator import validate_node


def crossover(a: Node, b: Node) -> Optional[Node]:
    root_a = clone_node(a)
    root_b = clone_node(b)

    nodes_a = collect_nodes(root_a)
    nodes_b = collect_nodes(root_b)
    if not nodes_a or not nodes_b:
        return None

    target_a = random.choice(nodes_a)
    target_b = random.choice(nodes_b)

    child = replace_subtree(root_a, target_a, target_b)

    ok, _ = validate_node(child)
    if not ok:
        return None
    return child