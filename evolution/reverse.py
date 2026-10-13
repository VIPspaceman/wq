# evolution/reverse.py
# -*- coding: utf-8 -*-

"""
方向探索算子。

对齐 V3 reverse_alpha()：
    对 Alpha 的某个二元节点做 +/-/* 变换，探索方向反转。

注意事项
--------
当前基础运算符集合中没有 unary negative，
因此这里以结构变换代替数学取负。
这是 **本项目内部的探索策略**，不是 BRAIN 官方推荐做法。
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
from alpha.generator import generate_grammar_alpha
from alpha.validator import validate_node


def reverse_alpha(node: Node) -> Optional[Node]:
    root = clone_node(node)
    nodes = collect_nodes(root)

    candidates = [x for x in nodes if x.kind == "binary"]
    if not candidates:
        return None

    target = random.choice(candidates)

    if target.value == "-":
        target.value = "+"
    elif target.value == "+":
        target.value = "-"
    elif target.value == "*":
        replacement = generate_grammar_alpha()
        if replacement is None:
            return None
        root = replace_subtree(root, target, replacement)
    else:
        return None

    ok, _ = validate_node(root)
    if not ok:
        return None
    return root