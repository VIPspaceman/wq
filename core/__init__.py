# -*- coding: utf-8 -*-

"""
core：系统唯一的候选对象与共享工具。

职责边界
--------
本模块是 **唯一真相来源（single source of truth）**：

    - AlphaCandidate      : 系统唯一的候选对象
    - expression_hash     : 表达式哈希（唯一实现）
    - structure_signature : 结构签名（唯一实现）
    - get_field_registry  : 字段池注册表（唯一来源）
    - get_operator_registry : 算子池注册表（唯一来源）

设计动机
--------
在引入 core/ 之前，以下逻辑在多个模块中重复实现：

    - expression_hash     : utils/hashing.py + storage/database.py
    - structure_signature : alpha/ast.py + tools/auto_submit.py（正则近似）
    - 字段池              : alpha/families.py + alpha/fields.py +
                            alpha/saturation.py + tools/auto_submit.py
    - 算子池              : alpha/operators.py + agent/prompts.py

core/ 把这些收敛到一处。**现有模块暂不改动**（保持向后兼容），
新代码应优先使用 core/。

不含：
    - 业务逻辑（生成 / 评估 / 修复）
    - 网络请求
    - 持久化

典型用法
--------
    from core import AlphaCandidate, expression_hash

    cand = AlphaCandidate.from_expression("rank(ts_delta(close,5))")
    print(cand.expr_hash, cand.structure_sig)
"""

from core.candidate import AlphaCandidate
from core.hashing import expression_hash
from core.registry import (
    FieldRegistry,
    OperatorRegistry,
    get_field_registry,
    get_operator_registry,
)
from core.signature import structure_signature

__all__ = [
    "AlphaCandidate",
    "expression_hash",
    "structure_signature",
    "FieldRegistry",
    "OperatorRegistry",
    "get_field_registry",
    "get_operator_registry",
]
