# -*- coding: utf-8 -*-

"""
字段池 / 算子池注册表（唯一来源）。

职责边界
--------
提供对当前 FieldCatalog / OperatorCatalog 的只读访问：

    - FieldRegistry    : 字段池
    - OperatorRegistry : 算子池
    - get_field_registry()    : 单例入口
    - get_operator_registry() : 单例入口

设计动机
--------
在引入 core/ 之前，字段池 / 算子池在多个模块中重复硬编码：

    - alpha/families.py   : _FUNDAMENTAL_FIELD_POOL
    - alpha/fields.py     : hardcoded_catalog() 逐条 FieldDefinition
    - alpha/saturation.py : _PRICE_VOLUME_FIELDS
    - tools/auto_submit.py: _FUND_FIELDS / _PV_FIELDS
    - agent/prompts.py    : ALLOWED_OPERATORS / ALLOWED_FIELDS

core/registry.py 是唯一来源。**现有模块暂不改动**（保持向后兼容），
新代码应通过 get_field_registry() / get_operator_registry() 访问。

运行时读取（关键）
------------------
本模块 **不** 在 import 时缓存 catalog 内容，而是每次调用时
通过 alpha.fields.default_catalog() / alpha.operators.default_catalog()
读取最新值。这样 set_default_catalog() 生效后立即反映。

不含：
    - 网络请求（在 brain/capabilities.py）
    - catalog 的加载 / 替换（在 alpha/fields.py / alpha/operators.py）
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


# ============================================================
# FieldRegistry
# ============================================================

class FieldRegistry:
    """
    字段池只读视图。

    每次调用都从 alpha.fields.default_catalog() 读取最新值，
    因此 set_default_catalog() 后立即生效。
    """

    # --------------------------------------------------------
    # 基础查询
    # --------------------------------------------------------

    def names(self) -> Tuple[str, ...]:
        """返回所有字段名。"""
        from alpha.fields import default_catalog
        return default_catalog().names()

    def get(self, name: str):
        """返回字段定义（FieldDefinition）；不存在返回 None。"""
        from alpha.fields import default_catalog
        return default_catalog().get(name)

    def contains(self, name: str) -> bool:
        """字段是否存在。"""
        from alpha.fields import default_catalog
        return name in default_catalog()

    def field_types(self) -> Dict[str, str]:
        """返回 {field_name: field_type}。"""
        from alpha.fields import default_catalog
        return default_catalog().field_types()

    def price_fields(self) -> Tuple[str, ...]:
        """返回所有 price 类型字段。"""
        from alpha.fields import default_catalog
        return default_catalog().price_fields()

    def by_type(self, field_type: str) -> Tuple[str, ...]:
        """返回指定类型的字段。"""
        from alpha.fields import default_catalog
        return default_catalog().by_type(field_type)

    def summary(self) -> Dict[str, int]:
        """返回 {field_type: count, "_total": n}。"""
        from alpha.fields import default_catalog
        return default_catalog().summary()

    def __len__(self) -> int:
        from alpha.fields import default_catalog
        return len(default_catalog())


# ============================================================
# OperatorRegistry
# ============================================================

class OperatorRegistry:
    """
    算子池只读视图。

    每次调用都从 alpha.operators.default_catalog() 读取最新值，
    因此 set_default_catalog() 后立即生效。
    """

    # --------------------------------------------------------
    # 基础查询
    # --------------------------------------------------------

    def unary(self) -> Tuple[str, ...]:
        """返回所有一元算子。"""
        from alpha.operators import default_catalog
        return default_catalog().unary()

    def ts(self) -> Tuple[str, ...]:
        """返回所有一元时序算子。"""
        from alpha.operators import default_catalog
        return default_catalog().ts()

    def binary(self) -> Tuple[str, ...]:
        """返回所有二元算子（+ - * /）。"""
        from alpha.operators import default_catalog
        return default_catalog().binary()

    def ts_binary(self) -> Tuple[str, ...]:
        """返回所有二元时序算子（ts_corr / ts_cov）。"""
        from alpha.operators import default_catalog
        return default_catalog().ts_binary()

    def windows(self) -> Tuple[int, ...]:
        """返回所有可用窗口。"""
        from alpha.operators import default_catalog
        return default_catalog().windows()

    def get(self, name: str):
        """返回算子定义（OperatorDefinition）；不存在返回 None。"""
        from alpha.operators import default_catalog
        return default_catalog().get(name)

    def contains(self, name: str) -> bool:
        """算子是否存在。"""
        from alpha.operators import default_catalog
        return default_catalog().get(name) is not None

    def summary(self) -> Dict[str, int]:
        """返回 {kind: count, "total": n}。"""
        from alpha.operators import default_catalog
        return default_catalog().summary()


# ============================================================
# 单例入口
# ============================================================

_field_registry: Optional[FieldRegistry] = None
_operator_registry: Optional[OperatorRegistry] = None


def get_field_registry() -> FieldRegistry:
    """
    返回 FieldRegistry 单例。

    注意：单例本身无状态（每次调用都读最新 catalog），
    因此 set_default_catalog() 后无需重建单例。
    """
    global _field_registry
    if _field_registry is None:
        _field_registry = FieldRegistry()
    return _field_registry


def get_operator_registry() -> OperatorRegistry:
    """
    返回 OperatorRegistry 单例。

    注意：单例本身无状态（每次调用都读最新 catalog），
    因此 set_default_catalog() 后无需重建单例。
    """
    global _operator_registry
    if _operator_registry is None:
        _operator_registry = OperatorRegistry()
    return _operator_registry
