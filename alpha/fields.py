# alpha/fields.py
# -*- coding: utf-8 -*-

"""
Data Field Catalog。

职责边界
--------
只提供数据字段的目录信息：
    - 名称（id）
    - 语义类型（price / volume / return / matrix）

API schema（已由本机实测确认）
-------------------------------
    {
      "id": "accrued_liabilities_total",
      "description": "...",
      "dataset": {"id": "fundamental2", "name": "..."},
      "category": {"id": "fundamental", "name": "..."},
      "type": "MATRIX",   # 或 "VECTOR"
      "coverage": 0.44,
      "region": "USA",
      "delay": 1,
      "universe": "TOP3000",
      ...
    }

类型映射规则
------------
    1. type != "MATRIX" → 排除（暂不支持 VECTOR）
    2. id 命中 price 白名单 → "price"
    3. id 命中 volume 白名单 → "volume"
    4. id 命中 return 白名单 → "return"
    5. 其它 → "matrix"

加载优先级
----------
    1. 手动 set_default_catalog()
    2. load_from_api() / load_from_json()
    3. hardcoded_catalog()（最终降级，6 个字段）

本模块不含：
    - 网络请求（在 brain/capabilities.py）
    - 缓存策略
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 名称白名单（用于类型映射）
# ============================================================

_PRICE_NAMES = frozenset({
    "open", "high", "low", "close", "vwap", "price",
    "mid", "bid", "ask",
    "cap", "market_cap", "mktcap", "marketcap",
})

_VOLUME_NAMES = frozenset({
    "volume", "vol", "adv20", "adv",
    "dollar_volume", "dollarvol",
    "shares_out", "sharesout", "shares",
    "turnover",
})

_RETURN_NAMES = frozenset({
    "returns", "ret", "ret_1d", "pct_change",
    "log_returns", "logret",
})


# ============================================================
# 数据定义
# ============================================================

@dataclass(frozen=True)
class FieldDefinition:
    name: str
    field_type: str
    dataset: str = ""
    dataset_name: str = ""
    category: str = ""
    coverage: float = 0.0
    description: str = ""


# ============================================================
# Catalog
# ============================================================

class FieldCatalog:
    def __init__(self, fields: Iterable[FieldDefinition]) -> None:
        self._fields: Tuple[FieldDefinition, ...] = tuple(fields)
        self._by_name: Dict[str, FieldDefinition] = {
            f.name: f for f in self._fields
        }

    def get(self, name: str) -> Optional[FieldDefinition]:
        return self._by_name.get(name)

    def names(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self._fields)

    def field_types(self) -> Dict[str, str]:
        return {f.name: f.field_type for f in self._fields}

    def price_fields(self) -> Tuple[str, ...]:
        return tuple(
            f.name for f in self._fields
            if f.field_type == "price"
        )

    def by_type(self, field_type: str) -> Tuple[str, ...]:
        return tuple(
            f.name for f in self._fields
            if f.field_type == field_type
        )

    def all(self) -> Tuple[FieldDefinition, ...]:
        return self._fields

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._fields)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self._fields:
            counts[f.field_type] = counts.get(f.field_type, 0) + 1
        counts["_total"] = len(self._fields)
        return counts

    # --------------------------------------------------------
    # 合并
    # --------------------------------------------------------

    def merge(self, other: "FieldCatalog") -> "FieldCatalog":
        """
        合并两个 catalog：self 为基础层，other 为扩展层。

        规则
        ----
        - 同名字段：**other 优先**（远程覆盖本地）
        - 仅 self 有的字段：保留（如本地硬编码的 close/volume）
        - 仅 other 有的字段：追加

        返回新的 FieldCatalog，不修改 self / other。
        """
        merged: Dict[str, FieldDefinition] = {
            f.name: f for f in self._fields
        }
        # other 覆盖同名
        for f in other._fields:
            merged[f.name] = f

        return FieldCatalog(list(merged.values()))


# ============================================================
# 默认 Catalog（硬编码）
# ============================================================

def hardcoded_catalog() -> FieldCatalog:
    """
    V3 已验证的 6 个价量字段 + fundamental6 基本面字段。

    最终降级选项（未加载 API / JSON 时使用）。

    说明
    ----
    基本面字段来自 alpha/families.py 的 _FUNDAMENTAL_FIELD_POOL，
    类型统一标为 "matrix"（与 _infer_field_type 的默认一致）。
    加入它们是为了让 parser 能识别 debt / assets 等字段，
    否则 parse_expression('rank(ts_mean(debt,120))') 会返回 None。
    """
    return FieldCatalog([
        # ---- 价量字段（V3 已验证）----
        FieldDefinition("open",    "price"),
        FieldDefinition("high",    "price"),
        FieldDefinition("low",     "price"),
        FieldDefinition("close",   "price"),
        FieldDefinition("volume",  "volume"),
        FieldDefinition("returns", "return"),
        # ---- fundamental6 基本面字段 ----
        FieldDefinition("assets",           "matrix"),
        FieldDefinition("capex",            "matrix"),
        FieldDefinition("cash",             "matrix"),
        FieldDefinition("cashflow",         "matrix"),
        FieldDefinition("cashflow_op",      "matrix"),
        FieldDefinition("cogs",             "matrix"),
        FieldDefinition("current_ratio",    "matrix"),
        FieldDefinition("debt",             "matrix"),
        FieldDefinition("ebit",             "matrix"),
        FieldDefinition("ebitda",           "matrix"),
        FieldDefinition("employee",         "matrix"),
        FieldDefinition("enterprise_value", "matrix"),
        FieldDefinition("eps",              "matrix"),
        FieldDefinition("equity",           "matrix"),
        FieldDefinition("bookvalue_ps",     "matrix"),
    ])


# ============================================================
# 全局默认 Catalog（可变）
# ============================================================

_default_catalog: FieldCatalog = hardcoded_catalog()


def default_catalog() -> FieldCatalog:
    return _default_catalog


def set_default_catalog(catalog: FieldCatalog) -> None:
    """
    替换全局默认 catalog。

    ⚠️ 与 operators.py 一致：项目内部模块 **不** 在 import 时
    缓存字段常量，而是运行时通过 default_catalog() 读取。
    """
    global _default_catalog
    global FIELD_TYPES, FIELDS, PRICE_FIELDS

    _default_catalog = catalog
    FIELD_TYPES = catalog.field_types()
    FIELDS = catalog.names()
    PRICE_FIELDS = catalog.price_fields()

    logger.info("FieldCatalog 已更新: %s", catalog.summary())


# ============================================================
# schema 解析
# ============================================================

def _extract_id(item: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "name", "field"):
        v = item.get(key)
        if v:
            return str(v).strip()
    return None


def _extract_type(item: Dict[str, Any]) -> str:
    for key in ("type", "fieldType", "field_type"):
        v = item.get(key)
        if v:
            return str(v).strip()
    return ""


def _extract_dataset(item: Dict[str, Any]) -> Tuple[str, str]:
    """返回 (dataset_id, dataset_name)。"""
    v = item.get("dataset")
    if isinstance(v, dict):
        return (
            str(v.get("id") or ""),
            str(v.get("name") or ""),
        )
    if v is not None:
        return str(v), ""
    return "", ""


def _extract_category(item: Dict[str, Any]) -> str:
    v = item.get("category")
    if isinstance(v, dict):
        return str(v.get("id") or v.get("name") or "")
    if v is not None:
        return str(v)
    return ""


def _extract_coverage(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("coverage", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _infer_field_type(field_id: str) -> str:
    """从 id 推断语义类型。"""
    n = field_id.lower()
    if n in _PRICE_NAMES:
        return "price"
    if n in _VOLUME_NAMES:
        return "volume"
    if n in _RETURN_NAMES:
        return "return"
    return "matrix"


# ============================================================
# 从 API 构建 Catalog
# ============================================================

def catalog_from_api(
    fields: List[Dict[str, Any]],
    *,
    max_fields: int = 200,
    min_coverage: float = 0.0,
) -> FieldCatalog:
    """
    把 BRAIN API 返回的字段列表转成 FieldCatalog。

    Parameters
    ----------
    fields       : 原始 API 返回
    max_fields   : 最多加载多少个（默认 200）
    min_coverage : 最低覆盖率阈值（0.0~1.0），低于此值过滤
    """
    defs: List[FieldDefinition] = []
    seen: set = set()

    # 保留 V3 已知字段
    for f in hardcoded_catalog().all():
        defs.append(f)
        seen.add(f.name)

    for item in fields:
        if not isinstance(item, dict):
            continue

        field_id = _extract_id(item)
        if not field_id or field_id in seen:
            continue

        # 排除非 MATRIX
        raw_type = _extract_type(item).upper()
        if raw_type and raw_type != "MATRIX":
            continue

        # 覆盖率过滤
        coverage = _extract_coverage(item)
        if coverage < min_coverage:
            continue

        dataset_id, dataset_name = _extract_dataset(item)
        category = _extract_category(item)
        field_type = _infer_field_type(field_id)
        description = str(item.get("description", "") or "")

        defs.append(FieldDefinition(
            name=field_id,
            field_type=field_type,
            dataset=dataset_id,
            dataset_name=dataset_name,
            category=category,
            coverage=coverage,
            description=description,
        ))
        seen.add(field_id)

        if len(defs) >= max_fields:
            break

    catalog = FieldCatalog(defs)
    logger.info(
        "从 API 构建 FieldCatalog: %s (max=%d, min_cov=%.2f)",
        catalog.summary(), max_fields, min_coverage,
    )
    return catalog


# ============================================================
# 便捷入口
# ============================================================

def load_from_api(
    session,
    endpoints,
    polling,
    *,
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    instrument_type: str = "EQUITY",
    dataset_id: Optional[str] = None,
    max_fields: int = 200,
    max_pages: int = 40,
    min_coverage: float = 0.0,
    set_as_default: bool = True,
    merge_with_local: bool = True,
) -> Optional[FieldCatalog]:
    """
    从 BRAIN API 加载字段。失败返回 None。

    merge_with_local : True（默认）时把远程字段合并进
        hardcoded_catalog（保留本地独有的 close/volume 等）。
    """
    from brain.capabilities import fetch_data_fields

    fields = fetch_data_fields(
        session, endpoints, polling,
        region=region, delay=delay, universe=universe,
        instrument_type=instrument_type,
        dataset_id=dataset_id,
        max_pages=max_pages,
    )
    if not fields:
        logger.warning("从 API 获取数据字段失败或为空")
        return None

    remote = catalog_from_api(
        fields,
        max_fields=max_fields,
        min_coverage=min_coverage,
    )

    if merge_with_local:
        # 以"当前 default_catalog"为 base，而非 hardcoded_catalog()。
        #
        # 原因：多次调用 load_from_api 加载不同 dataset 时，
        # 若每次都以 hardcoded_catalog() 为 base，则上一次加载的
        # 远程字段会被覆盖丢失。
        #   第一次 analyst4: base=21 + analyst4=47 → 47
        #   第二次 option8 : base=21 + option8=65  → 65（analyst4 丢失）
        #
        # 改为 default_catalog() 后：
        #   第一次 analyst4: base=21 + analyst4=47 → 47
        #   第二次 option8 : base=47 + option8=65  → 47+65（analyst4 保留）
        #
        # 若尚未加载过任何远程 dataset，default_catalog() 就是
        # hardcoded_catalog()，行为与旧版一致。
        base = default_catalog()
        catalog = base.merge(remote)
        logger.info(
            "远程字段池已合并进本地: base=%s remote=%s merged=%s",
            base.summary(), remote.summary(), catalog.summary(),
        )
    else:
        catalog = remote

    if set_as_default:
        set_default_catalog(catalog)
    return catalog


def load_from_json(
    path: str,
    *,
    max_fields: int = 200,
    min_coverage: float = 0.0,
    set_as_default: bool = True,
    merge_with_local: bool = True,
) -> Optional[FieldCatalog]:
    """从本地 JSON 加载。merge_with_local 同 load_from_api。"""
    if not os.path.exists(path):
        logger.warning("JSON catalog 不存在: %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("读取 %s 失败: %r", path, e)
        return None

    if isinstance(data, list):
        fields = data
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        fields = data["results"]
    else:
        logger.warning("JSON catalog 格式未知: %s", path)
        return None

    remote = catalog_from_api(
        fields,
        max_fields=max_fields,
        min_coverage=min_coverage,
    )

    if merge_with_local:
        base = hardcoded_catalog()
        catalog = base.merge(remote)
        logger.info(
            "JSON 字段池已合并进本地: base=%s remote=%s merged=%s",
            base.summary(), remote.summary(), catalog.summary(),
        )
    else:
        catalog = remote

    if set_as_default:
        set_default_catalog(catalog)
    return catalog


# ============================================================
# 模块级常量（兼容旧代码）
# ============================================================

FIELD_TYPES: Dict[str, str] = _default_catalog.field_types()
FIELDS: Tuple[str, ...] = _default_catalog.names()
PRICE_FIELDS: Tuple[str, ...] = _default_catalog.price_fields()
