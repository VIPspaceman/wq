# alpha/operators.py
# -*- coding: utf-8 -*-

"""
Operator Catalog。

职责边界
--------
只提供当前可用运算符的目录信息。

分类策略（Step 8b 修正 v4）
---------------------------
在 v3 基础上，额外排除 **单位敏感** 的 ts 算子：
    - ts_product
它本质是"累乘"，会让输入字段的单位指数变化，触发 BRAIN 的
unit checker 报 WARNING。

本模块的排除名单汇总
--------------------
    前缀排除      : group_ / vec_
    名称排除      : if_else / trade_when / bucket / ...
    逻辑比较排除  : and / or / not / equal / ...
    命名二元算术  : add / subtract / multiply / divide / ...
    单位敏感 unary: sqrt / log / inverse / signed_power / densify
    单位敏感 ts   : ts_product

不含：
    - 网络请求（在 brain/capabilities.py）
    - 缓存策略
    - 运算符语义实现
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 排除规则
# ============================================================

# 名字前缀：明确不支持（需要 group 字段 / 向量输入）
_UNSUPPORTED_PREFIXES: Tuple[str, ...] = (
    "group_",
    "vec_",
)

# 名字名单：明确不支持（3 参数 / 特殊语义 / arity 不确定）
_UNSUPPORTED_NAMES: frozenset = frozenset({
    "if_else", "trade_when", "bucket",
    "days_from_last_change", "hump", "kth_element", "last_diff_value",
    "is_nan",
})

# 逻辑 / 比较类：返回 0/1 序列，套 rank 后无信息量
_NOISY_BINARY: frozenset = frozenset({
    "and", "or", "not",
    "equal", "greater", "less",
    "greater_equal", "less_equal", "not_equal",
})

# 命名形式的算术二元算子：AST 不支持中缀以外的序列化
_NAMED_BINARY_ARITH: frozenset = frozenset({
    "add", "subtract", "multiply", "divide",
    "power", "signed_power",
    "min", "max",
})

# 单位敏感的 unary 算子
# 它们要求无量纲输入；套在价格/成交量上会触发 BRAIN 的 unit checker
_UNIT_SENSITIVE_UNARY: frozenset = frozenset({
    "sqrt",           # 需要无量纲
    "log",            # 需要无量纲
    "inverse",        # 1/x，单位变化
    "signed_power",   # 同 sqrt
    "densify",        # 效果不稳定，保守排除
})

# 单位敏感的 ts 算子
# ★ 新增：乘法累积会改变单位，BRAIN unit checker 拒绝
_UNIT_SENSITIVE_TS: frozenset = frozenset({
    "ts_product",     # close^d 这类单位指数变化
})

# 符号形式的二元运算符（唯一被 binary 接受的）
_SYMBOLIC_BINARY: frozenset = frozenset({"+", "-", "*", "/"})

# 补充白名单：API 不返回但实际可用的算子
# 按 "kind: name" 格式显式声明
_MANUAL_OPERATORS = {
    "ts_backfill": "ts",   # API 不返回，但实测可用
}

# 内部包装算子：由 alpha/ast.py:safe_field() 等内部 helper 生成，
# **不是** 用户直接书写的算子，因此 BRAIN /operators API 不会返回它们。
#
# ⚠️ 关键：这些算子必须 **始终** 存在于 catalog 中，否则
#    alpha/validator.py:_check_operators 会拒绝所有含它们的表达式，
#    导致 generate_grammar_alpha 100% 返回 None。
#
# 背景（2026-10-11）：
#     safe_field(name) = winsorize(ts_backfill(name, 120))
#     远程 catalog 加载后（load_from_api → set_default_catalog）
#     会 **完全替换** 本地 catalog，而 API 不返回 winsorize，
#     导致 winsorize 丢失 → 所有 safe_field 表达式被拒。
#
# 因此 catalog_from_api 必须把本列表合并进远程 catalog。
#
# 格式：{name: kind}
_INTERNAL_WRAPPER_OPERATORS = {
    "winsorize": "unary",   # safe_field 外层
    "ts_backfill": "ts",    # safe_field 内层
}

# 二元时序算子白名单（2 个字段输入 + 1 个窗口）
# 这些算子在 BRAIN 里是 ts_xxx(x, y, d) 形式，
# 与一元 ts_xxx(x, d) 不同，需要 AST 的 ts_binary 节点支持。
_BINARY_TS_NAMES: frozenset = frozenset({
    "ts_corr",   # 滚动相关系数
})

# 可变参数记号（出现在 definition 里）
_VARARGS_TOKENS: frozenset = frozenset({"..", "..."})


# ============================================================
# 数据定义
# ============================================================

@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    kind: str          # "unary" / "ts" / "binary"
    arity: int         # 1 或 2
    category: str = ""
    description: str = ""
    definition: str = ""


# ============================================================
# Catalog
# ============================================================

class OperatorCatalog:
    def __init__(
        self,
        unary: Iterable[str],
        ts: Iterable[str],
        binary: Iterable[str],
        windows: Iterable[int],
        *,
        definitions: Optional[Dict[str, OperatorDefinition]] = None,
        other: Optional[Iterable[str]] = None,
        ts_binary: Optional[Iterable[str]] = None,
    ) -> None:
        self._unary = tuple(unary)
        self._ts = tuple(ts)
        self._binary = tuple(binary)
        self._windows = tuple(windows)
        self._other = tuple(other or ())
        self._ts_binary = tuple(ts_binary or ())

        if definitions is not None:
            self._defs = dict(definitions)
        else:
            self._defs: Dict[str, OperatorDefinition] = {}
            for n in self._unary:
                self._defs[n] = OperatorDefinition(n, "unary", 1)
            for n in self._ts:
                self._defs[n] = OperatorDefinition(n, "ts", 1)
            for n in self._binary:
                self._defs[n] = OperatorDefinition(n, "binary", 2)
            for n in self._ts_binary:
                self._defs[n] = OperatorDefinition(n, "ts_binary", 3)

    # ---- 查询 ----
    def unary(self) -> Tuple[str, ...]:
        return self._unary

    def ts(self) -> Tuple[str, ...]:
        return self._ts

    def binary(self) -> Tuple[str, ...]:
        return self._binary

    def windows(self) -> Tuple[int, ...]:
        return self._windows

    def other(self) -> Tuple[str, ...]:
        return self._other

    def ts_binary(self) -> Tuple[str, ...]:
        return self._ts_binary

    def get(self, name: str) -> Optional[OperatorDefinition]:
        return self._defs.get(name)

    def all_definitions(self) -> Dict[str, OperatorDefinition]:
        return dict(self._defs)

    def summary(self) -> Dict[str, int]:
        return {
            "unary": len(self._unary),
            "ts": len(self._ts),
            "binary": len(self._binary),
            "ts_binary": len(self._ts_binary),
            "other": len(self._other),
            "total": len(self._defs),
        }

    # --------------------------------------------------------
    # 合并
    # --------------------------------------------------------

    def merge(self, other: "OperatorCatalog") -> "OperatorCatalog":
        """
        合并两个 catalog：self 为基础层，other 为扩展层。

        规则
        ----
        - 同名算子：**other 优先**（远程覆盖本地）
        - 仅 self 有的算子：保留（如 winsorize，API 不返回）
        - 仅 other 有的算子：追加
        - windows：other 非空则用 other，否则用 self

        返回新的 OperatorCatalog，不修改 self / other。
        """
        # ---- 合并各分类列表（去重，保持顺序）----
        def _merge_list(base, extra):
            seen = set()
            out = []
            for x in list(base) + list(extra):
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return tuple(out)

        merged_unary = _merge_list(self._unary, other._unary)
        merged_ts = _merge_list(self._ts, other._ts)
        merged_binary = _merge_list(self._binary, other._binary)
        merged_ts_binary = _merge_list(
            self._ts_binary, other._ts_binary,
        )
        merged_other = _merge_list(self._other, other._other)

        # ---- 合并 definitions：other 优先 ----
        merged_defs: Dict[str, OperatorDefinition] = dict(self._defs)
        merged_defs.update(other._defs)

        # ---- windows：other 非空则用 other ----
        merged_windows = (
            other._windows if other._windows else self._windows
        )

        return OperatorCatalog(
            unary=merged_unary,
            ts=merged_ts,
            binary=merged_binary,
            windows=merged_windows,
            definitions=merged_defs,
            other=merged_other,
            ts_binary=merged_ts_binary,
        )


# ============================================================
# 默认 Catalog（硬编码，最终降级）
# ============================================================

_DEFAULT_WINDOWS: Tuple[int, ...] = (
    2, 3, 5, 10, 15, 20, 30,
)


def hardcoded_catalog() -> OperatorCatalog:
    """
    V3 已验证的硬编码默认值。最终降级选项。
    """
    return OperatorCatalog(
        # winsorize：safe_field() 依赖它（winsorize(ts_backfill(x,120))）。
        # 若缺失，_check_operators 会拒绝所有 safe_field 表达式，
        # 导致 generate_grammar_alpha 100% 返回 None。
        unary=("rank", "abs", "sign", "winsorize"),
        ts=("ts_delay", "ts_delta", "ts_mean", "ts_std_dev",
            "ts_rank", "ts_backfill"),
        binary=("+", "-", "*", "/"),
        windows=_DEFAULT_WINDOWS,
        ts_binary=("ts_corr",),
    )


# ============================================================
# 全局默认 Catalog（可变）
# ============================================================

_default_catalog: OperatorCatalog = hardcoded_catalog()


def default_catalog() -> OperatorCatalog:
    return _default_catalog


def set_default_catalog(catalog: OperatorCatalog) -> None:
    """
    替换全局默认 catalog。

    ⚠️ 注意 Python 的 import 绑定语义：
        from alpha.operators import UNARY_OPERATORS
    会在 import 时把名字绑定到 **当时的值**。之后调用本函数更新
    模块级变量，**不会** 自动更新已 import 的模块里的名字。

    因此本项目内部：
        - parser.py / mutation.py 等模块 **不在 import 时缓存常量**，
          而是在运行时通过 default_catalog() 读取最新值
        - 模块级常量保留只是为了兼容旧代码 / 外部脚本
    """
    global _default_catalog
    global UNARY_OPERATORS, TS_OPERATORS, BINARY_OPERATORS, WINDOWS
    global TS_BINARY_OPERATORS

    _default_catalog = catalog
    UNARY_OPERATORS = catalog.unary()
    TS_OPERATORS = catalog.ts()
    BINARY_OPERATORS = catalog.binary()
    TS_BINARY_OPERATORS = catalog.ts_binary()
    WINDOWS = catalog.windows()

    # 防御性检查：内部包装算子必须存在，否则 safe_field 表达式
    # 会被 _check_operators 拒绝（历史 bug，见 _INTERNAL_WRAPPER_OPERATORS）。
    missing = [
        name for name, kind in _INTERNAL_WRAPPER_OPERATORS.items()
        if catalog.get(name) is None
    ]
    if missing:
        logger.warning(
            "⚠️ 新 catalog 缺少内部包装算子 %s，"
            "safe_field 表达式可能被 validate_node 拒绝",
            missing,
        )

    logger.info("OperatorCatalog 已更新: %s", catalog.summary())


# ============================================================
# definition 解析
# ============================================================

def _count_required_params(definition: str) -> Optional[int]:
    """
    从 definition 里解析 **必需参数个数**。

    例：
        'add(x, y, filter = false), x + y'  → 2
        'sqrt(x)'                            → 1
        'if_else(cond, x, y)'                → 3
        'max(x, y, ..)'                      → 2  （.. 是可变参数记号）

    规则：
        - 只取第一个 '(' 到匹配 ')' 里的内容
        - 按顶层逗号分割
        - 排除含 '=' 的参数（可选参数）
        - 排除 '..' 和 '...'（可变参数记号）
        - 无法解析 → None
    """
    if not definition:
        return None

    start = definition.find("(")
    if start < 0:
        return None

    depth = 0
    end = -1
    for i in range(start, len(definition)):
        ch = definition[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None

    inner = definition[start + 1:end].strip()
    if not inner:
        return 0

    parts: List[str] = []
    depth = 0
    cur = ""
    for ch in inner:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())

    required = [
        p for p in parts
        if "=" not in p and p not in _VARARGS_TOKENS
    ]
    return len(required)


# ============================================================
# 分类
# ============================================================

def _classify(
    name: str,
    definition: str,
) -> Optional[str]:
    """
    根据名字 + definition 判断运算符种类。

    返回 "unary" / "ts" / "binary" / None（→ other）。
    """
    n = (name or "").strip()
    if not n:
        return None

    # -------- 明确不支持 --------
    for prefix in _UNSUPPORTED_PREFIXES:
        if n.startswith(prefix):
            return None
    if n in _UNSUPPORTED_NAMES:
        return None
    if n in _NOISY_BINARY:
        return None
    if n in _NAMED_BINARY_ARITH:
        return None
    if n in _UNIT_SENSITIVE_UNARY:
        return None
    if n in _UNIT_SENSITIVE_TS:      # ★ 新增：ts_product 排除
        return None

    # -------- 符号二元 --------
    if n in _SYMBOLIC_BINARY:
        return "binary"

    # -------- 解析 arity --------
    arity = _count_required_params(definition)
    if arity is None:
        return None

    # -------- 二元时序算子（白名单）--------
    # ts_corr(x, y, d) / ts_cov(x, y, d)：arity == 3
    if n in _BINARY_TS_NAMES:
        if arity == 3:
            return "ts_binary"
        return None

    # -------- ts_ 前缀：只接受 2 参数形式 ts_xxx(x, d) --------
    if n.startswith("ts_"):
        if arity == 2:
            return "ts"
        return None

    # -------- 按 arity 分类 --------
    if arity == 1:
        return "unary"
    if arity == 2:
        return "binary"
    return None


# ============================================================
# 从 BRAIN API 构建 Catalog
# ============================================================

def catalog_from_api(
    operators: List[Dict[str, Any]],
) -> OperatorCatalog:
    """
    把 BRAIN API 返回的运算符列表转成 OperatorCatalog。
    """
    unary: List[str] = []
    ts: List[str] = []
    binary: List[str] = []
    ts_binary: List[str] = []
    other: List[str] = []
    defs: Dict[str, OperatorDefinition] = {}
    seen: set = set()

    for item in operators:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)

        category = str(item.get("category", "") or "")
        description = str(item.get("description", "") or "")
        definition = str(item.get("definition", "") or "")

        kind = _classify(name, definition)

        if kind is None:
            other.append(name)
            continue

        if kind == "binary":
            arity = 2
        elif kind == "ts_binary":
            arity = 3
        else:
            arity = 1
        defs[name] = OperatorDefinition(
            name=name,
            kind=kind,
            arity=arity,
            category=category,
            description=description,
            definition=definition,
        )

        if kind == "unary":
            unary.append(name)
        elif kind == "ts":
            ts.append(name)
        elif kind == "binary":
            binary.append(name)
        elif kind == "ts_binary":
            ts_binary.append(name)

    # 合并补充白名单
    for name, kind in _MANUAL_OPERATORS.items():
        if kind == "unary" and name not in unary:
            unary.append(name)
            defs.setdefault(name, OperatorDefinition(
                name=name, kind="unary", arity=1,
                definition=f"{name}(x)",
            ))
        elif kind == "ts" and name not in ts:
            ts.append(name)
            defs.setdefault(name, OperatorDefinition(
                name=name, kind="ts", arity=1,
                definition=f"{name}(x, d)",
            ))
        elif kind == "binary" and name not in binary:
            binary.append(name)
            defs.setdefault(name, OperatorDefinition(
                name=name, kind="binary", arity=2,
                definition=f"{name}(x, y)",
            ))
        elif kind == "ts_binary" and name not in ts_binary:
            ts_binary.append(name)
            defs.setdefault(name, OperatorDefinition(
                name=name, kind="ts_binary", arity=3,
                definition=f"{name}(x, y, d)",
            ))

    # 合并内部包装算子（safe_field 等依赖）
    # ⚠️ 必须无条件合并：远程 catalog 会完全替换本地 catalog，
    #    而 API 不返回这些内部算子。若丢失，_check_operators
    #    会拒绝所有 safe_field 表达式。
    for name, kind in _INTERNAL_WRAPPER_OPERATORS.items():
        if kind == "unary" and name not in unary:
            unary.append(name)
            defs.setdefault(name, OperatorDefinition(
                name=name, kind="unary", arity=1,
                definition=f"{name}(x)",
            ))
        elif kind == "ts" and name not in ts:
            ts.append(name)
            defs.setdefault(name, OperatorDefinition(
                name=name, kind="ts", arity=1,
                definition=f"{name}(x, d)",
            ))
        elif kind == "binary" and name not in binary:
            binary.append(name)
            defs.setdefault(name, OperatorDefinition(
                name=name, kind="binary", arity=2,
                definition=f"{name}(x, y)",
            ))
        elif kind == "ts_binary" and name not in ts_binary:
            ts_binary.append(name)
            defs.setdefault(name, OperatorDefinition(
                name=name, kind="ts_binary", arity=3,
                definition=f"{name}(x, y, d)",
            ))

    # 兜底：确保 + - * / 一定在 binary 里
    for b in _SYMBOLIC_BINARY:
        if b not in binary:
            binary.append(b)
            defs.setdefault(
                b,
                OperatorDefinition(
                    name=b,
                    kind="binary",
                    arity=2,
                    definition=f"{b}(x, y)",
                ),
            )

    # binary 排序：+ - * / 的顺序
    binary_order = ["+", "-", "*", "/"]
    binary_sorted = [b for b in binary_order if b in binary]
    for b in binary:
        if b not in binary_sorted:
            binary_sorted.append(b)

    catalog = OperatorCatalog(
        unary=tuple(sorted(set(unary))),
        ts=tuple(sorted(set(ts))),
        binary=tuple(binary_sorted),
        windows=_DEFAULT_WINDOWS,
        definitions=defs,
        other=tuple(sorted(set(other))),
        ts_binary=tuple(sorted(set(ts_binary))),
    )

    logger.info(
        "从 API 构建 OperatorCatalog: %s", catalog.summary()
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
    set_as_default: bool = True,
    merge_with_local: bool = True,
) -> Optional[OperatorCatalog]:
    """
    从 BRAIN API 加载运算符并可选地设为默认。

    Parameters
    ----------
    merge_with_local : bool
        True（默认）：远程 catalog **合并进** hardcoded_catalog，
            保留本地独有的算子（如 winsorize）。
        False：远程 catalog **完全替换** 本地（旧行为）。

    返回：
        - 成功 → OperatorCatalog（合并后的）
        - 失败 → None（**不抛异常**）
    """
    from brain.capabilities import fetch_operators

    ops = fetch_operators(session, endpoints, polling)
    if not ops:
        logger.warning(
            "从 API 获取运算符失败或为空，保持当前 catalog"
        )
        return None

    remote = catalog_from_api(ops)

    if merge_with_local:
        base = hardcoded_catalog()
        catalog = base.merge(remote)
        logger.info(
            "远程 catalog 已合并进本地: base=%s remote=%s merged=%s",
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
    set_as_default: bool = True,
    merge_with_local: bool = True,
) -> Optional[OperatorCatalog]:
    """
    从本地 JSON 文件加载运算符（离线 / 缓存模式）。

    merge_with_local : 同 load_from_api。
    """
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
        ops = data
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        ops = data["results"]
    else:
        logger.warning("JSON catalog 格式未知: %s", path)
        return None

    remote = catalog_from_api(ops)

    if merge_with_local:
        base = hardcoded_catalog()
        catalog = base.merge(remote)
        logger.info(
            "JSON catalog 已合并进本地: base=%s remote=%s merged=%s",
            base.summary(), remote.summary(), catalog.summary(),
        )
    else:
        catalog = remote

    if set_as_default:
        set_default_catalog(catalog)
    return catalog


# ============================================================
# 模块级常量（兼容旧代码 / 外部脚本）
# ============================================================

# ⚠️ 这些常量在 import 时绑定当前值。
# 项目内部模块（parser / mutation）**不** 直接使用它们，
# 而是运行时通过 default_catalog() 读取最新值。
# 保留它们只是为了兼容外部脚本。

UNARY_OPERATORS: Tuple[str, ...] = _default_catalog.unary()
TS_OPERATORS: Tuple[str, ...] = _default_catalog.ts()
BINARY_OPERATORS: Tuple[str, ...] = _default_catalog.binary()
TS_BINARY_OPERATORS: Tuple[str, ...] = _default_catalog.ts_binary()
WINDOWS: Tuple[int, ...] = _default_catalog.windows()
