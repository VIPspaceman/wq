# -*- coding: utf-8 -*-

"""
BRAIN 饱和过滤。

背景
----
基于 2026-10-11 累计 8 条 BRAIN 提交实测：

通过（PASS）:
    - O086Z7pp: rank(quantile(ts_av_diff(volume,20))
                      + (-1*rank(ts_av_diff(close,20))))
    - GrOxlAbQ: rank((-1*ts_decay_linear(ts_quantile(returns,10)))
                      + quantile(ts_arg_min(eps,120)))
    - j28Arpej: -1*winsorize(ts_av_diff(close,15))

被拒:
    - debt/assets 系（self-corr 0.79-0.93）
    - capex 单字段系（sharpe/fitness/sub-universe 都不够）
    - std_dev(returns)*close（self-corr 0.98）

规律
----
    - 含价量字段（close/volume/returns/open/high/low/vwap）
      → 可能通过
    - 纯基本面（无任何价量字段）→ 大概率被拒
    - 不管是裸比率还是单字段，只要没有价量字段就被拒

判定规则（2026-10-11 更新）
--------------------------
**含基本面 + 无价量 = BRAIN 死角**

    条件 A：含 >= 1 个基本面字段（_FUNDAMENTAL_FIELD_POOL）
    条件 B：不含任何价量字段
            （close/open/high/low/volume/returns/vwap）

两个条件同时满足 → 判饱和（返回 True）。

说明
----
- 旧规则还要求"不含任何 ts_* 算子"，但该条件太严，
  导致 rank(ts_av_diff(bookvalue_ps,120)) 等漏网。
  新规则删掉该条件，只保留"含基本面 + 无价量"。
- 跨字段（如 returns + eps）因含价量字段 → 放行。

职责边界
--------
只负责：判断一个 AST 是否属于"纯基本面"（BRAIN 死角）。
不含：过滤动作本身（由 search/generation.py 承担）。

TODO
----
- 当前规则是经验规则，非 BRAIN 官方文档。
  未来若积累更多样本，可调整价量字段列表。
"""

from __future__ import annotations

from typing import Optional

from alpha.ast import Node, collect_nodes
from alpha.families import _FUNDAMENTAL_FIELD_POOL


# 价量字段列表（含任一即视为"含价量字段"）
_PRICE_VOLUME_FIELDS = frozenset({
    "close",
    "open",
    "high",
    "low",
    "volume",
    "returns",
    "vwap",
})


def _count_ts_ops(node: Optional[Node]) -> int:
    """
    统计"用户写的" ts_* 算子数量（排除 T1 自动包装的 ts_backfill）。

    背景
    ----
    pipeline 里所有字段都被 T1 自动包装为
        winsorize(ts_backfill(field, 120))
    所以实际 AST 里每个 field 节点外面都有 2 层包装。
    若把 ts_backfill 也算作 ts 算子，条件 A 永远不满足，
    is_brain_saturated 永远返回 False。

    TODO
    ----
    当前简化处理：只要 value == "ts_backfill" 就排除。
    若用户明确写了 ts_backfill(x, 20)（窗口不是 120），
    这是用户意图，理论上应算作 ts 算子，但当前不区分。
    """
    if node is None:
        return 0
    return sum(
        1
        for n in collect_nodes(node)
        if n.kind == "ts" and n.value != "ts_backfill"
    )


def _count_fundamental_fields(node: Optional[Node]) -> int:
    """统计 AST 里 value 属于基本面字段池的 field 节点数量。"""
    if node is None:
        return 0
    pool = set(_FUNDAMENTAL_FIELD_POOL)
    return sum(
        1
        for n in collect_nodes(node)
        if n.kind == "field" and n.value in pool
    )


def _has_price_volume_field(node: Optional[Node]) -> bool:
    """AST 里是否含任一价量字段。"""
    if node is None:
        return False
    return any(
        n.kind == "field" and n.value in _PRICE_VOLUME_FIELDS
        for n in collect_nodes(node)
    )


def is_brain_saturated(node: Optional[Node]) -> bool:
    """
    判断 AST 是否属于"纯基本面"（BRAIN 死角）。

    判定规则（2026-10-11 更新）：
        **含基本面 + 无价量 = BRAIN 死角**

        条件 A：含 >= 1 个基本面字段（_FUNDAMENTAL_FIELD_POOL）
        条件 B：不含任何价量字段
                （close/open/high/low/volume/returns/vwap）

    两个条件同时满足才返回 True。

    命中示例：
        debt/assets, rank(debt/assets),
        quantile(ts_arg_max(capex,120)),
        rank(ts_av_diff(bookvalue_ps,120)),
        rank(ts_sum(enterprise_value,5))

    放行示例（含价量字段）：
        -1*winsorize(ts_av_diff(close,15))
        rank(quantile(ts_av_diff(volume,20))
             + (-1*rank(ts_av_diff(close,20))))
        rank(ts_quantile(returns,10) + quantile(ts_arg_min(eps,120)))
        rank(close/volume)

    Parameters
    ----------
    node : Node | None
        None 时返回 False（不拦截）。

    Returns
    -------
    True  : 应被挡在 simulation 之前
    False : 放行
    """
    if node is None:
        return False

    # 条件 A：含 >= 1 个基本面字段
    if _count_fundamental_fields(node) < 1:
        return False

    # 条件 B：不含任何价量字段
    if _has_price_volume_field(node):
        return False

    return True


def get_saturation_reason(node: Optional[Node]) -> str:
    """
    返回饱和原因的简短说明，用于日志。

    非饱和时返回空字符串。
    """
    if not is_brain_saturated(node):
        return ""
    return "fundamental_field + no_price_volume"
