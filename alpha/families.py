# alpha/families.py
# -*- coding: utf-8 -*-

"""
Grammar 家族。

职责边界
--------
只负责：
    - 各家族的表达式生成函数
    - GRAMMAR_FAMILIES 注册表
    - detect_family（从表达式字符串推断家族）

设计（方案 B）
--------------
每个模板的**结构**固定，但**同类算子从池中采样**：

    - TREND 池：ts_delta / ts_av_diff / ts_sum
    - MEAN  池：ts_mean / ts_decay_linear
    - VOL   池：ts_std_dev
    - RANK  池：rank / zscore / scale

**默认降级**：
    如果当前 catalog 里池成员不可用（如未加载 API 时），
    `_pick_ts` / `_pick_unary` 会从可用成员里选；
    池里一个都没有时返回 None → 上层跳过本次生成。

**经典版锚点**：
    每个 family 保留至少 1 个 = V3 原模板的"经典版"，
    保证不加载 API 时行为完全等价于 V3，
    且加载后也有一定比例生成经典 alpha。

不含：
    - 校验（由 validator 承担）
    - 变异 / 交叉（由 evolution 承担）
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional, Tuple

from alpha.ast import (
    Node,
    binary,
    field,
    safe_field,
    ts,
    ts_binary,
    unary,
)
from alpha.operators import default_catalog


logger = logging.getLogger(__name__)


# ============================================================
# 语义算子池
# ============================================================

# ---- ts 池 ----
_TREND_TS_POOL: Tuple[str, ...] = (
    "ts_delta",       # 一阶差分
    "ts_av_diff",     # 与均值的差
    "ts_sum",         # 窗口求和
)

_MEAN_TS_POOL: Tuple[str, ...] = (
    "ts_mean",           # 简单均值
    "ts_decay_linear",   # 线性加权均值
)

_VOL_TS_POOL: Tuple[str, ...] = (
    "ts_std_dev",     # 标准差
)

# ---- unary 池 ----
# 只包含"截面变换"类算子；不包含 log / sqrt / abs / sign 等逐元素变换
_RANK_UNARY_POOL: Tuple[str, ...] = (
    "rank",
    "zscore",
    "scale",
)

# ---- 二元时序池（仅 ts_corr；ts_cov BRAIN 不支持，已移除）----
_BINARY_TS_POOL: Tuple[str, ...] = (
    "ts_corr",
)

# 基本面字段池（来自 fundamental6 dataset）
_FUNDAMENTAL_FIELD_POOL = (
    "assets",
    "capex",
    "cash",
    "cashflow",
    "cashflow_op",
    "cogs",
    "current_ratio",
    "debt",
    "ebit",
    "ebitda",
    "employee",
    "enterprise_value",
    "eps",
    "equity",
    "bookvalue_ps",
)


# ============================================================
# 采样工具
# ============================================================

def _pick_ts(pool: Tuple[str, ...]) -> Optional[str]:
    """
    从 pool 里选一个 **catalog 里实际存在** 的 ts 算子。

    池里没有可用 → 返回 None。
    """
    catalog = default_catalog()
    valid = [op for op in pool if op in catalog.ts()]
    if not valid:
        return None
    return random.choice(valid)


def _pick_unary(pool: Tuple[str, ...]) -> Optional[str]:
    """同 _pick_ts，针对 unary 算子。"""
    catalog = default_catalog()
    valid = [op for op in pool if op in catalog.unary()]
    if not valid:
        return None
    return random.choice(valid)


def _pick_fundamental_field() -> Optional[str]:
    """
    从 fundamental 字段池里随机选一个当前 catalog 里存在的。
    池里一个都没有时返回 None。
    """
    from alpha.fields import default_catalog as _fc
    catalog = _fc()
    valid = [f for f in _FUNDAMENTAL_FIELD_POOL if f in catalog]
    if not valid:
        return None
    return random.choice(valid)


def _pick_ts_binary(pool: Tuple[str, ...]) -> Optional[str]:
    """
    从 pool 里选一个 **catalog 里实际存在** 的二元时序算子。

    池里没有可用 → 返回 None。
    """
    catalog = default_catalog()
    valid = [op for op in pool if op in catalog.ts_binary()]
    if not valid:
        return None
    return random.choice(valid)


def _pick_window(candidates: Tuple[int, ...]) -> int:
    """
    从 candidates 里选一个 catalog 里存在的窗口。

    若一个都不匹配（例如远程 catalog 换了窗口集），
    降级用 candidates 本身。
    """
    catalog = default_catalog()
    ws = catalog.windows()
    valid = [w for w in candidates if w in ws]
    if not valid:
        valid = list(candidates)
    if not valid:
        return 10
    return random.choice(valid)


# ============================================================
# MOMENTUM
# ============================================================

def price_momentum() -> Optional[Node]:
    """
    随机版：<RANK>(<TREND_TS>(close, W))
    """
    window = _pick_window((3, 5, 10, 20))
    ts_op = _pick_ts(_TREND_TS_POOL)
    outer = _pick_unary(_RANK_UNARY_POOL)
    if ts_op is None or outer is None:
        return None
    return unary(outer, ts(ts_op, safe_field("close"), window))


def price_momentum_classic() -> Optional[Node]:
    """
    经典版（V3）：rank(ts_delta(close, W))
    """
    window = _pick_window((3, 5, 10, 20))
    return unary("rank", ts("ts_delta", safe_field("close"), window))


def price_momentum_ratio() -> Optional[Node]:
    """
    随机版：<RANK>(<TREND_TS>(close, W) / <VOL_TS>(returns, W2))
    """
    window = _pick_window((5, 10, 20))
    vol_w = _pick_window((10, 20))
    trend_op = _pick_ts(_TREND_TS_POOL)
    vol_op = _pick_ts(_VOL_TS_POOL)
    outer = _pick_unary(_RANK_UNARY_POOL)
    if trend_op is None or vol_op is None or outer is None:
        return None
    delta = ts(trend_op, safe_field("close"), window)
    vol = ts(vol_op, safe_field("returns"), vol_w)
    return unary(outer, binary("/", delta, vol))


# ---- T-3：新增纯价量模板 ----

def momentum_acceleration() -> Optional[Node]:
    """
    动量加速度：二阶差分。

    例：rank(ts_delta(ts_delta(close, W1), W2))

    - 内层 W1 从 (5, 10, 20) 随机
    - 外层 W2 从 (3, 5, 10) 随机
    """
    w1 = _pick_window((5, 10, 20))
    w2 = _pick_window((3, 5, 10))
    inner = ts("ts_delta", safe_field("close"), w1)
    return unary("rank", ts("ts_delta", inner, w2))


# ============================================================
# MEAN_REVERSION
# ============================================================

def price_vs_mean() -> Optional[Node]:
    """
    随机版：<RANK>(close / <MEAN_TS>(close, W))
    """
    window = _pick_window((10, 20, 30))
    mean_op = _pick_ts(_MEAN_TS_POOL)
    outer = _pick_unary(_RANK_UNARY_POOL)
    if mean_op is None or outer is None:
        return None
    return unary(
        outer,
        binary(
            "/",
            safe_field("close"),
            ts(mean_op, safe_field("close"), window),
        ),
    )


def price_vs_mean_classic() -> Optional[Node]:
    """
    经典版（V3）：rank(close / ts_mean(close, W))
    """
    window = _pick_window((10, 20, 30))
    return unary(
        "rank",
        binary(
            "/",
            safe_field("close"),
            ts("ts_mean", safe_field("close"), window),
        ),
    )


def mean_reversion() -> Optional[Node]:
    """
    随机版：<RANK>(close - <MEAN_TS>(close, W))
    """
    window = _pick_window((10, 20, 30))
    mean_op = _pick_ts(_MEAN_TS_POOL)
    outer = _pick_unary(_RANK_UNARY_POOL)
    if mean_op is None or outer is None:
        return None
    mean = ts(mean_op, safe_field("close"), window)
    deviation = binary("-", safe_field("close"), mean)
    return unary(outer, deviation)


# ============================================================
# REVERSAL
# ============================================================

def reversal() -> Optional[Node]:
    """
    随机版：rank(sign(<TREND_TS>(close, W)))

    FIX-4：去掉冗余 abs 层（sign(abs(x)) 等价 sign(x)）。
    """
    window = _pick_window((3, 5, 10))
    ts_op = _pick_ts(_TREND_TS_POOL)
    if ts_op is None:
        return None
    delta = ts(ts_op, safe_field("close"), window)
    return unary("rank", unary("sign", delta))


def reversal_classic() -> Optional[Node]:
    """
    经典版（V3）：rank(sign(ts_delta(close, W)))

    FIX-4：去掉冗余 abs 层。
    """
    window = _pick_window((3, 5, 10))
    delta = ts("ts_delta", safe_field("close"), window)
    return unary("rank", unary("sign", delta))


# ============================================================
# VOLUME
# ============================================================

def volume_activity() -> Optional[Node]:
    """
    随机版：<RANK>(volume / <MEAN_TS>(volume, W))
    """
    window = _pick_window((10, 20, 30))
    mean_op = _pick_ts(_MEAN_TS_POOL)
    outer = _pick_unary(_RANK_UNARY_POOL)
    if mean_op is None or outer is None:
        return None
    ratio = binary(
        "/",
        safe_field("volume"),
        ts(mean_op, safe_field("volume"), window),
    )
    return unary(outer, ratio)


def volume_activity_classic() -> Optional[Node]:
    """
    经典版（V3）：rank(volume / ts_mean(volume, W))
    """
    window = _pick_window((10, 20, 30))
    return unary(
        "rank",
        binary(
            "/",
            safe_field("volume"),
            ts("ts_mean", safe_field("volume"), window),
        ),
    )


def volume_trend() -> Optional[Node]:
    """
    随机版：<RANK>(<TREND_TS>(volume, W))
    """
    window = _pick_window((5, 10, 20))
    ts_op = _pick_ts(_TREND_TS_POOL)
    outer = _pick_unary(_RANK_UNARY_POOL)
    if ts_op is None or outer is None:
        return None
    delta = ts(ts_op, safe_field("volume"), window)
    return unary(outer, delta)


def price_volume_corr() -> Optional[Node]:
    """
    价量背离：<RANK>(<BINARY_TS>(close, volume, W))

    例：rank(ts_corr(close, volume, W))

    - 二元时序算子从 _BINARY_TS_POOL 随机选（仅 ts_corr）
    - W 从 (10, 20, 30) 随机
    """
    outer = _pick_unary(_RANK_UNARY_POOL)
    ts_op = _pick_ts_binary(_BINARY_TS_POOL)
    if outer is None or ts_op is None:
        return None

    window = _pick_window((10, 20, 30))
    corr = ts_binary(
        ts_op,
        safe_field("close"),
        safe_field("volume"),
        window,
    )
    return unary(outer, corr)


# ============================================================
# VOLATILITY
# ============================================================

def volatility_signal() -> Optional[Node]:
    """
    随机版：<RANK>(<VOL_TS>(returns, W))
    """
    window = _pick_window((10, 20, 30))
    vol_op = _pick_ts(_VOL_TS_POOL)
    outer = _pick_unary(_RANK_UNARY_POOL)
    if vol_op is None or outer is None:
        return None
    volatility = ts(vol_op, safe_field("returns"), window)
    return unary(outer, volatility)


def volatility_signal_classic() -> Optional[Node]:
    """
    经典版（V3）：rank(ts_std_dev(returns, W))
    """
    window = _pick_window((10, 20, 30))
    return unary("rank", ts("ts_std_dev", safe_field("returns"), window))


def low_volatility() -> Optional[Node]:
    """
    随机版：<RANK>(close) - <RANK>(<VOL_TS>(returns, W))

    FIX-3：原实现为 close - rank(...)，price 类型 - rank 类型
    会被 validator 拒绝（infer_type 返回 None）→ 死模板。
    改为 rank(close) - rank(...)，左右同为 rank 类型。
    """
    window = _pick_window((10, 20, 30))
    vol_op = _pick_ts(_VOL_TS_POOL)
    outer = _pick_unary(_RANK_UNARY_POOL)
    if vol_op is None or outer is None:
        return None
    volatility = ts(vol_op, field("returns"), window)
    return binary(
        "-",
        unary(outer, field("close")),
        unary(outer, volatility),
    )


def vol_of_vol() -> Optional[Node]:
    """
    波动率的波动。

    例：rank(ts_std_dev(ts_std_dev(returns, W1), W2))

    - 内层 W1 从 (10, 20, 30) 随机
    - 外层 W2 从 (5, 10, 20) 随机
    """
    w1 = _pick_window((10, 20, 30))
    w2 = _pick_window((5, 10, 20))
    inner = ts("ts_std_dev", safe_field("returns"), w1)
    return unary("rank", ts("ts_std_dev", inner, w2))


# ============================================================
# FUNDAMENTAL
# ============================================================

# ---- 混合模板（基本面 + 价量）----
# T-1：已删除 3 个纯基本面模板
#   fundamental_momentum / fundamental_mean_reversion / fundamental_ratio
#   原因：纯基本面结构 100% 被 BRAIN 拒（self-corr / sub-universe），
#         且被 alpha/saturation.py 的 is_brain_saturated 挡在 simulation 之前。
#         历史 BRAIN PASS 的 5 条里 4 条是跨字段组合。
# 背景：纯基本面结构 100% 被 BRAIN 拒（self-corr / sub-universe），
#       含价量字段的结构 PASS 率 60%+。
#       以下 3 个模板保证含 >= 1 个价量字段，避免被 saturation 挡。

# 混合模板可用的价量字段（不含 returns，returns 单独用于 vol 场景）
_HYBRID_PRICE_VOLUME_POOL: Tuple[str, ...] = (
    "close",
    "volume",
)


def fundamental_price_ratio() -> Optional[Node]:
    """
    基本面 / 价量 比值：<RANK>(field / <MEAN_TS>(pv, W))

    例：rank(field / ts_mean(close, W))

    - field 从 _FUNDAMENTAL_FIELD_POOL 随机选
    - pv 从 close / volume 随机选
    - W 从 (5, 10, 20, 30) 随机
    """
    outer = _pick_unary(_RANK_UNARY_POOL)
    mean_op = _pick_ts(_MEAN_TS_POOL)
    fname = _pick_fundamental_field()
    if outer is None or mean_op is None or fname is None:
        return None

    pv = random.choice(_HYBRID_PRICE_VOLUME_POOL)
    window = _pick_window((5, 10, 20, 30))

    ratio = binary(
        "/",
        safe_field(fname),
        ts(mean_op, safe_field(pv), window),
    )
    return unary(outer, ratio)


def fundamental_volume_momentum() -> Optional[Node]:
    """
    基本面动量 × 价量动量：<RANK>(<TREND_TS>(field, W1)
                                    * <TREND_TS>(volume, W2))

    例：rank(ts_delta(field, W1) * ts_delta(volume, W2))

    - field 从 _FUNDAMENTAL_FIELD_POOL 随机选
    - 两边窗口独立随机
    """
    outer = _pick_unary(_RANK_UNARY_POOL)
    ts_op = _pick_ts(_TREND_TS_POOL)
    fname = _pick_fundamental_field()
    if outer is None or ts_op is None or fname is None:
        return None

    w1 = _pick_window((3, 5, 10, 20))
    w2 = _pick_window((3, 5, 10, 20))

    fund_mom = ts(ts_op, safe_field(fname), w1)
    vol_mom = ts(ts_op, safe_field("volume"), w2)
    product = binary("*", fund_mom, vol_mom)
    return unary(outer, product)


def fundamental_mean_hybrid() -> Optional[Node]:
    """
    基本面均值回归 / 价量波动：<RANK>((field - <MEAN_TS>(field, W1))
                                       / <VOL_TS>(returns, W2))

    例：rank((field - ts_mean(field, W1)) / ts_std_dev(returns, W2))

    - field 从 _FUNDAMENTAL_FIELD_POOL 随机选
    - 分母固定用 returns 的波动率（价量字段）
    """
    outer = _pick_unary(_RANK_UNARY_POOL)
    mean_op = _pick_ts(_MEAN_TS_POOL)
    vol_op = _pick_ts(_VOL_TS_POOL)
    fname = _pick_fundamental_field()
    if (
        outer is None
        or mean_op is None
        or vol_op is None
        or fname is None
    ):
        return None

    w1 = _pick_window((10, 20, 30))
    w2 = _pick_window((10, 20, 30))

    mean = ts(mean_op, safe_field(fname), w1)
    deviation = binary("-", safe_field(fname), mean)
    vol = ts(vol_op, safe_field("returns"), w2)
    ratio = binary("/", deviation, vol)
    return unary(outer, ratio)


# ---- T-2：新增混合模板 ----

def fundamental_multi_hybrid() -> Optional[Node]:
    """
    基本面 × 价量 × 时间三重组合。

    例：rank((fund / ts_mean(volume, W1)) * ts_delta(close, W2))

    - fund 从 _FUNDAMENTAL_FIELD_POOL 随机选
    - 分母用 volume 的均值（价量字段）
    - 乘子用 close 的差分（价量字段）
    - 保证含 >= 1 个价量字段，避免被 saturation 挡
    """
    fund = _pick_fundamental_field()
    if fund is None:
        return None
    w1 = _pick_window((10, 20, 30))
    w2 = _pick_window((5, 10, 20))
    # 内层先 rank 到 "rank" 类型，避免 fund/volume 得到 "ratio"
    # 与 price 相乘时类型不兼容（infer_type 返回 None）。
    inner = unary(
        "rank",
        binary(
            "/",
            safe_field(fund),
            ts("ts_mean", safe_field("volume"), w1),
        ),
    )
    trend = ts("ts_delta", safe_field("close"), w2)
    # rank * price → price（* 分支：left=="rank" 返回 right_type）
    return unary("rank", binary("*", inner, trend))


def fundamental_corr_hybrid() -> Optional[Node]:
    """
    基本面 × ts_corr 组合。

    例：rank(ts_corr(rank(fund), rank(close), W))

    - fund 从 _FUNDAMENTAL_FIELD_POOL 随机选
    - 两侧先 rank 到同类型（rank），再 ts_corr
    - 含 close（价量字段），避免被 saturation 挡
    """
    fund = _pick_fundamental_field()
    if fund is None:
        return None
    w = _pick_window((10, 20, 30))
    a = unary("rank", safe_field(fund))
    b = unary("rank", safe_field("close"))
    return unary("rank", ts_binary("ts_corr", a, b, w))


# ============================================================
# 家族注册表
# ============================================================

GRAMMAR_FAMILIES = {
    "MOMENTUM": [
        price_momentum,
        price_momentum_classic,
        price_momentum_ratio,
        # T-3：动量加速度（二阶差分）
        momentum_acceleration,
    ],
    "MEAN_REVERSION": [
        price_vs_mean,
        price_vs_mean_classic,
        mean_reversion,
    ],
    "REVERSAL": [
        reversal,
        reversal_classic,
    ],
    "VOLUME": [
        volume_activity,
        volume_activity_classic,
        volume_trend,
        # 价量背离（ts_corr）
        price_volume_corr,
    ],
    "VOLATILITY": [
        volatility_signal,
        volatility_signal_classic,
        low_volatility,
        # T-3：波动率的波动
        vol_of_vol,
    ],
    "FUNDAMENTAL": [
        # T-1：已删除 3 个纯基本面模板
        #   fundamental_momentum / fundamental_mean_reversion /
        #   fundamental_ratio
        # 混合模板（基本面 + 价量）
        fundamental_price_ratio,
        fundamental_volume_momentum,
        fundamental_mean_hybrid,
        # T-2：新增混合模板
        fundamental_multi_hybrid,
        fundamental_corr_hybrid,
    ],
}


# ============================================================
# 家族推断（不变）
# ============================================================

# 预编译 FUNDAMENTAL 字段匹配正则（FIX-2）
# 用 (?<![A-Za-z0-9_]) / (?![A-Za-z0-9_]) 而非 \b，
# 因为字段名含下划线，\b 对下划线不生效。
_FUNDAMENTAL_FIELD_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(f) for f in _FUNDAMENTAL_FIELD_POOL)
    + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def detect_family(expr: str) -> str:
    e = expr.lower()

    # -------- FIX-2：FUNDAMENTAL 优先 --------
    # 纯字段优先：只要含任一基本面字段，立即归 FUNDAMENTAL。
    m = _FUNDAMENTAL_FIELD_RE.search(expr)
    if m:
        logger.debug(
            "detect_family: FUNDAMENTAL 命中字段 %r | expr=%s",
            m.group(0), expr[:80],
        )
        return "FUNDAMENTAL"

    if "ts_delta(close" in e or "close/ts_mean(close" in e:
        return "MOMENTUM"

    if "ts_mean(close" in e or "close-ts_mean(close" in e:
        return "MEAN_REVERSION"

    if "volume" in e and "close" in e:
        return "PRICE_VOLUME"

    if "volume" in e:
        return "VOLUME"

    if "ts_std_dev" in e:
        return "VOLATILITY"

    if "returns" in e:
        return "RETURN"

    return "OTHER"
