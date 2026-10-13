# brain/models.py
# -*- coding: utf-8 -*-

"""
BRAIN 通信层的数据模型。

职责边界
--------
本模块只定义数据结构：
    - SimulationSettings  : BRAIN Simulation settings（映射到 POST /simulations）
    - SimulationRequest   : POST /simulations 的请求体
    - AlphaMetrics        : 从 /alphas/{id} 提取的指标
    - SimulationResult    : Simulation 完成后的结果

本模块不含：
    - 网络请求
    - 轮询
    - 认证
    - 进化/评估/通知/存储

关于 BRAIN payload
------------------
字段名严格对齐 V3 已验证的请求体，例如：
    instrumentType / unitHandling / nanHandling / pasteurization 等。
未在本模块中出现的 BRAIN 字段，一律视为“未验证”，不做假设。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ============================================================
# Simulation settings
# ============================================================

@dataclass
class SimulationSettings:
    """
    BRAIN REGULAR Simulation 的 settings 块。

    字段默认值与 V3 CONFIG 完全对齐；
    只做结构化，不改变语义。
    """
    instrument_type: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 4
    neutralization: str = "INDUSTRY"
    truncation: float = 0.08
    pasteurization: str = "ON"
    unit_handling: str = "VERIFY"
    nan_handling: str = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = False

    def to_brain_payload(self) -> Dict[str, Any]:
        """
        转换为 BRAIN POST /simulations 中的 settings 字段。

        注意：字段名必须是 BRAIN 侧使用的大小写，例如
            instrumentType / unitHandling / nanHandling
        而不是 Python 侧的 snake_case。
        """
        return {
            "instrumentType": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": self.truncation,
            "pasteurization": self.pasteurization,
            "unitHandling": self.unit_handling,
            "nanHandling": self.nan_handling,
            "language": self.language,
            "visualization": self.visualization,
        }


# ============================================================
# Simulation request
# ============================================================

@dataclass
class SimulationRequest:
    """
    POST /simulations 的请求体。
    """
    expression: str
    settings: SimulationSettings
    type: str = "REGULAR"

    def to_brain_payload(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "settings": self.settings.to_brain_payload(),
            "regular": self.expression,
        }


# ============================================================
# Metrics
# ============================================================

@dataclass
class AlphaMetrics:
    """
    V3 中从 /alphas/{id} 提取的核心指标。

    仅包含 V3 已确认可用的字段：
        sharpe / fitness / turnover
    """
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0


# ============================================================
# Simulation result
# ============================================================

@dataclass
class SimulationResult:
    """
    Simulation 完成后（COMPLETE + /alphas/{id} 200）返回的结果。

    Attributes
    ----------
    alpha_id : BRAIN 生成的 Alpha ID
    metrics  : 提取后的核心指标
    raw      : 原始返回（保真，供后续扩展）
    """
    alpha_id: str
    metrics: AlphaMetrics
    raw: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Simulation 状态快照（供 polling 层使用）
# ============================================================

@dataclass
class SimulationProgress:
    """
    一次 polling 的快照。

    与 V3 wait_for_simulation() 中处理的字段保持一致：
        progress / status
    status 取值（V3 已观察到）：
        "ERROR" / "WARNING" / "COMPLETE" / 其它（进行中）
    """
    status: Optional[str] = None
    progress: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)