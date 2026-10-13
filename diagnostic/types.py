# diagnostic/types.py
# -*- coding: utf-8 -*-

"""
诊断类型枚举。

职责边界
--------
只定义枚举，不含任何逻辑。

DefectType
----------
    NONE          : 无缺陷
    LOW_SHARPE    : Sharpe 低于 PASS 阈值但可能可修复
    LOW_FITNESS   : Fitness 低于 PASS 阈值
    HIGH_TURNOVER : Turnover 高于上限
    UNITS_ERROR   : BRAIN 报单位不兼容
    SYNTAX_ERROR  : 其它语法/语义错误
    UNKNOWN       : 无法归类

Severity
--------
    INFO     : 提示
    WARNING  : 可尝试修复
    ERROR    : 不可修复，应丢弃
"""

from __future__ import annotations

from enum import Enum


class DefectType(str, Enum):
    NONE = "NONE"
    LOW_SHARPE = "LOW_SHARPE"
    LOW_FITNESS = "LOW_FITNESS"
    HIGH_TURNOVER = "HIGH_TURNOVER"
    TURNOVER_TOO_LOW = "TURNOVER_TOO_LOW"   # FIX-6b
    UNITS_ERROR = "UNITS_ERROR"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# 每种缺陷的严重程度（用于上层决定是否值得修复）
DEFECT_SEVERITY = {
    DefectType.NONE: Severity.INFO,
    DefectType.LOW_SHARPE: Severity.WARNING,
    DefectType.LOW_FITNESS: Severity.WARNING,
    DefectType.HIGH_TURNOVER: Severity.WARNING,
    # FIX-6b：与 HIGH_TURNOVER 语义对齐（同为 WARNING）
    DefectType.TURNOVER_TOO_LOW: Severity.WARNING,
    DefectType.UNITS_ERROR: Severity.ERROR,   # 单位错误本质是结构问题
    DefectType.SYNTAX_ERROR: Severity.ERROR,
    DefectType.UNKNOWN: Severity.ERROR,
}
