# diagnostic/__init__.py
# -*- coding: utf-8 -*-

"""
diagnostic：Alpha 回测结果诊断层。

职责边界
--------
    - 接收 BRAIN 回测返回的 result dict
    - 输出结构化 DiagnosticReport

    不依赖：
        - brain/        （诊断只处理结果，不发起请求）
        - execution/    （不感知并发）
        - search/       （不参与编排）
        - agent/        （agent 依赖本层，反之不行）

子模块：
    types       : DefectType / Severity 枚举
    thresholds  : DiagnosticThreshold 数据类
    report      : DiagnosticReport
    diagnoser   : diagnose() 主函数

典型用法
--------
    from diagnostic import diagnose

    report = diagnose(result, expression)
    if report.is_pass:
        ...         # PASS
    elif report.needs_refine:
        ...         # 触发 AI 修复
"""

from diagnostic.diagnoser import diagnose
from diagnostic.report import DiagnosticReport
from diagnostic.thresholds import DEFAULT_THRESHOLD, DiagnosticThreshold
from diagnostic.types import DefectType, Severity

__all__ = [
    "diagnose",
    "DiagnosticReport",
    "DiagnosticThreshold",
    "DEFAULT_THRESHOLD",
    "DefectType",
    "Severity",
]