# diagnostic/report.py
# -*- coding: utf-8 -*-

"""
DiagnosticReport：单个 Alpha 的诊断结果。

Step 10-Y 更新
--------------
新增 `direction_hint` 字段：
    ""           — 无方向提示
    "positive"   — 正 sharpe，正常修复
    "negative"   — 负 sharpe，应尝试取反
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from diagnostic.types import (
    DEFECT_SEVERITY,
    DefectType,
    Severity,
)


@dataclass
class DiagnosticReport:
    expression: str

    is_pass: bool = False
    is_candidate: bool = False
    is_reject: bool = False

    defects: List[DefectType] = field(default_factory=list)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    status: str = ""

    # Step 10-Y：方向提示
    # ""           — 无提示
    # "positive"   — 正 sharpe CANDIDATE
    # "negative"   — 负 sharpe CANDIDATE，应尝试取反
    direction_hint: str = ""

    # --------------------------------------------------------
    # 便捷属性
    # --------------------------------------------------------

    @property
    def needs_refine(self) -> bool:
        """
        是否需要触发 AI 修复。

        条件：
            - 是 CANDIDATE
            - 至少有一个缺陷
            - 所有缺陷的严重度 <= WARNING（ERROR 级无法修复）
        """
        if not self.is_candidate or not self.defects:
            return False
        for d in self.defects:
            sev = DEFECT_SEVERITY.get(d, Severity.ERROR)
            if sev == Severity.ERROR:
                return False
        return True

    @property
    def max_severity(self) -> Severity:
        if not self.defects:
            return Severity.INFO
        severities = [
            DEFECT_SEVERITY.get(d, Severity.ERROR)
            for d in self.defects
        ]
        if Severity.ERROR in severities:
            return Severity.ERROR
        if Severity.WARNING in severities:
            return Severity.WARNING
        return Severity.INFO

    @property
    def sharpe(self) -> float:
        return float(self.raw_metrics.get("sharpe", 0) or 0)

    @property
    def fitness(self) -> float:
        return float(self.raw_metrics.get("fitness", 0) or 0)

    @property
    def turnover(self) -> float:
        return float(self.raw_metrics.get("turnover", 0) or 0)

    @property
    def is_negative_direction(self) -> bool:
        return self.direction_hint == "negative"

    # --------------------------------------------------------
    # 序列化
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["defects"] = [x.value for x in self.defects]
        return d

    def summary(self) -> str:
        """单行摘要，用于日志。"""
        if self.is_pass:
            tag = "PASS"
        elif self.is_candidate:
            tag = "CANDIDATE"
        else:
            tag = "REJECT"
        defects = ",".join(d.value for d in self.defects) or "-"
        hint = (
            f" dir={self.direction_hint}"
            if self.direction_hint else ""
        )
        return (
            f"[{tag:9s}] "
            f"sharpe={self.sharpe:+.4f} "
            f"fitness={self.fitness:+.4f} "
            f"turnover={self.turnover:.4f} "
            f"defects={defects}{hint} | "
            f"{self.expression}"
        )