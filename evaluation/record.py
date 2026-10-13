# evaluation/record.py
# -*- coding: utf-8 -*-

"""
AlphaRecord：一次 Simulation 的完整记录。

Step 10d 更新
-------------
- from_dict 兼容字符串 tags（CSV 反序列化场景）
  老格式: ["AI_FIXED", "AI_REFINED"]  → list
  CSV 读: "AI_FIXED,AI_REFINED"        → 按逗号拆分
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from alpha.ast import (
    Node,
    node_count,
    node_depth,
    node_to_expr,
)
from alpha.families import detect_family
from evaluation.classifier import Classification, classify_alpha
from evaluation.metrics import extract_metrics
from evaluation.official_checks import extract_official_checks
from evaluation.scoring import calculate_score


def _now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _parse_tags(raw: Any) -> List[str]:
    """
    兼容多种 tags 输入：
        - list[str]           → 直接返回
        - "a,b,c"             → 按逗号拆分
        - "" / None           → []
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


# ============================================================
# AlphaRecord
# ============================================================

@dataclass
class AlphaRecord:
    expression: str
    alpha_id: str

    sharpe: float
    fitness: float
    turnover: float

    score: float
    classification: Classification

    family: str
    complexity: int
    depth: int

    timestamp: str = field(default_factory=_now_timestamp)

    official_checks: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    tags: List[str] = field(default_factory=list)

    # --------------------------------------------------------
    # 便捷方法
    # --------------------------------------------------------

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    # --------------------------------------------------------
    # 序列化
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["classification"] = self.classification.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlphaRecord":
        return cls(
            expression=d["expression"],
            alpha_id=d.get("alpha_id", ""),
            sharpe=float(d.get("sharpe", 0.0)),
            fitness=float(d.get("fitness", 0.0)),
            turnover=float(d.get("turnover", 0.0)),
            score=float(d.get("score", 0.0)),
            classification=Classification(
                d.get("classification", "REJECT")
            ),
            family=d.get("family", "OTHER"),
            complexity=int(d.get("complexity", 0)),
            depth=int(d.get("depth", 0)),
            timestamp=d.get("timestamp", _now_timestamp()),
            official_checks=list(d.get("official_checks", [])),
            extra=dict(d.get("extra", {})),
            tags=_parse_tags(d.get("tags", [])),
        )

    def is_candidate_or_better(self) -> bool:
        return self.classification in (
            Classification.CANDIDATE,
            Classification.STRONG,
            Classification.FINAL,
        )


# ============================================================
# build_record
# ============================================================

def build_record(
    node: Node,
    result: Dict[str, Any],
    *,
    avg_corr: float = 0.0,
) -> AlphaRecord:
    expression = node_to_expr(node)

    metrics = extract_metrics(result)

    complexity = node_count(node)
    depth = node_depth(node)

    family = node.family or detect_family(expression)

    score = calculate_score(
        metrics.sharpe,
        metrics.fitness,
        metrics.turnover,
        complexity,
        avg_corr=avg_corr,
    )

    classification = classify_alpha(
        metrics.sharpe,
        metrics.fitness,
        metrics.turnover,
    )

    checks = extract_official_checks(result)

    return AlphaRecord(
        expression=expression,
        alpha_id=str(result.get("_alpha_id", "")),
        sharpe=metrics.sharpe,
        fitness=metrics.fitness,
        turnover=metrics.turnover,
        score=score,
        classification=classification,
        family=family,
        complexity=complexity,
        depth=depth,
        timestamp=_now_timestamp(),
        official_checks=checks.to_list(),
        extra={"is_raw": result.get("is", {})},
        tags=[],
    )
