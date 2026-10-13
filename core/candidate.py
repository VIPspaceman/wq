# -*- coding: utf-8 -*-

"""
AlphaCandidate：系统唯一的候选对象。

职责边界
--------
定义候选 Alpha 的 **唯一** 数据结构，贯穿：

    generated → filtered → submitted → evaluated → classified

设计动机
--------
在引入 core/ 之前，候选对象在不同阶段用不同结构表示：

    - 生成阶段 : alpha.ast.Node
    - 执行阶段 : execution.task.Task
    - 评估阶段 : evaluation.record.AlphaRecord
    - 诊断阶段 : diagnostic.report.DiagnosticReport

阶段之间靠手工字段搬运，容易丢字段 / 不一致。
AlphaCandidate 是贯穿全流程的统一载体。

兼容层
------
提供 from_record() / to_record() 与现有 AlphaRecord 互转，
使现有模块（evaluation/、storage/）无需改动即可逐步迁移。

不含：
    - 表达式解析（延迟到需要时，由调用方 parse）
    - 评估逻辑（在 evaluation/）
    - 持久化（在 storage/）
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from alpha.ast import Node
from core.hashing import expression_hash
from core.signature import structure_signature


# ============================================================
# 常量
# ============================================================

# stage 取值
STAGE_GENERATED = "generated"
STAGE_FILTERED = "filtered"
STAGE_SUBMITTED = "submitted"
STAGE_EVALUATED = "evaluated"
STAGE_CLASSIFIED = "classified"

# status 取值（与 evaluation.classifier.Classification 对齐）
STATUS_PENDING = "PENDING"
STATUS_FINAL = "FINAL"
STATUS_STRONG = "STRONG"
STATUS_CANDIDATE = "CANDIDATE"
STATUS_REJECT = "REJECT"
STATUS_NEGATIVE = "NEGATIVE"


# ============================================================
# AlphaCandidate
# ============================================================

@dataclass
class AlphaCandidate:
    """
    系统唯一的候选对象。

    Attributes
    ----------
    身份
        candidate_id   : 唯一 ID（默认用 expr_hash）
        expression     : 表达式字符串
        ast            : 解析后的 AST（可延迟；None 表示未解析）
        expr_hash      : expression_hash(expression)
        structure_sig  : structure_signature(ast)

    元数据
        family         : 家族名（如 "MOMENTUM"）
        generation     : 所属代数
        parent_ids     : 父候选 ID 列表（mutation/crossover 用）
        source         : 来源（generator/mutation/refine/combine/...）

    状态
        stage          : 生命周期阶段
        status         : 分级结果
        rejection_reasons : 被拒原因列表

    评估
        alpha_id       : BRAIN 返回的 alpha ID
        sharpe / fitness / turnover / score
        official_checks: {check_name: result}
        raw_metrics    : 原始指标（保真）
        tags           : 标签（AI_FIXED / SIGN_FLIPPED / COMBINED / ...）
    """

    # ---- 身份 ----
    candidate_id: str = ""
    expression: str = ""
    ast: Optional[Node] = None
    expr_hash: str = ""
    structure_sig: str = ""

    # ---- 元数据 ----
    family: str = "UNKNOWN"
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    source: str = ""

    # ---- 状态 ----
    stage: str = STAGE_GENERATED
    status: str = STATUS_PENDING
    rejection_reasons: List[str] = field(default_factory=list)

    # ---- 评估 ----
    alpha_id: Optional[str] = None
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    score: Optional[float] = None
    official_checks: Dict[str, str] = field(default_factory=dict)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # --------------------------------------------------------
    # 构造
    # --------------------------------------------------------

    @classmethod
    def from_expression(
        cls,
        expr: str,
        *,
        parse: bool = True,
        **kwargs: Any,
    ) -> "AlphaCandidate":
        """
        从表达式构造，自动填 candidate_id / expr_hash / structure_sig。

        Parameters
        ----------
        expr : str
            表达式字符串。
        parse : bool
            True（默认）时尝试解析 AST 并计算 structure_sig。
            解析失败时 ast=None、structure_sig=""（不抛异常）。
        **kwargs
            其余字段（family / generation / source / ...）。

        Returns
        -------
        AlphaCandidate
        """
        h = expression_hash(expr)

        ast: Optional[Node] = None
        sig = ""
        if parse:
            try:
                from alpha.parser import parse_expression
                ast = parse_expression(expr)
                if ast is not None:
                    sig = structure_signature(ast)
            except Exception:
                ast = None
                sig = ""

        # candidate_id 默认用 expr_hash；调用方可显式覆盖
        candidate_id = kwargs.pop("candidate_id", "") or h

        return cls(
            candidate_id=candidate_id,
            expression=expr,
            ast=ast,
            expr_hash=h,
            structure_sig=sig,
            **kwargs,
        )

    # --------------------------------------------------------
    # 便捷方法
    # --------------------------------------------------------

    def add_tag(self, tag: str) -> None:
        """添加标签（去重）。"""
        if tag not in self.tags:
            self.tags.append(tag)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def add_rejection(self, reason: str) -> None:
        """记录被拒原因（去重）。"""
        if reason not in self.rejection_reasons:
            self.rejection_reasons.append(reason)

    def ensure_ast(self) -> Optional[Node]:
        """
        确保 ast 已解析（延迟解析）。

        返回 ast；解析失败返回 None。
        """
        if self.ast is not None:
            return self.ast
        try:
            from alpha.parser import parse_expression
            self.ast = parse_expression(self.expression)
            if self.ast is not None and not self.structure_sig:
                self.structure_sig = structure_signature(self.ast)
        except Exception:
            self.ast = None
        return self.ast

    def is_evaluated(self) -> bool:
        """是否已有评估结果。"""
        return self.sharpe is not None

    def is_candidate_or_better(self) -> bool:
        """是否为 CANDIDATE / STRONG / FINAL。"""
        return self.status in (
            STATUS_CANDIDATE, STATUS_STRONG, STATUS_FINAL,
        )

    # --------------------------------------------------------
    # 序列化
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为 dict（用于 JSON / DB 写入）。

        注意：ast 不序列化（Node 不是 JSON 友好）。
        反序列化时由 from_dict 按需重新解析。
        """
        return {
            "candidate_id": self.candidate_id,
            "expression": self.expression,
            "expr_hash": self.expr_hash,
            "structure_sig": self.structure_sig,
            "family": self.family,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "source": self.source,
            "stage": self.stage,
            "status": self.status,
            "rejection_reasons": list(self.rejection_reasons),
            "alpha_id": self.alpha_id,
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover": self.turnover,
            "score": self.score,
            "official_checks": dict(self.official_checks),
            "raw_metrics": dict(self.raw_metrics),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlphaCandidate":
        """
        从 dict 反序列化。

        容错：缺失字段用默认值；expr_hash / structure_sig 缺失时重算。
        """
        expr = str(d.get("expression") or "")

        expr_hash = str(d.get("expr_hash") or "")
        if not expr_hash and expr:
            expr_hash = expression_hash(expr)

        structure_sig = str(d.get("structure_sig") or "")

        candidate_id = str(d.get("candidate_id") or "") or expr_hash

        return cls(
            candidate_id=candidate_id,
            expression=expr,
            ast=None,  # 延迟解析
            expr_hash=expr_hash,
            structure_sig=structure_sig,
            family=str(d.get("family") or "UNKNOWN"),
            generation=int(d.get("generation") or 0),
            parent_ids=list(d.get("parent_ids") or []),
            source=str(d.get("source") or ""),
            stage=str(d.get("stage") or STAGE_GENERATED),
            status=str(d.get("status") or STATUS_PENDING),
            rejection_reasons=list(d.get("rejection_reasons") or []),
            alpha_id=d.get("alpha_id"),
            sharpe=_opt_float(d.get("sharpe")),
            fitness=_opt_float(d.get("fitness")),
            turnover=_opt_float(d.get("turnover")),
            score=_opt_float(d.get("score")),
            official_checks=dict(d.get("official_checks") or {}),
            raw_metrics=dict(d.get("raw_metrics") or {}),
            tags=list(d.get("tags") or []),
        )

    # --------------------------------------------------------
    # 兼容层：AlphaRecord <-> AlphaCandidate
    # --------------------------------------------------------

    @classmethod
    def from_record(cls, record: Any) -> "AlphaCandidate":
        """
        从现有 evaluation.record.AlphaRecord 转换。

        兼容层：使现有模块无需改动即可逐步迁移到 AlphaCandidate。

        注意：AlphaRecord.classification 是 Classification 枚举，
        这里取其 .value 作为 status。
        """
        expr = str(getattr(record, "expression", "") or "")

        classification = getattr(record, "classification", None)
        if classification is None:
            status = STATUS_PENDING
        elif hasattr(classification, "value"):
            status = str(classification.value)
        else:
            status = str(classification)

        return cls(
            candidate_id=expression_hash(expr) if expr else "",
            expression=expr,
            ast=None,
            expr_hash=expression_hash(expr) if expr else "",
            structure_sig="",
            family=str(getattr(record, "family", "UNKNOWN") or "UNKNOWN"),
            generation=0,
            parent_ids=[],
            source="",
            stage=STAGE_CLASSIFIED,
            status=status,
            rejection_reasons=[],
            alpha_id=getattr(record, "alpha_id", None) or None,
            sharpe=_opt_float(getattr(record, "sharpe", None)),
            fitness=_opt_float(getattr(record, "fitness", None)),
            turnover=_opt_float(getattr(record, "turnover", None)),
            score=_opt_float(getattr(record, "score", None)),
            official_checks={},
            raw_metrics={},
            tags=list(getattr(record, "tags", []) or []),
        )

    def to_record(self) -> Any:
        """
        转换为现有 evaluation.record.AlphaRecord。

        兼容层：使 AlphaCandidate 可喂给现有 storage/ 层。

        注意：AlphaRecord 要求 sharpe/fitness/turnover/score 为 float，
        这里 None → 0.0。
        """
        from evaluation.classifier import Classification
        from evaluation.record import AlphaRecord

        try:
            classification = Classification(self.status)
        except ValueError:
            classification = Classification.REJECT

        return AlphaRecord(
            expression=self.expression,
            alpha_id=self.alpha_id or "",
            sharpe=self.sharpe if self.sharpe is not None else 0.0,
            fitness=self.fitness if self.fitness is not None else 0.0,
            turnover=self.turnover if self.turnover is not None else 0.0,
            score=self.score if self.score is not None else 0.0,
            classification=classification,
            family=self.family,
            complexity=0,
            depth=0,
            official_checks=[],
            extra={},
            tags=list(self.tags),
        )


# ============================================================
# 内部辅助
# ============================================================

def _opt_float(value: Any) -> Optional[float]:
    """安全转 float；None / 空 / 非法 → None。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
