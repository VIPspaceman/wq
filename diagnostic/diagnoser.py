# diagnostic/diagnoser.py
# -*- coding: utf-8 -*-

"""
诊断主函数。

Step 10-Y 更新
--------------
- CANDIDATE 判定支持负方向：
    正方向：0.80 <= sharpe < 1.25  → CANDIDATE, direction_hint="positive"
    负方向：sharpe <= -0.50         → CANDIDATE, direction_hint="negative"

职责边界
--------
    diagnose(result, expression, threshold=DEFAULT_THRESHOLD)
        → DiagnosticReport

不含：
    - LLM 调用（在 agent/）
    - 网络请求
    - 修复逻辑
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from diagnostic.report import DiagnosticReport
from diagnostic.thresholds import (
    DEFAULT_THRESHOLD,
    DiagnosticThreshold,
)
from diagnostic.types import DefectType
from evaluation.official_checks import (
    all_checks_pass,
    parse_official_checks,
)


logger = logging.getLogger(__name__)


# ============================================================
# 错误文本 → DefectType
# ============================================================

def _classify_error(message: str) -> DefectType:
    if not message:
        return DefectType.UNKNOWN

    m = message.lower()

    if "incompatible unit" in m:
        return DefectType.UNITS_ERROR

    syntax_keywords = (
        "syntax",
        "parse",
        "unexpected token",
        "invalid expression",
        "unknown operator",
        "unknown field",
    )
    if any(k in m for k in syntax_keywords):
        return DefectType.SYNTAX_ERROR

    return DefectType.UNKNOWN


# ============================================================
# 主函数
# ============================================================

def diagnose(
    result: Dict[str, Any],
    expression: str,
    threshold: DiagnosticThreshold = DEFAULT_THRESHOLD,
) -> DiagnosticReport:
    """
    对 BRAIN 回测结果做结构化诊断。
    """
    report = DiagnosticReport(expression=expression)

    if not isinstance(result, dict):
        report.is_reject = True
        report.status = "INVALID"
        report.defects.append(DefectType.UNKNOWN)
        report.error_message = "result is not a dict"
        return report

    status = str(result.get("status", "") or "")
    report.status = status

    # -------- WARNING / ERROR 路径 --------
    if status in ("WARNING", "ERROR"):
        message = str(result.get("message", "") or "")
        report.error_message = message
        report.is_reject = True
        report.defects.append(_classify_error(message))
        return report

    # -------- COMPLETE 路径 --------
    is_data = result.get("is", {}) or {}
    if not isinstance(is_data, dict):
        is_data = {}

    sharpe = float(is_data.get("sharpe", 0) or 0)
    fitness = float(is_data.get("fitness", 0) or 0)
    turnover = float(is_data.get("turnover", 0) or 0)

    report.raw_metrics = {
        "sharpe": sharpe,
        "fitness": fitness,
        "turnover": turnover,
    }

    # ---- BRAIN-3：官方 checks 覆盖 ----
    # 若 result 里含 checks（is.checks 或 is_raw.checks），
    # 用官方 checks 结果覆盖本地 sharpe/fitness 判断。
    # 无 checks 数据时退回原逻辑（向后兼容）。
    #
    # TODO: 当前仅用 all_checks_pass 做整体 PASS/REJECT 判定，
    #       未把 FAIL 的 check 名映射到具体 DefectType。
    #       未来可扩展：CONCENTRATED_WEIGHT → 新 DefectType 等。
    official = parse_official_checks(result)
    if official:
        report.raw_metrics["official_checks"] = official

        if all_checks_pass(result, ignore_pending=True):
            report.is_pass = True
            return report

        # 官方 checks 未全过 → 标记 REJECT，并保留缺陷分类
        report.is_reject = True
        if sharpe < threshold.pass_sharpe:
            report.defects.append(DefectType.LOW_SHARPE)
        if fitness < threshold.pass_fitness:
            report.defects.append(DefectType.LOW_FITNESS)
        if turnover > threshold.high_turnover:
            report.defects.append(DefectType.HIGH_TURNOVER)
        if turnover < threshold.min_turnover:
            report.defects.append(DefectType.TURNOVER_TOO_LOW)
        if not report.defects:
            # 本地阈值都过但官方 checks 未过（如 CONCENTRATED_WEIGHT /
            # LOW_SUB_UNIVERSE_SHARPE / SELF_CORRELATION FAIL）
            report.defects.append(DefectType.UNKNOWN)
        return report

    # ---- PASS（无 checks 数据，退回本地阈值） ----
    # FIX-6b：新增 turnover 下限判断
    if (
        sharpe >= threshold.pass_sharpe
        and fitness >= threshold.pass_fitness
        and turnover >= threshold.min_turnover
    ):
        report.is_pass = True
        return report

    # FIX-6b：turnover 过低 → 标记缺陷（不 PASS）
    if (
        sharpe >= threshold.pass_sharpe
        and fitness >= threshold.pass_fitness
        and turnover < threshold.min_turnover
    ):
        report.is_reject = True
        report.defects.append(DefectType.TURNOVER_TOO_LOW)
        return report

    # ---- CANDIDATE：正方向 ----
    if threshold.candidate_sharpe <= sharpe < threshold.pass_sharpe:
        report.is_candidate = True
        report.direction_hint = "positive"

        if sharpe < threshold.pass_sharpe:
            report.defects.append(DefectType.LOW_SHARPE)
        if fitness < threshold.pass_fitness:
            report.defects.append(DefectType.LOW_FITNESS)
        if turnover > threshold.high_turnover:
            report.defects.append(DefectType.HIGH_TURNOVER)

        return report

    # ---- CANDIDATE：负方向（新） ----
    if sharpe <= -threshold.negative_candidate_sharpe:
        report.is_candidate = True
        report.direction_hint = "negative"

        # 负 sharpe 的 alpha 结构可能有效，只是方向反了
        report.defects.append(DefectType.LOW_SHARPE)

        # 注意：负 sharpe 时 fitness 通常也为负（数学相关），
        # 这不是独立缺陷，不自动标记 LOW_FITNESS。
        # 仅当 fitness 显著低于 PASS 阈值时才标记。
        if fitness < -threshold.pass_fitness:
            report.defects.append(DefectType.LOW_FITNESS)

        # turnover 高的话也标记
        if turnover > threshold.high_turnover:
            report.defects.append(DefectType.HIGH_TURNOVER)

        return report

    # ---- REJECT ----
    report.is_reject = True
    report.defects.append(DefectType.LOW_SHARPE)
    return report
