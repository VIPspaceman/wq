# notification/events.py
# -*- coding: utf-8 -*-

"""
把 Coordinator 的事件转成通知消息。

Step 10d 更新
-------------
- 新增 4 个 AI_REFINE_* 事件的消息格式
"""

from __future__ import annotations

import logging
from typing import Optional

from evaluation.record import AlphaRecord
from notification.notifier import Notifier
from search.events import EventType


logger = logging.getLogger(__name__)


# ============================================================
# 格式化辅助
# ============================================================

def _fmt_record(record: AlphaRecord, title: str) -> str:
    tags = f"tags     : {','.join(record.tags)}" if record.tags else ""
    lines = [
        title,
        f"expr     : {record.expression}",
        f"family   : {record.family}",
        f"alpha_id : {record.alpha_id}",
        f"sharpe   : {record.sharpe:+.4f}",
        f"fitness  : {record.fitness:+.4f}",
        f"turnover : {record.turnover:.4f}",
        f"score    : {record.score:+.4f}",
        f"class    : {record.classification.value}",
    ]
    if tags:
        lines.append(tags)
    return "\n".join(lines)


# ============================================================
# EventNotifier
# ============================================================

class EventNotifier:
    """
    可作为 Coordinator.on_event 传入。
    """

    def __init__(
        self,
        notifier: Notifier,
        verbose: bool = False,
    ) -> None:
        self._notifier = notifier
        self._verbose = verbose

    def __call__(
        self, event: EventType, payload: dict,
    ) -> None:
        try:
            message = self._format(event, payload)
        except Exception:
            logger.exception("格式化事件失败: %s", event)
            return

        if message is None:
            return

        try:
            self._notifier.send(message)
        except Exception:
            logger.exception("Notifier.send 失败")

    # --------------------------------------------------------
    # 格式化
    # --------------------------------------------------------

    def _format(
        self, event: EventType, payload: dict,
    ) -> Optional[str]:
        # ---------------- 生命周期 ----------------
        if event == EventType.PROGRAM_STARTED:
            agent_enabled = payload.get("agent_enabled", False)
            return (
                "🚀 Mining 开始\n"
                f"population_size = {payload.get('population_size')}\n"
                f"generations     = {payload.get('generations')}\n"
                f"random_seed     = {payload.get('random_seed')}\n"
                f"agent           = "
                f"{'ENABLED' if agent_enabled else 'disabled'}"
            )

        if event == EventType.GENERATION_STARTED:
            if not self._verbose:
                return None
            return (
                f"⚙️ Generation {payload.get('generation')} 开始\n"
                f"population = {payload.get('population_size')}"
            )

        if event == EventType.GENERATION_COMPLETED:
            lines = [
                f"✅ Generation {payload.get('generation')} 完成",
                f"submitted = {payload.get('submitted')}",
                f"completed = {payload.get('completed')}",
                f"failed    = {payload.get('failed')}",
                f"skipped   = {payload.get('skipped')}",
                f"archive   = {payload.get('archive_size')}",
            ]
            # 有 AI 修复时附上统计
            refine_submitted = payload.get("refine_submitted", 0)
            if refine_submitted:
                lines.append(
                    f"ai refine : "
                    f"submitted={refine_submitted}, "
                    f"succeeded={payload.get('refine_succeeded')}, "
                    f"failed={payload.get('refine_failed')}, "
                    f"exhausted={payload.get('refine_exhausted')}"
                )
            return "\n".join(lines)

        # ---------------- 单条 Alpha ----------------
        if event == EventType.CANDIDATE_FOUND:
            if not self._verbose:
                return None
            record = payload.get("record")
            if record is None:
                return None
            return _fmt_record(record, "🎯 CANDIDATE FOUND")

        if event == EventType.STRONG_ALPHA_FOUND:
            record = payload.get("record")
            if record is None:
                return None
            return _fmt_record(record, "💎 STRONG ALPHA")

        if event == EventType.FINAL_ALPHA_FOUND:
            record = payload.get("record")
            if record is None:
                return None
            return _fmt_record(record, "🏆 FINAL ALPHA")

        if event == EventType.SIMULATION_ERROR:
            return (
                "⚠️ Simulation 失败\n"
                f"task    = {payload.get('task_id')}\n"
                f"expr    = {payload.get('expression')}\n"
                f"error   = {payload.get('error')}"
            )

        # ---------------- Step 10d：AI Refine ----------------
        if event == EventType.AI_REFINE_STARTED:
            if not self._verbose:
                return None
            defects = payload.get("defects", [])
            return (
                "🤖 AI Refine 开始\n"
                f"expr    = {payload.get('original_expression')}\n"
                f"defects = {','.join(defects) if defects else '-'}"
            )

        if event == EventType.AI_REFINE_SUCCEEDED:
            is_pass = payload.get("is_pass", False)
            if is_pass:
                head = "🎉 AI FIXED (now PASS)"
            else:
                head = "🛠️ AI Refined (not yet PASS)"
            return (
                f"{head}\n"
                f"expr     = {payload.get('expression')}\n"
                f"alpha_id = {payload.get('alpha_id')}\n"
                f"sharpe   = {payload.get('sharpe'):+.4f}\n"
                f"fitness  = {payload.get('fitness'):+.4f}\n"
                f"turnover = {payload.get('turnover'):.4f}"
            )

        if event == EventType.AI_REFINE_FAILED:
            return (
                "❌ AI Refine 失败\n"
                f"expr    = {payload.get('original_expression')}\n"
                f"outcome = {payload.get('outcome')}\n"
                f"error   = {payload.get('error')}"
            )

        if event == EventType.AI_REFINE_EXHAUSTED:
            return (
                "🛑 AI Refine 熔断\n"
                f"expr = {payload.get('original_expression')}"
            )

        # ---------------- 结束 ----------------
        if event == EventType.MINING_COMPLETED:
            counts = payload.get("counts", {}) or {}
            lines = [
                "🎉 Mining 完成",
                f"generations = {payload.get('generations')}",
                f"archive     = {payload.get('archive_size')}",
                "counts:",
            ]
            for k, v in counts.items():
                lines.append(f"  {k:10s}: {v}")
            return "\n".join(lines)

        if event == EventType.MINING_FAILED:
            return (
                "❌ Mining 失败\n"
                f"error = {payload.get('error')}"
            )

        return None