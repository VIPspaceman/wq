# agent/refine_agent.py
# -*- coding: utf-8 -*-

"""
LLM Refining Agent。

职责边界
--------
    - 接收 DiagnosticReport
    - 通过 openai SDK 调用 LLM（兼容 DeepSeek / OpenAI）
    - 清洗输出
    - 熔断控制
    - 返回 RefineResult

不含：
    - BRAIN 提交（由 search/ 层做）
    - 诊断（由 diagnostic/ 层做）
    - 持久化（由 storage/ 层做）

配置
----
api_key 从环境变量读：
    DEEPSEEK_API_KEY   （优先）
    OPENAI_API_KEY     （备用）
其余参数由 config.settings.AgentSettings 提供。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agent.cleaner import clean_llm_output, is_probably_valid_expression
from agent.failures import (
    RefineFailureTracker,
    RefineOutcome,
)
from agent.prompts import build_system_prompt, build_user_prompt
from diagnostic.report import DiagnosticReport


logger = logging.getLogger(__name__)


# 从项目根目录加载 .env（不覆盖已存在的系统环境变量）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ============================================================
# 结果
# ============================================================

@dataclass
class RefineResult:
    success: bool
    original_expression: str
    refined_expression: Optional[str] = None
    outcome: RefineOutcome = RefineOutcome.SKIPPED
    error: Optional[str] = None
    raw_llm_output: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "original_expression": self.original_expression,
            "refined_expression": self.refined_expression,
            "outcome": self.outcome.value,
            "error": self.error,
        }


# ============================================================
# 主类
# ============================================================

class LLMRefiningAgent:
    """
    单次修复的 Agent。

    Attributes
    ----------
    api_key     : LLM API Key（缺省时从环境变量读）
    base_url    : OpenAI 兼容 API base（DeepSeek 用 https://api.deepseek.com）
    model       : 模型名（DeepSeek 用 deepseek-chat 或 deepseek-reasoner）
    timeout     : 单次请求超时（秒）
    max_retries : SDK 层重试次数（针对 5xx / 网络错误）
    tracker     : 熔断计数器（可外部注入，用于跨 agent 共享状态）
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 30.0,
        max_retries: int = 2,
        max_attempts_per_expression: int = 3,
        tracker: Optional[RefineFailureTracker] = None,
    ) -> None:
        # ---- API key 解析 ----
        resolved_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY_WQ")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY_WQ")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not resolved_key:
            raise RuntimeError(
                "缺少 LLM API Key。请设置环境变量：\n"
                "  DEEPSEEK_API_KEY_WQ（推荐）\n"
                "  或 DEEPSEEK_API_KEY / OPENAI_API_KEY"
            )

        self._api_key = resolved_key
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        self._tracker = tracker or RefineFailureTracker(
            max_attempts=max_attempts_per_expression,
        )

        # 延迟导入（避免 smoke test 在没有 openai 时挂掉）
        self._client = None

    # --------------------------------------------------------
    # SDK 初始化
    # --------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "缺少 openai SDK。请运行：\n"
                    "  pip install openai"
                ) from e
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._client

    # --------------------------------------------------------
    # 公开属性
    # --------------------------------------------------------

    @property
    def tracker(self) -> RefineFailureTracker:
        return self._tracker

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def refine(self, report: DiagnosticReport) -> RefineResult:
        """
        同步调用 LLM 修复一个 CANDIDATE。

        失败不抛异常，通过 RefineResult 返回原因。
        """
        original = report.expression

        # -------- 熔断：超过上限直接跳过 --------
        if not self._tracker.can_attempt(original):
            self._tracker.record(original, RefineOutcome.EXHAUSTED)
            return RefineResult(
                success=False,
                original_expression=original,
                outcome=RefineOutcome.EXHAUSTED,
                error="refine attempts exhausted",
            )

        # -------- 请求 LLM --------
        raw = None
        try:
            raw = self._call_llm(report)
        except Exception as e:
            self._tracker.record(original, RefineOutcome.LLM_FAILED)
            logger.warning(
                "LLM 调用失败: %r | expr=%s",
                e, original[:60],
            )
            return RefineResult(
                success=False,
                original_expression=original,
                outcome=RefineOutcome.LLM_FAILED,
                error=repr(e),
            )

        # -------- 清洗 --------
        cleaned = clean_llm_output(raw)
        if not cleaned or not is_probably_valid_expression(cleaned):
            self._tracker.record(original, RefineOutcome.INVALID_OUTPUT)
            logger.warning(
                "LLM 输出无法清洗为有效表达式 | raw=%r", raw,
            )
            return RefineResult(
                success=False,
                original_expression=original,
                outcome=RefineOutcome.INVALID_OUTPUT,
                error="empty or malformed LLM output",
                raw_llm_output=raw,
            )

        # -------- 成功 --------
        self._tracker.record(original, RefineOutcome.SUCCESS)
        logger.info(
            "LLM refine OK: %s → %s",
            original[:50], cleaned[:80],
        )
        return RefineResult(
            success=True,
            original_expression=original,
            refined_expression=cleaned,
            outcome=RefineOutcome.SUCCESS,
            raw_llm_output=raw,
        )

    # --------------------------------------------------------
    # LLM 调用（同步）
    # --------------------------------------------------------

    def _call_llm(self, report: DiagnosticReport) -> str:
        client = self._get_client()

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(report)

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=256,
        )

        if not response or not response.choices:
            raise RuntimeError("LLM 返回为空")

        message = response.choices[0].message
        if message is None:
            raise RuntimeError("LLM 返回 message 为 None")

        content = message.content or ""
        if not content.strip():
            logger.warning(
                "LLM 返回空内容 | finish_reason=%s",
                getattr(response.choices[0], "finish_reason", "?"),
            )
        return content
