# utils/logging_context.py
# -*- coding: utf-8 -*-

"""
日志上下文。

职责边界
--------
    - 用 contextvars 保存"当前线程正在处理什么任务"的信息
    - 提供 logging.Filter / logging.Formatter 以便注入到日志输出

设计动机
--------
并发 Simulation 时，多个 worker 同时调用 brain/* 的模块级 logger，
如果不带上下文，日志会混在一起无法区分哪条对应哪个 task。

通过 contextvars：
    - worker-0 在自己的线程 set_context(worker="worker-0", task_id="t1", ...)
    - worker-1 在自己的线程 set_context(worker="worker-1", task_id="t2", ...)
    - 主线程 / 其它线程互不影响
    - **brain/ 层代码完全不需要改动**，日志自动带上上下文

不含：
    - 业务逻辑
    - 日志级别管理（在 config/logging.py）
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Dict, Optional


# ============================================================
# ContextVar
# ============================================================

_CONTEXT: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "wq_log_context", default=None
)


# ============================================================
# 设置 / 恢复
# ============================================================

def set_log_context(**kwargs: str) -> Token:
    """
    设置当前线程/协程的日志上下文。

    返回的 Token 应传给 reset_log_context() 以恢复。
    """
    return _CONTEXT.set({k: str(v) for k, v in kwargs.items()})


def reset_log_context(token: Token) -> None:
    """恢复之前的日志上下文。"""
    _CONTEXT.reset(token)


def get_log_context() -> Optional[Dict[str, str]]:
    return _CONTEXT.get()


# ============================================================
# 格式化
# ============================================================

def format_log_context(ctx: Optional[Dict[str, str]]) -> str:
    """
    把上下文格式化为 "[worker-0|t000001|rank(...)]" 形式。

    空上下文 → 返回空字符串（不影响原有格式）。
    """
    if not ctx:
        return ""
    parts = [v for v in ctx.values() if v]
    if not parts:
        return ""
    return "[" + "|".join(parts) + "]"


# ============================================================
# logging Filter / Formatter
# ============================================================

class ContextFilter(logging.Filter):
    """
    把当前上下文注入到 LogRecord.ctx。

    挂在 handler 上，保证 record 一定带 ctx 属性。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.ctx = format_log_context(get_log_context())
        return True


class ContextFormatter(logging.Formatter):
    """
    兜底 Formatter：即使 record 没有 ctx 属性也不会抛 KeyError。

    用于以下场景：
        - 用户直接 import LOG_FORMAT 但没调 setup_logging
        - 第三方库自己创建的 logger 用了不同的 handler
    """

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "ctx"):
            record.ctx = ""
        return super().format(record)