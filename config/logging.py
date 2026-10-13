# config/logging.py
# -*- coding: utf-8 -*-

"""
日志初始化。

职责边界
--------
只负责 logging 配置；不做业务逻辑。
调用方应在程序启动时调用 setup_logging()。

上下文注入
----------
日志会自动携带当前线程的上下文（worker / task_id / expression），
详见 utils/logging_context.py。

格式效果：
    无上下文：  [INFO] brain.authentication: ✅ 登录成功
    有上下文：  [INFO] brain.authentication[worker-0|t000001|rank(...)]: ✅ 登录成功
"""

from __future__ import annotations

import logging
import logging.config
import sys
from typing import Optional

from utils.logging_context import ContextFilter, ContextFormatter


# ============================================================
# 格式
# ============================================================

LOG_FORMAT = (
    "%(asctime)s "
    "[%(levelname)s] "
    "%(name)s%(ctx)s: "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# 初始化
# ============================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    初始化全局 logging。

    Parameters
    ----------
    level    : 控制台日志级别
    log_file : 可选，写入文件（追加模式）
    """
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": sys.stdout,
            "level": level,
        }
    }

    root_handlers = ["console"]

    if log_file:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": log_file,
            "mode": "a",
            "encoding": "utf-8",
            "level": level,
        }
        root_handlers.append("file")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                # 使用 ContextFormatter 而非标准 Formatter，
                # 以便在 record 缺 ctx 时兜底
                "()": ContextFormatter,
                "format": LOG_FORMAT,
                "datefmt": DATE_FORMAT,
            },
        },
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": root_handlers,
        },
    }

    logging.config.dictConfig(config)

    # 给 root 下所有 handler 挂 ContextFilter，
    # 保证每条 record 都带 ctx 属性。
    ctx_filter = ContextFilter()
    for h in logging.getLogger().handlers:
        h.addFilter(ctx_filter)