# notification/telegram.py
# -*- coding: utf-8 -*-

"""
Telegram Bot 通知。

依赖
----
requests

API
---
使用 Telegram Bot HTTP API（官方文档）：
    POST https://api.telegram.org/bot<TOKEN>/sendMessage

本模块 **不** 引入 python-telegram-bot 之类的重依赖；
只用 requests，保持与项目一致。

错误处理
--------
send() 不会抛异常。
HTTP 失败 / 网络异常 → 记录 warning 并返回。
"""

from __future__ import annotations

import logging

import requests

from notification.notifier import Notifier


logger = logging.getLogger(__name__)


# Telegram 文本上限 4096；留一点余量
_MAX_LEN = 4000


class TelegramNotifier(Notifier):
    _API = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout: float = 10.0,
    ) -> None:
        if not bot_token or not chat_id:
            raise ValueError(
                "TelegramNotifier 需要 bot_token 与 chat_id"
            )
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout

    # --------------------------------------------------------
    # 发送
    # --------------------------------------------------------

    def send(self, text: str) -> None:
        if not text:
            return

        if len(text) > _MAX_LEN:
            text = text[: _MAX_LEN - 12] + "\n...(截断)"

        url = f"{self._API}/bot{self._bot_token}/sendMessage"

        try:
            response = requests.post(
                url,
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
                timeout=self._timeout,
            )
        except Exception as e:
            logger.warning("Telegram 发送异常: %r", e)
            return

        if not response.ok:
            logger.warning(
                "Telegram 发送失败: HTTP %s %s",
                response.status_code,
                (response.text or "")[:200],
            )