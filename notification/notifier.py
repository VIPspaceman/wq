# notification/notifier.py
# -*- coding: utf-8 -*-

"""
Notifier 抽象。

职责边界
--------
只定义接口。任何实现的 send() **不应** 抛异常；
发送失败应内部记录并返回。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Notifier(ABC):
    @abstractmethod
    def send(self, text: str) -> None:
        """
        发送一条纯文本消息。

        约定：
            - 失败时内部记录，不抛异常
            - 实现方负责长度截断、重试等策略
        """