# notification/__init__.py
# -*- coding: utf-8 -*-

"""
notification：通知层。

子模块：
    notifier : Notifier 抽象
    telegram : Telegram Bot 实现
    events   : EventNotifier（EventType → 消息）

⚠️ 本层只做"发送消息"，不做任何业务逻辑。
   通知失败绝不能导致 Mining 主流程崩溃。
"""

from notification.events import EventNotifier
from notification.notifier import Notifier
from notification.telegram import TelegramNotifier

__all__ = [
    "Notifier",
    "TelegramNotifier",
    "EventNotifier",
]