# brain/authentication.py
# -*- coding: utf-8 -*-

"""
BRAIN 认证。

职责边界
--------
只负责：通过 POST /authentication 建立已认证的 requests.Session。

不含：
    - Simulation 提交
    - 轮询
    - 表达式 / 进化 / 评估
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from config.settings import BrainCredentials, BrainEndpoints


logger = logging.getLogger(__name__)


class AuthenticationError(RuntimeError):
    """BRAIN 认证失败。"""


def create_authenticated_session(
    credentials: "BrainCredentials",
    endpoints: "BrainEndpoints",
    request_timeout: float = 30.0,
    max_retries: int = 3,
) -> requests.Session:
    """
    建立 BRAIN 认证会话。

    行为对齐 V3 WorldQuantBrainClient.login()：
        - POST {base_url}/authentication
        - Basic Auth
        - 期望 status_code == 201 或 response.ok
        - 失败时抛 AuthenticationError

    长跑容错
    --------
    对 SSL / 连接抖动做有限重试（最多 max_retries 次），
    指数退避 5s / 15s / 45s；全部失败才抛 AuthenticationError。
    """
    session = requests.Session()

    url = endpoints.url(endpoints.authentication)

    logger.debug("BRAIN 登录中...")

    backoffs = [5.0, 15.0, 45.0]
    last_error: Optional[Exception] = None

    for attempt in range(max(1, max_retries)):
        try:
            response = session.post(
                url,
                auth=(credentials.username, credentials.password),
                timeout=request_timeout,
            )
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
        ) as e:
            last_error = e
            logger.warning(
                "BRAIN 登录网络抖动 (attempt %d/%d): %r",
                attempt + 1, max_retries, e,
            )
            if attempt < max_retries - 1:
                delay = backoffs[min(attempt, len(backoffs) - 1)]
                time.sleep(delay)
                continue
            raise AuthenticationError(
                f"BRAIN 登录失败（网络抖动）: {e}"
            ) from e
        except Exception as e:
            raise AuthenticationError(f"BRAIN 登录失败: {e}") from e

        if response.status_code == 201 or response.ok:
            logger.debug("BRAIN 登录成功")
            return session

        raise AuthenticationError(
            "❌ BRAIN 登录失败\n"
            f"HTTP: {response.status_code}\n"
            f"{response.text}"
        )

    raise AuthenticationError(
        f"BRAIN 登录失败（重试耗尽）: {last_error}"
    )
