# brain/client.py
# -*- coding: utf-8 -*-

"""
BRAIN HTTP 客户端。

职责边界
--------
只负责：
    - 携带已认证 Session
    - GET with retry（对齐 V3 get_with_retry）
    - POST simulation（对齐 V3 simulate_alpha 的 POST 阶段，不含轮询）

不含：
    - 轮询逻辑（在 polling.py）
    - Simulation 编排（在 simulation.py）
    - Location header 解析（在 simulation.py）
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import requests

if TYPE_CHECKING:
    from config.settings import BrainEndpoints, PollingSettings


logger = logging.getLogger(__name__)


class BrainClient:
    """
    对 BRAIN REST API 的薄封装。

    该类不关心 payload 结构，也不关心 Simulation 状态机；
    仅提供带重试的 HTTP 调用。
    """

    def __init__(
        self,
        session: requests.Session,
        endpoints: "BrainEndpoints",
        polling: "PollingSettings",
    ) -> None:
        self._session = session
        self._endpoints = endpoints
        self._polling = polling

    # --------------------------------------------------------
    # GET with retry
    # --------------------------------------------------------

    def get_with_retry(
        self,
        url: str,
    ) -> Optional[requests.Response]:
        """
        与 V3 get_with_retry() 行为一致：
            - 重试 http_retry + 1 次
            - Timeout / RequestException 重试
            - 重试之间 sleep poll_interval
            - 全部失败返回 None
        """
        for attempt in range(self._polling.http_retry + 1):
            try:
                return self._session.get(
                    url,
                    timeout=self._polling.request_timeout,
                )
            except requests.exceptions.Timeout:
                if attempt >= self._polling.http_retry:
                    logger.warning("GET 最终超时: %s", url)
                    return None
                logger.warning(
                    "GET timeout (%d/%d): %s",
                    attempt + 1,
                    self._polling.http_retry,
                    url,
                )
                time.sleep(self._polling.poll_interval)
            except requests.exceptions.RequestException as e:
                if attempt >= self._polling.http_retry:
                    logger.warning("GET failed: %s (%s)", url, e)
                    return None
                logger.warning("GET 异常: %s", e)
                time.sleep(self._polling.poll_interval)

        return None

    # --------------------------------------------------------
    # GET alpha check
    # --------------------------------------------------------

    def check_alpha(self, alpha_id: str) -> Optional[Dict[str, Any]]:
        """
        查询单个 alpha 的 check 状态。

        GET /alphas/{alpha_id}/check

        TODO: 端点来自社区项目参考（rocky-d/wqb），
              非 BRAIN 官方文档。使用前请先用一条测试 alpha 手动验证。
              如返回 404 或 405，说明端点已变，需查官方文档。

        Returns
        -------
        dict  : 成功且响应为 JSON 时返回解析后的 dict
        None  : 网络失败 / 非 200 / JSON 解析失败
        """
        if not alpha_id:
            logger.warning("check_alpha: 空 alpha_id")
            return None

        url = self._endpoints.url(f"/alphas/{alpha_id}/check")
        response = self.get_with_retry(url)
        if response is None:
            logger.warning("check_alpha: GET 失败 %s", alpha_id)
            return None

        if response.status_code != 200:
            logger.warning(
                "check_alpha: HTTP %d for %s",
                response.status_code, alpha_id,
            )
            return None

        try:
            return response.json()
        except ValueError:
            logger.warning(
                "check_alpha: JSON 解析失败 %s", alpha_id,
            )
            return None

    # --------------------------------------------------------
    # POST alpha submit
    # --------------------------------------------------------

    def submit_alpha(self, alpha_id: str) -> bool:
        """
        提交单个 alpha 到 BRAIN。

        POST /alphas/{alpha_id}/submit

        TODO: 端点来自社区项目参考（rocky-d/wqb），
              非 BRAIN 官方文档。使用前请先用一条测试 alpha 手动验证。
              如返回 404 或 405，说明端点已变，需查官方文档。

        Returns
        -------
        True  : 响应 2xx
        False : 网络失败 / 非 2xx
        """
        if not alpha_id:
            logger.warning("submit_alpha: 空 alpha_id")
            return False

        url = self._endpoints.url(f"/alphas/{alpha_id}/submit")
        try:
            response = self._session.post(
                url,
                timeout=self._polling.request_timeout,
            )
        except requests.exceptions.Timeout:
            logger.warning("submit_alpha: timeout %s", alpha_id)
            return False
        except requests.exceptions.RequestException as e:
            logger.warning("submit_alpha: 异常 %s (%s)", alpha_id, e)
            return False

        if 200 <= response.status_code < 300:
            return True

        logger.warning(
            "submit_alpha: HTTP %d for %s",
            response.status_code, alpha_id,
        )
        return False

    # --------------------------------------------------------
    # POST simulation
    # --------------------------------------------------------

    def post_simulation(
        self,
        payload: Dict[str, Any],
    ) -> Optional[requests.Response]:
        """
        提交 Simulation。

        与 V3 simulate_alpha 的 POST 阶段行为一致：
            - POST /simulations
            - 网络异常 / timeout 返回 None
            - 成功时返回 Response（调用方负责校验 status_code / Location）
        """
        url = self._endpoints.url(self._endpoints.simulations)
        try:
            return self._session.post(
                url,
                json=payload,
                timeout=self._polling.request_timeout,
            )
        except requests.exceptions.Timeout:
            logger.warning("Simulation 提交 timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning("Simulation 提交异常: %s", e)
            return None
