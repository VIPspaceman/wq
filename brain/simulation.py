# brain/simulation.py
# -*- coding: utf-8 -*-

"""
Simulation 提交 + 等待（编排层）。

职责边界
--------
对齐 V3 WorldQuantBrainClient.simulate_alpha()：

    - 使用 SimulationRequest 构造 payload
    - POST /simulations
    - 校验 status_code == 201
    - 从 Location header 取轮询地址
    - 委托 SimulationPoller 轮询

**新增（Step 5 修正）**
    - 当 BRAIN 返回 HTTP 429 + CONCURRENT_SIMULATION_LIMIT_EXCEEDED 时，
      抛出 ConcurrencyLimitExceeded，供上层用更长的退避策略重试。
    - 其它 429（一般性限流）抛出 SimulationError。
    - 其它非 201 仍返回 None（保持 V3 行为）。

不含：
    - payload 数据结构定义（在 models.py）
    - HTTP 重试（在 client.py）
    - 轮询循环（在 polling.py）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import requests

from brain.models import SimulationRequest

if TYPE_CHECKING:
    from brain.client import BrainClient
    from brain.models import SimulationSettings
    from brain.polling import SimulationPoller


logger = logging.getLogger(__name__)


# ============================================================
# 异常
# ============================================================

class SimulationError(RuntimeError):
    """Simulation 相关错误的基类。"""


class ConcurrencyLimitExceeded(SimulationError):
    """
    BRAIN 返回 HTTP 429 + CONCURRENT_SIMULATION_LIMIT_EXCEEDED。

    这表示 BRAIN 侧识别到当前账号/环境并发 simulation 数已达上限。

    ⚠️ 这是 **BRAIN 官方 API 返回的事实**，不是工程推测。
    ⚠️ 具体上限是多少，BRAIN 未在本响应中返回；
       需要通过实际的 capability probing 探测（Step 8）。

    上层应对策略：
        - 使用 **更长的** 退避时间重试
        - 不要立即失败
        - 不要自动调整 max_workers（Step 8 才做）
    """


# ============================================================
# 内部辅助
# ============================================================

def _extract_error_detail(response: requests.Response) -> str:
    """尝试从 BRAIN 错误响应中提取 detail 字段。"""
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("detail", ""))
    except Exception:
        pass
    text = response.text or ""
    return text[:200]


def _is_concurrency_limit(detail: str) -> bool:
    return "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in detail


# ============================================================
# Runner
# ============================================================

class SimulationRunner:
    def __init__(
        self,
        client: "BrainClient",
        poller: "SimulationPoller",
        settings: "SimulationSettings",
    ) -> None:
        self._client = client
        self._poller = poller
        self._settings = settings

    # --------------------------------------------------------
    # 提交 + 等待
    # --------------------------------------------------------

    def simulate_alpha(
        self,
        expression: str,
    ) -> Optional[Dict[str, Any]]:
        request = SimulationRequest(
            expression=expression,
            settings=self._settings,
        )
        payload = request.to_brain_payload()

        response = self._client.post_simulation(payload)
        if response is None:
            return None

        # -------- 429：并发限制 / 一般限流 --------
        if response.status_code == 429:
            detail = _extract_error_detail(response)

            if _is_concurrency_limit(detail):
                raise ConcurrencyLimitExceeded(
                    f"BRAIN concurrent simulation limit exceeded "
                    f"({detail or 'no detail'})"
                )

            raise SimulationError(
                f"HTTP 429 rate limited: {detail or 'no detail'}"
            )

        # -------- 其它非 201：保持 V3 行为 --------
        if response.status_code != 201:
            logger.warning(
                "Simulation 提交失败: HTTP %s\n%s",
                response.status_code,
                response.text[:1000],
            )
            return None

        location = response.headers.get("Location")
        if not location:
            logger.warning("没有 Location header")
            return None

        logger.info("Location: %s", location)
        return self._poller.wait_for_simulation(location)
