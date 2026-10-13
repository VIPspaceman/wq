# brain/polling.py
# -*- coding: utf-8 -*-

"""
Simulation 轮询。

对齐 V3 wait_for_simulation()，并修正：

    BRAIN 现在的 WARNING 若带 message 字段，属于 **terminal 状态**
    （计算已结束但有错误），不是"进行中"。

    例：
        {
          "status": "WARNING",
          "message": "Incompatible unit for input of \"sqrt\" ...",
          "alpha": "A1v7A8Ke"
        }

    之前的行为是等 warning_timeout 再返回 None，导致：
        WARNING → 等 60s → timeout → 重试 → 又 WARNING → 又等 60s → ...
        每条单位不兼容的 alpha 浪费 3 次 simulation。

    修正后：
        WARNING + message → 立即返回 None（不等待、不重试）
        WARNING 无 message → 保持原有 timeout 逻辑
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from brain.client import BrainClient
    from config.settings import BrainEndpoints, PollingSettings


logger = logging.getLogger(__name__)


class SimulationPoller:
    def __init__(
        self,
        client: "BrainClient",
        endpoints: "BrainEndpoints",
        polling: "PollingSettings",
    ) -> None:
        self._client = client
        self._endpoints = endpoints
        self._polling = polling

    def wait_for_simulation(
        self,
        location: str,
    ) -> Optional[Dict[str, Any]]:
        start_time = time.time()
        warning_start: Optional[float] = None
        first_response = True
        last_progress: Optional[float] = None
        last_status: Optional[str] = None

        while True:
            elapsed = time.time() - start_time
            if elapsed > self._polling.simulation_timeout:
                logger.warning("Simulation 总超时")
                return None

            response = self._client.get_with_retry(location)
            if response is None:
                time.sleep(self._polling.poll_interval)
                continue

            if response.status_code not in (200, 202):
                logger.warning(
                    "查询 Simulation 异常: HTTP %s",
                    response.status_code,
                )
                time.sleep(self._polling.poll_interval)
                continue

            try:
                data = response.json()
            except Exception:
                logger.warning("JSON 解析失败")
                time.sleep(self._polling.poll_interval)
                continue

            if first_response:
                logger.debug(
                    "BRAIN Simulation 首返: %s", data,
                )
                first_response = False

            progress = data.get("progress")
            if progress is not None:
                try:
                    p = float(progress)
                    if (
                        last_progress is None
                        or abs(p - last_progress) >= 0.0001
                    ):
                        logger.debug(
                            "Simulation 进度: %.1f%%", p * 100.0,
                        )
                        last_progress = p
                except Exception:
                    pass

            status = data.get("status")
            if status is not None and status != last_status:
                logger.debug("Simulation 状态: %s", status)
                last_status = status

            # -------- ERROR --------
            if status == "ERROR":
                logger.warning(
                    "Simulation ERROR:\n%s",
                    json.dumps(data, ensure_ascii=False, indent=2),
                )
                return None

            # -------- WARNING --------
            if status == "WARNING":
                message = data.get("message", "")

                # ★ 修正：带 message → terminal，立即返回
                if message:
                    logger.warning(
                        "Simulation WARNING (terminal): %s",
                        message,
                    )
                    return None

                # 无 message → 保持原有 timeout 逻辑
                # 但使用更短的超时（warning_timeout 与 poll_interval 的较小值），
                # 避免无 message 的 WARNING 拖到 simulation_timeout。
                if warning_start is None:
                    warning_start = time.time()
                    logger.warning(
                        "Simulation 进入 WARNING (无 message):\n%s",
                        json.dumps(data, ensure_ascii=False, indent=2),
                    )
                warning_elapsed = time.time() - warning_start
                effective_warning_timeout = min(
                    self._polling.warning_timeout,
                    self._polling.poll_interval * 3,
                )
                if warning_elapsed > effective_warning_timeout:
                    logger.warning(
                        "WARNING 持续超过 %ss（无 message），放弃",
                        effective_warning_timeout,
                    )
                    return None
            else:
                warning_start = None

            # -------- progress 100% 但 status 未变 --------
            if (
                progress is not None
                and status is None
            ):
                try:
                    if float(progress) >= 1.0:
                        logger.warning(
                            "progress=100%% 但 status 缺失，"
                            "继续轮询等待 COMPLETE"
                        )
                except Exception:
                    pass

            # -------- COMPLETE --------
            if status == "COMPLETE":
                alpha_id = data.get("alpha")
                if not alpha_id:
                    logger.warning("Simulation 完成但无 Alpha ID")
                    return None

                logger.info("Simulation 完成 | Alpha ID=%s", alpha_id)

                alpha_url = (
                    f"{self._endpoints.base_url}"
                    f"{self._endpoints.alphas}"
                    f"/{alpha_id}"
                )
                alpha_response = self._client.get_with_retry(alpha_url)
                if alpha_response is None:
                    return None
                if not alpha_response.ok:
                    logger.warning("Alpha 详情获取失败")
                    return None
                try:
                    alpha_data = alpha_response.json()
                except Exception:
                    return None

                alpha_data["_alpha_id"] = alpha_id
                return alpha_data

            time.sleep(self._polling.poll_interval)
