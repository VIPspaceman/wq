# brain/capabilities.py
# -*- coding: utf-8 -*-

"""
BRAIN Capability Discovery。

职责边界
--------
只负责从 BRAIN API 获取平台能力信息：
    - fetch_operators()   : GET /operators
    - fetch_data_fields() : GET /data-fields（自动分页 + 429 处理）
    - discover()          : 统一入口，可一次性拿 operators / data_fields

API 事实（已由本机实测确认）
----------------------------
/data-fields ：
    - limit 上限 50（超过报 400 "pagination limit too high"）
    - 速率限制很紧，连续请求立即 429
    - 按 id 字母序返回

响应格式
--------
两种可能（均兼容）：
    A. 直接数组      : [{...}, {...}]
    B. 包裹在 results: {"results": [{...}]}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import requests
    from config.settings import BrainEndpoints, PollingSettings


logger = logging.getLogger(__name__)


# ============================================================
# 常量
# ============================================================

DATA_FIELDS_PAGE_LIMIT = 50       # API 单页上限
DEFAULT_MAX_PAGES = 40            # 默认最多拉 40 页 = 2000 条
PAGE_DELAY_SECONDS = 1.5          # 分页之间延时（避免 429）
RATE_LIMIT_WAIT_SECONDS = 5.0     # 遇 429 时等待多久再重试当前页
RATE_LIMIT_MAX_RETRY = 3          # 429 重试上限
NETWORK_MAX_RETRY = 2             # 网络错误重试上限
RETRY_AFTER_MAX_SECONDS = 60.0    # Retry-After 头允许的最大等待秒数

# 命中任一关键词即视为限流（小写匹配）
_RATE_LIMIT_KEYWORDS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttl",
)


# ============================================================
# 工具
# ============================================================

def _extract_list(data: Any, endpoint: str) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return [x for x in results if isinstance(x, dict)]
    logger.warning(
        "%s 响应格式未知: type=%s keys=%s",
        endpoint,
        type(data).__name__,
        list(data.keys()) if isinstance(data, dict) else "N/A",
    )
    return []


def _safe_text(response, limit: int = 500) -> str:
    """
    安全读取 response.text 的前 limit 个字符。
    text 属性抛异常时返回空串。
    """
    try:
        return (response.text or "")[:limit]
    except Exception:
        return ""


def _is_rate_limit(response) -> bool:
    if response.status_code == 429:
        return True
    body = _safe_text(response).lower()
    return any(kw in body for kw in _RATE_LIMIT_KEYWORDS)


def _retry_after_seconds(response) -> Optional[float]:
    """
    从 Retry-After 头解析等待秒数。
    仅支持数字形式（HTTP-date 形式忽略）；非法值返回 None。
    """
    try:
        headers = getattr(response, "headers", None) or {}
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return min(value, RETRY_AFTER_MAX_SECONDS)


# ============================================================
# 数据
# ============================================================

@dataclass
class Capabilities:
    operators: List[Dict[str, Any]] = field(default_factory=list)
    data_fields: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# 请求
# ============================================================

def _safe_get(
    session: "requests.Session",
    url: str,
    timeout: float,
    params: Optional[Dict[str, Any]] = None,
):
    """
    发起 GET；网络异常 → 返回 None。**不吞 429**，由调用方判断。
    """
    try:
        return session.get(url, params=params, timeout=timeout)
    except Exception as e:
        logger.warning("GET %s 异常: %r", url, e)
        return None


# ============================================================
# 公开接口：fetch
# ============================================================

def fetch_operators(
    session: "requests.Session",
    endpoints: "BrainEndpoints",
    polling: "PollingSettings",
) -> List[Dict[str, Any]]:
    url = endpoints.url("/operators")
    logger.info("Capability Discovery: 请求 %s", url)

    response = _safe_get(session, url, timeout=polling.request_timeout)
    if response is None or response.status_code != 200:
        if response is not None:
            logger.warning(
                "GET /operators 返回 HTTP %s", response.status_code
            )
        return []

    try:
        data = response.json()
    except Exception as e:
        logger.warning("JSON 解析失败: %r", e)
        return []

    result = _extract_list(data, "/operators")
    logger.info("Capability Discovery: 获得 %d 个运算符", len(result))
    return result


def fetch_data_fields(
    session: "requests.Session",
    endpoints: "BrainEndpoints",
    polling: "PollingSettings",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    instrument_type: str = "EQUITY",
    dataset_id: Optional[str] = None,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> List[Dict[str, Any]]:
    """
    获取 BRAIN 平台数据字段（自动分页 + 429 重试）。

    Parameters
    ----------
    dataset_id : 指定 dataset.id（强烈推荐！否则会拉全平台字段）
    max_pages  : 最多拉几页（每页 50 条）
    """
    url = endpoints.url("/data-fields")

    collected: List[Dict[str, Any]] = []
    seen_ids: set = set()
    page = 0
    rate_limit_retries = 0
    network_retries = 0

    while page < max_pages:
        offset = page * DATA_FIELDS_PAGE_LIMIT
        params: Dict[str, Any] = {
            "region": region,
            "delay": delay,
            "universe": universe,
            "instrumentType": instrument_type,
            "limit": DATA_FIELDS_PAGE_LIMIT,
            "offset": offset,
        }
        if dataset_id:
            params["dataset.id"] = dataset_id

        logger.info(
            "Capability Discovery: /data-fields page=%d offset=%d dataset=%s",
            page, offset, dataset_id or "(all)",
        )

        response = _safe_get(
            session, url,
            timeout=polling.request_timeout,
            params=params,
        )

        # ---------- 网络错误 ----------
        if response is None:
            network_retries += 1
            if network_retries > NETWORK_MAX_RETRY:
                logger.warning(
                    "第 %d 页网络错误重试超过 %d 次，停止分页",
                    page, NETWORK_MAX_RETRY,
                )
                break
            logger.warning(
                "第 %d 页网络错误，等待 %.1fs 后重试 (%d/%d)",
                page, PAGE_DELAY_SECONDS,
                network_retries, NETWORK_MAX_RETRY,
            )
            time.sleep(PAGE_DELAY_SECONDS)
            continue    # 不推进 page

        # ---------- 429 ----------
        if _is_rate_limit(response):
            rate_limit_retries += 1
            if rate_limit_retries > RATE_LIMIT_MAX_RETRY:
                logger.warning(
                    "第 %d 页 429 重试超过 %d 次，停止分页",
                    page, RATE_LIMIT_MAX_RETRY,
                )
                break
            wait = _retry_after_seconds(response)
            if wait is None:
                wait = RATE_LIMIT_WAIT_SECONDS
            logger.warning(
                "第 %d 页 dataset=%s 429，等待 %.1fs 后重试 (%d/%d)",
                page, dataset_id or "(all)", wait,
                rate_limit_retries, RATE_LIMIT_MAX_RETRY,
            )
            time.sleep(wait)
            continue    # 不推进 page

        # ---------- 其它错误 ----------
        if response.status_code != 200:
            logger.warning(
                "第 %d 页 HTTP %s: %s",
                page, response.status_code,
                _safe_text(response, 200),
            )
            break

        # ---------- 解析 ----------
        try:
            data = response.json()
        except Exception as e:
            logger.warning("第 %d 页 JSON 解析失败: %r", page, e)
            break

        batch = _extract_list(data, "/data-fields")
        if not batch:
            logger.info("第 %d 页为空，停止分页", page)
            break

        # 去重
        new_count = 0
        for item in batch:
            key = str(
                item.get("id")
                or item.get("name")
                or item.get("field")
                or ""
            ).strip()
            if not key or key in seen_ids:
                continue
            seen_ids.add(key)
            collected.append(item)
            new_count += 1

        logger.info(
            "第 %d 页 dataset=%s：返回 %d 条（新增 %d，累计 %d）",
            page, dataset_id or "(all)", len(batch), new_count, len(collected),
        )

        # 满页 → 继续；不满 → 最后一页
        if len(batch) < DATA_FIELDS_PAGE_LIMIT:
            break

        page += 1
        # 成功推进 page 后重置重试计数，
        # 避免"429 → 成功 → 429 → 成功"无限循环。
        rate_limit_retries = 0
        network_retries = 0

        # 分页之间延时
        time.sleep(PAGE_DELAY_SECONDS)

    logger.info(
        "Capability Discovery: dataset=%s 共获取 %d 个数据字段",
        dataset_id or "(all)", len(collected),
    )
    return collected


# ============================================================
# 公开接口：discover（统一入口）
# ============================================================

def discover(
    session: "requests.Session",
    endpoints: "BrainEndpoints",
    polling: "PollingSettings",
    *,
    operators: bool = True,
    data_fields: bool = False,
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    instrument_type: str = "EQUITY",
    dataset_id: Optional[str] = None,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> Capabilities:
    """
    一次性探测平台能力。

    Parameters
    ----------
    operators    : 是否拉取 /operators
    data_fields   : 是否拉取 /data-fields（默认关闭，因为量大）
    region / delay / universe / instrument_type : 上下文
    dataset_id   : 指定 dataset.id（可选）
    max_pages    : 数据字段最多拉几页

    Returns
    -------
    Capabilities(operators=[...], data_fields=[...])
    """
    caps = Capabilities()

    if operators:
        caps.operators = fetch_operators(session, endpoints, polling)

    if data_fields:
        caps.data_fields = fetch_data_fields(
            session, endpoints, polling,
            region=region,
            delay=delay,
            universe=universe,
            instrument_type=instrument_type,
            dataset_id=dataset_id,
            max_pages=max_pages,
        )

    return caps
