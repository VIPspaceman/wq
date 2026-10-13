# -*- coding: utf-8 -*-

"""
brain/capabilities.py 单元测试。

覆盖：
    - _extract_list 的两种响应格式
    - _is_rate_limit 的 429 / 文本 / 异常分支
    - _safe_text 的异常分支
    - _retry_after_seconds 的解析分支
    - fetch_data_fields 的分页 / 429 重试 / 网络重试 / 去重 / 提前终止
    - fetch_operators 的正常与异常分支
    - discover 的组合行为
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from brain import capabilities as cap
from brain.capabilities import (
    DATA_FIELDS_PAGE_LIMIT,
    RATE_LIMIT_MAX_RETRY,
    NETWORK_MAX_RETRY,
    RETRY_AFTER_MAX_SECONDS,
    _extract_list,
    _is_rate_limit,
    _retry_after_seconds,
    _safe_text,
    discover,
    fetch_data_fields,
    fetch_operators,
)


# ============================================================
# 测试替身
# ============================================================

class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        text: str = "",
        json_data: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json = json_data
        self.headers = headers or {}

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("no json")
        return self._json


class _FakeSession:
    """
    按顺序返回预设响应；用完后返回最后一个（或抛异常）。
    """

    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self._responses:
            raise AssertionError("no more fake responses")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class _FakeEndpoints:
    def url(self, path: str) -> str:
        return "https://example.test" + path


class _FakePolling:
    request_timeout = 1.0


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """所有测试中禁用 time.sleep，避免拖慢。"""
    monkeypatch.setattr(cap.time, "sleep", lambda *_: None)


def _page(items: List[Dict[str, Any]]) -> _FakeResponse:
    return _FakeResponse(200, json_data=items)


def _full_page(start: int) -> _FakeResponse:
    return _page([{"id": f"f{start + i}"} for i in range(DATA_FIELDS_PAGE_LIMIT)])


# ============================================================
# _extract_list
# ============================================================

def test_extract_list_direct_array() -> None:
    data = [{"id": "a"}, {"id": "b"}]
    assert _extract_list(data, "/x") == data


def test_extract_list_wrapped_results() -> None:
    data = {"results": [{"id": "a"}, {"id": "b"}]}
    assert _extract_list(data, "/x") == [{"id": "a"}, {"id": "b"}]


def test_extract_list_filters_non_dict() -> None:
    data = [{"id": "a"}, "not-a-dict", 123]
    assert _extract_list(data, "/x") == [{"id": "a"}]


def test_extract_list_unknown_format() -> None:
    assert _extract_list("garbage", "/x") == []
    assert _extract_list({"foo": "bar"}, "/x") == []


# ============================================================
# _safe_text
# ============================================================

def test_safe_text_normal() -> None:
    assert _safe_text(_FakeResponse(200, "hello")) == "hello"


def test_safe_text_truncates() -> None:
    assert _safe_text(_FakeResponse(200, "x" * 1000), limit=10) == "x" * 10


def test_safe_text_handles_exception() -> None:
    class _Bad:
        status_code = 200

        @property
        def text(self):
            raise RuntimeError("boom")

    assert _safe_text(_Bad()) == ""


# ============================================================
# _is_rate_limit
# ============================================================

def test_is_rate_limit_429() -> None:
    assert _is_rate_limit(_FakeResponse(429))


def test_is_rate_limit_text() -> None:
    assert _is_rate_limit(
        _FakeResponse(400, "Rate limit exceeded, slow down")
    )


def test_is_rate_limit_underscore_variant() -> None:
    assert _is_rate_limit(_FakeResponse(400, "rate_limit hit"))


def test_is_rate_limit_too_many_requests() -> None:
    assert _is_rate_limit(_FakeResponse(400, "Too Many Requests"))


def test_is_rate_limit_throttled() -> None:
    assert _is_rate_limit(_FakeResponse(400, "request throttled"))


def test_is_rate_limit_false() -> None:
    assert not _is_rate_limit(_FakeResponse(200, "ok"))


def test_is_rate_limit_only_checks_prefix() -> None:
    """
    只检查前 500 字符：'rate limit' 出现在 500 字符之后时不应命中。
    """
    padding = "x" * 600
    body = padding + "rate limit"
    assert not _is_rate_limit(_FakeResponse(400, body))


def test_is_rate_limit_handles_exception() -> None:
    class _Bad:
        status_code = 200

        @property
        def text(self):
            raise RuntimeError("boom")

    assert not _is_rate_limit(_Bad())


# ============================================================
# _retry_after_seconds
# ============================================================

def test_retry_after_numeric() -> None:
    resp = _FakeResponse(429, headers={"Retry-After": "7"})
    assert _retry_after_seconds(resp) == 7.0


def test_retry_after_lowercase_header() -> None:
    resp = _FakeResponse(429, headers={"retry-after": "3.5"})
    assert _retry_after_seconds(resp) == 3.5


def test_retry_after_missing() -> None:
    assert _retry_after_seconds(_FakeResponse(429)) is None


def test_retry_after_invalid() -> None:
    resp = _FakeResponse(429, headers={"Retry-After": "soon"})
    assert _retry_after_seconds(resp) is None


def test_retry_after_negative() -> None:
    resp = _FakeResponse(429, headers={"Retry-After": "-1"})
    assert _retry_after_seconds(resp) is None


def test_retry_after_capped() -> None:
    resp = _FakeResponse(429, headers={"Retry-After": "9999"})
    assert _retry_after_seconds(resp) == RETRY_AFTER_MAX_SECONDS


# ============================================================
# fetch_operators
# ============================================================

def test_fetch_operators_ok() -> None:
    session = _FakeSession([_FakeResponse(200, json_data=[{"name": "add"}])])
    result = fetch_operators(session, _FakeEndpoints(), _FakePolling())
    assert result == [{"name": "add"}]


def test_fetch_operators_http_error() -> None:
    session = _FakeSession([_FakeResponse(500, text="boom")])
    assert fetch_operators(session, _FakeEndpoints(), _FakePolling()) == []


def test_fetch_operators_network_error() -> None:
    session = _FakeSession([RuntimeError("net down")])
    assert fetch_operators(session, _FakeEndpoints(), _FakePolling()) == []


def test_fetch_operators_bad_json() -> None:
    session = _FakeSession([_FakeResponse(200, text="not json")])
    assert fetch_operators(session, _FakeEndpoints(), _FakePolling()) == []


# ============================================================
# fetch_data_fields
# ============================================================

def test_fetch_data_fields_single_short_page() -> None:
    session = _FakeSession([_page([{"id": "a"}, {"id": "b"}])])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert [x["id"] for x in result] == ["a", "b"]
    assert len(session.calls) == 1


def test_fetch_data_fields_pagination() -> None:
    session = _FakeSession([
        _full_page(0),
        _page([{"id": "tail"}]),
    ])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert len(result) == DATA_FIELDS_PAGE_LIMIT + 1
    assert result[-1]["id"] == "tail"
    assert len(session.calls) == 2
    # 第二页 offset 应为 50
    assert session.calls[1]["params"]["offset"] == DATA_FIELDS_PAGE_LIMIT


def test_fetch_data_fields_dedup() -> None:
    session = _FakeSession([
        _page([{"id": "a"}, {"id": "a"}, {"id": "b"}]),
    ])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert [x["id"] for x in result] == ["a", "b"]


def test_fetch_data_fields_dedup_by_name_fallback() -> None:
    session = _FakeSession([
        _page([{"name": "n1"}, {"name": "n1"}, {"field": "f1"}]),
    ])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert len(result) == 2


def test_fetch_data_fields_skips_empty_key() -> None:
    session = _FakeSession([
        _page([{"id": ""}, {"id": "  "}, {"id": "ok"}]),
    ])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert [x["id"] for x in result] == ["ok"]


def test_fetch_data_fields_429_then_success() -> None:
    session = _FakeSession([
        _FakeResponse(429, text="rate limit"),
        _page([{"id": "a"}]),
    ])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert [x["id"] for x in result] == ["a"]
    assert len(session.calls) == 2


def test_fetch_data_fields_429_exhausts_retries() -> None:
    responses = [_FakeResponse(429, text="rate limit")
                 for _ in range(RATE_LIMIT_MAX_RETRY + 1)]
    session = _FakeSession(responses)
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert result == []
    assert len(session.calls) == RATE_LIMIT_MAX_RETRY + 1


def test_fetch_data_fields_429_uses_retry_after(monkeypatch) -> None:
    sleeps: List[float] = []
    monkeypatch.setattr(cap.time, "sleep", lambda s: sleeps.append(s))

    session = _FakeSession([
        _FakeResponse(429, headers={"Retry-After": "2"}),
        _page([{"id": "a"}]),
    ])
    fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert 2.0 in sleeps


def test_fetch_data_fields_429_resets_after_progress() -> None:
    """
    第 1 页 429 一次后成功；第 2 页再 429 一次后成功。
    重试计数应在推进 page 后重置，因此两页都能成功。
    """
    session = _FakeSession([
        _FakeResponse(429, text="rate limit"),
        _full_page(0),
        _FakeResponse(429, text="rate limit"),
        _page([{"id": "tail"}]),
    ])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert len(result) == DATA_FIELDS_PAGE_LIMIT + 1


def test_fetch_data_fields_network_retry_then_success() -> None:
    session = _FakeSession([
        RuntimeError("net down"),
        _page([{"id": "a"}]),
    ])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert [x["id"] for x in result] == ["a"]


def test_fetch_data_fields_network_exhausts_retries() -> None:
    responses = [RuntimeError("net down")
                 for _ in range(NETWORK_MAX_RETRY + 1)]
    session = _FakeSession(responses)
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert result == []
    assert len(session.calls) == NETWORK_MAX_RETRY + 1


def test_fetch_data_fields_http_error_stops() -> None:
    session = _FakeSession([_FakeResponse(500, text="boom")])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert result == []
    assert len(session.calls) == 1


def test_fetch_data_fields_bad_json_stops() -> None:
    session = _FakeSession([_FakeResponse(200, text="not json")])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert result == []


def test_fetch_data_fields_empty_page_stops() -> None:
    session = _FakeSession([_page([])])
    result = fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert result == []


def test_fetch_data_fields_respects_max_pages() -> None:
    session = _FakeSession([_full_page(0), _full_page(50), _full_page(100)])
    result = fetch_data_fields(
        session, _FakeEndpoints(), _FakePolling(), max_pages=2,
    )
    assert len(result) == DATA_FIELDS_PAGE_LIMIT * 2
    assert len(session.calls) == 2


def test_fetch_data_fields_passes_dataset_id() -> None:
    session = _FakeSession([_page([{"id": "a"}])])
    fetch_data_fields(
        session, _FakeEndpoints(), _FakePolling(),
        dataset_id="fundamental6",
    )
    assert session.calls[0]["params"]["dataset.id"] == "fundamental6"


def test_fetch_data_fields_omits_dataset_id_when_none() -> None:
    session = _FakeSession([_page([{"id": "a"}])])
    fetch_data_fields(session, _FakeEndpoints(), _FakePolling())
    assert "dataset.id" not in session.calls[0]["params"]


# ============================================================
# discover
# ============================================================

def test_discover_operators_only() -> None:
    session = _FakeSession([_FakeResponse(200, json_data=[{"name": "add"}])])
    caps = discover(session, _FakeEndpoints(), _FakePolling())
    assert caps.operators == [{"name": "add"}]
    assert caps.data_fields == []


def test_discover_data_fields_only() -> None:
    session = _FakeSession([_page([{"id": "a"}])])
    caps = discover(
        session, _FakeEndpoints(), _FakePolling(),
        operators=False, data_fields=True,
    )
    assert caps.operators == []
    assert [x["id"] for x in caps.data_fields] == ["a"]


def test_discover_both() -> None:
    session = _FakeSession([
        _FakeResponse(200, json_data=[{"name": "add"}]),
        _page([{"id": "a"}]),
    ])
    caps = discover(
        session, _FakeEndpoints(), _FakePolling(),
        operators=True, data_fields=True,
    )
    assert caps.operators == [{"name": "add"}]
    assert [x["id"] for x in caps.data_fields] == ["a"]
