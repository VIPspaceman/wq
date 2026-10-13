# evaluation/official_checks.py
# -*- coding: utf-8 -*-

"""
从 BRAIN Simulation result 提取官方 checks。

背景
----
BRAIN 的 alpha 详情返回中，is.checks[] 是一份 **官方检查列表**，
例如（示例字段，非穷举）：

    {"name": "LOW_SHARPE",   "result": "FAIL",    "limit": 1.25, "value": -1.6}
    {"name": "LOW_FITNESS",  "result": "FAIL",    "limit": 1.0,  "value": -0.85}
    {"name": "HIGH_TURNOVER","result": "PASS",    "limit": 0.7,  "value": 0.5346}
    {"name": "SELF_CORRELATION", "result": "PENDING"}
    {"name": "MATCHES_COMPETITION", "result": "PASS",
     "competitions": [{"id": "challenge", "name": "Challenge"}]}

【重要说明】
------------------------------------------------------------
1. 这是 **BRAIN 官方返回的事实**，不是本项目定义的阈值。
2. 本项目 **不** 把这些 checks 声称为"官方 Alpha 提交标准"，
   因为完整提交流程还可能涉及未在此列表中的其它条件。
3. 本项目 **不** 对 checks 结果做任何改写 —— 原样保留。
------------------------------------------------------------

职责边界
--------
只做提取与结构化；不做任何判定策略、不做过滤。

不含：
    - 我们的内部 heuristic（在 config/thresholds.py）
    - Score / Classification（在 scoring.py / classifier.py）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ============================================================
# 单条 check
# ============================================================

@dataclass
class OfficialCheck:
    name: str
    result: str                        # "PASS" / "FAIL" / "PENDING" / 其它
    limit: Any = None
    value: Any = None
    # 其他字段（如 competitions），原样保留
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OfficialCheck":
        if not isinstance(d, dict):
            return cls(name="UNKNOWN", result="UNKNOWN")
        known = {"name", "result", "limit", "value"}
        return cls(
            name=str(d.get("name", "UNKNOWN")),
            result=str(d.get("result", "UNKNOWN")),
            limit=d.get("limit"),
            value=d.get("value"),
            extra={k: v for k, v in d.items() if k not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {"name": self.name, "result": self.result}
        if self.limit is not None:
            out["limit"] = self.limit
        if self.value is not None:
            out["value"] = self.value
        out.update(self.extra)
        return out


# ============================================================
# checks 集合
# ============================================================

@dataclass
class OfficialChecks:
    checks: List[OfficialCheck] = field(default_factory=list)

    # --------------------------------------------------------
    # 便捷查询
    # --------------------------------------------------------

    def by_name(self, name: str) -> List[OfficialCheck]:
        return [c for c in self.checks if c.name == name]

    def has_fail(self) -> bool:
        return any(c.result == "FAIL" for c in self.checks)

    def has_pending(self) -> bool:
        return any(c.result == "PENDING" for c in self.checks)

    def all_pass(self) -> bool:
        return (
            len(self.checks) > 0
            and all(c.result == "PASS" for c in self.checks)
        )

    def to_list(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.checks]

    def __len__(self) -> int:
        return len(self.checks)


# ============================================================
# 提取
# ============================================================

def extract_official_checks(result: Dict[str, Any]) -> OfficialChecks:
    """
    从 BRAIN result dict 中提取官方 checks。

    与 BRAIN 返回结构对齐：
        result["is"]["checks"]  -> List[dict]

    若缺失或结构异常 → 返回空 OfficialChecks（不抛异常）。
    """
    is_data = result.get("is", {})
    if not isinstance(is_data, dict):
        return OfficialChecks()

    raw = is_data.get("checks", [])
    if not isinstance(raw, list):
        return OfficialChecks()

    return OfficialChecks(
        checks=[OfficialCheck.from_dict(x) for x in raw]
    )


# ============================================================
# BRAIN-3：从 raw_json 解析 checks（本地完整检查）
# ============================================================

def _extract_checks_list(raw_json: Any) -> List[Dict[str, Any]]:
    """
    从多种输入形态中提取 checks 列表。

    支持：
        - dict: {"is_raw": {"checks": [...]}}   ← DB raw_json 格式
        - dict: {"is":     {"checks": [...]}}   ← pipeline result 格式
        - dict: {"checks": [...]}               ← 直接是 is 块
        - str : JSON 字符串（上述任一形态）
        - 其它 / 缺失 → []

    不抛异常。
    """
    if raw_json is None:
        return []

    # str → 尝试 JSON 解析
    if isinstance(raw_json, str):
        if not raw_json.strip():
            return []
        try:
            import json as _json
            raw_json = _json.loads(raw_json)
        except (ValueError, TypeError):
            return []

    if not isinstance(raw_json, dict):
        return []

    # 依次尝试 is_raw / is / 直接 checks
    for key in ("is_raw", "is"):
        block = raw_json.get(key)
        if isinstance(block, dict):
            checks = block.get("checks")
            if isinstance(checks, list):
                return [c for c in checks if isinstance(c, dict)]

    # 直接是 is 块
    checks = raw_json.get("checks")
    if isinstance(checks, list):
        return [c for c in checks if isinstance(c, dict)]

    return []


def parse_official_checks(raw_json: Any) -> Dict[str, str]:
    """
    从 raw_json 解析 BRAIN 官方 checks。

    支持两种输入：
        - dict: {"is_raw": {...}} 或 {"is": {...}}
        - str : JSON 字符串

    Returns
    -------
    {check_name: result}
        例如 {"LOW_SHARPE": "PASS", "SELF_CORRELATION": "PENDING", ...}

    无 checks 数据时返回空 dict（不抛异常）。
    """
    checks = _extract_checks_list(raw_json)
    out: Dict[str, str] = {}
    for c in checks:
        name = str(c.get("name", "") or "")
        if not name:
            continue
        out[name] = str(c.get("result", "UNKNOWN") or "UNKNOWN")
    return out


def all_checks_pass(
    raw_json: Any,
    ignore_pending: bool = True,
) -> bool:
    """
    所有检查（除 PENDING）都是 PASS 才返回 True。

    Parameters
    ----------
    raw_json : dict | str
        同 parse_official_checks。
    ignore_pending : bool
        True（默认）时忽略 PENDING（如 self-corr 未结算）。
        False 时 PENDING 视为不通过。

    Returns
    -------
    True  : 有 checks 数据，且所有非 PENDING 项都是 PASS
    False : 无 checks 数据，或存在 FAIL / 其它非 PASS 结果
    """
    results = parse_official_checks(raw_json)
    if not results:
        return False

    for name, result in results.items():
        if result == "PASS":
            continue
        if result == "PENDING" and ignore_pending:
            continue
        # FAIL / UNKNOWN / 其它 → 不通过
        return False

    return True
