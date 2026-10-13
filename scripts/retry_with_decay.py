# -*- coding: utf-8 -*-

"""
高 turnover alpha 的 decay 补跑脚本。

功能
----
从 ``experiments/{experiment_id}/alpha_results.json`` 读取已评估的 alpha，
筛选出 turnover 偏高但 sharpe / fitness 达标的记录，
用不同的 decay 值重新提交 simulation，观察 turnover 是否下降。

用法
----
    python scripts/retry_with_decay.py --experiment-id exp_xxx
    python scripts/retry_with_decay.py --experiment-id exp_xxx \
        --min-turnover 0.7 --decays 20,50 --max-per-run 10

注意
----
- 本脚本 **不写 DB**，结果仅供人工参考。
- 每条 sim 独立 try/except，单条失败不影响其它。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# 把项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brain import create_authenticated_session
from brain.client import BrainClient
from brain.models import SimulationSettings
from brain.polling import SimulationPoller
from brain.simulation import SimulationRunner
from config.settings import AppSettings


# PASS 门槛
_PASS_TURNOVER = 0.7
_PASS_SHARPE = 1.25
_PASS_FITNESS = 1.0


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="retry_with_decay.py",
        description="对高 turnover alpha 用不同 decay 重新提交 sim",
    )
    parser.add_argument(
        "--experiment-id", required=True,
        help="experiment id（对应 experiments/{id}/）",
    )
    parser.add_argument(
        "--min-turnover", type=float, default=0.7,
        help="只处理 turnover 大于此值的记录（默认 0.7）",
    )
    parser.add_argument(
        "--min-sharpe", type=float, default=1.0,
        help="只处理 sharpe 大于此值的记录（默认 1.0）",
    )
    parser.add_argument(
        "--min-fitness", type=float, default=0.8,
        help="只处理 fitness 大于此值的记录（默认 0.8）",
    )
    parser.add_argument(
        "--decays", default="20,50",
        help="逗号分隔的 decay 值（默认 20,50）",
    )
    parser.add_argument(
        "--max-per-run", type=int, default=10,
        help="最多处理几条（默认 10）",
    )
    return parser.parse_args(argv)


def _parse_decays(text: str) -> List[int]:
    decays: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            decays.append(int(part))
        except ValueError:
            print(f"  warn: 忽略非法 decay 值: {part!r}")
    return decays


def _extract_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """从 simulation 结果里提取 sharpe / fitness / turnover。"""
    is_data = result.get("is", {}) or {}

    def _get_float(key: str) -> float:
        try:
            return float(is_data.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "sharpe": _get_float("sharpe"),
        "fitness": _get_float("fitness"),
        "turnover": _get_float("turnover"),
    }


def _build_settings_with_decay(
    base: SimulationSettings,
    decay: int,
) -> SimulationSettings:
    """基于现有 settings 构造一个 decay 被替换的副本。"""
    return SimulationSettings(
        instrument_type=base.instrument_type,
        region=base.region,
        universe=base.universe,
        delay=base.delay,
        decay=decay,
        neutralization=base.neutralization,
        truncation=base.truncation,
        pasteurization=base.pasteurization,
        unit_handling=base.unit_handling,
        nan_handling=base.nan_handling,
        language=base.language,
        visualization=base.visualization,
    )


def main(argv=None) -> None:
    args = _parse_args(argv)

    decays = _parse_decays(args.decays)
    if not decays:
        print("❌ 没有有效的 decay 值")
        sys.exit(2)

    settings = AppSettings.from_env()

    # -------- 读 alpha_results.json --------
    path = os.path.join(
        "experiments", args.experiment_id, "alpha_results.json"
    )
    if not os.path.exists(path):
        print(f"❌ 找不到 {path}")
        sys.exit(2)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 {path} 失败: {e}")
        sys.exit(2)

    if not isinstance(data, list):
        print(f"❌ {path} 不是 list")
        sys.exit(2)

    # -------- 筛选 --------
    candidates: List[Dict[str, Any]] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        try:
            turnover = float(r.get("turnover") or 0.0)
            sharpe = float(r.get("sharpe") or 0.0)
            fitness = float(r.get("fitness") or 0.0)
        except (TypeError, ValueError):
            continue
        if (
            turnover > args.min_turnover
            and sharpe > args.min_sharpe
            and fitness > args.min_fitness
        ):
            candidates.append(r)

    candidates.sort(
        key=lambda x: float(x.get("score") or 0.0),
        reverse=True,
    )
    candidates = candidates[: max(0, args.max_per_run)]

    print(f"找到 {len(candidates)} 条高 turnover 的 alpha 待补跑")
    if not candidates:
        return

    # -------- 登录 BRAIN --------
    session = create_authenticated_session(
        settings.credentials,
        settings.endpoints,
        request_timeout=settings.polling.request_timeout,
    )
    client = BrainClient(session, settings.endpoints, settings.polling)

    successful: List[Dict[str, Any]] = []

    # -------- 逐条补跑 --------
    for idx, rec in enumerate(candidates, start=1):
        expression = str(rec.get("expression") or "")
        alpha_id = str(rec.get("alpha_id") or "")
        if not expression:
            print(f"[{idx}/{len(candidates)}] 跳过：无 expression")
            continue

        print()
        print("=" * 80)
        print(
            f"[{idx}/{len(candidates)}] alpha_id={alpha_id} "
            f"turnover={rec.get('turnover')} "
            f"sharpe={rec.get('sharpe')} "
            f"fitness={rec.get('fitness')}"
        )
        print(f"  expr: {expression}")
        print("=" * 80)

        for decay in decays:
            try:
                new_settings = _build_settings_with_decay(
                    settings.simulation, decay,
                )
                poller = SimulationPoller(
                    client, settings.endpoints, settings.polling,
                )
                runner = SimulationRunner(
                    client, poller, new_settings,
                )
                result = runner.simulate_alpha(expression)
                if result is None:
                    print(f"  decay={decay:3d} | (无结果)")
                    time.sleep(0.5)
                    continue

                m = _extract_metrics(result)
                print(
                    f"  decay={decay:3d} | "
                    f"sharpe={m['sharpe']:+.4f} "
                    f"fitness={m['fitness']:+.4f} "
                    f"turnover={m['turnover']:.4f}"
                )

                if (
                    m["turnover"] <= _PASS_TURNOVER
                    and m["sharpe"] >= _PASS_SHARPE
                    and m["fitness"] >= _PASS_FITNESS
                ):
                    print("  🎉 PASS! 可提交")
                    successful.append({
                        "alpha_id": alpha_id,
                        "expression": expression,
                        "decay": decay,
                        "sharpe": m["sharpe"],
                        "fitness": m["fitness"],
                        "turnover": m["turnover"],
                    })
            except Exception as e:
                print(f"  decay={decay:3d} | error: {e!r}")

            time.sleep(0.5)

    # -------- 汇总 --------
    print()
    print("=" * 80)
    print(f"共 {len(successful)} 条变体达到 PASS 门槛")
    for s in successful:
        print(
            f"  alpha_id={s['alpha_id']}, decay={s['decay']} "
            f"(sharpe={s['sharpe']:+.4f} "
            f"fitness={s['fitness']:+.4f} "
            f"turnover={s['turnover']:.4f})"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
