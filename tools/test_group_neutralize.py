# -*- coding: utf-8 -*-

"""
group_neutralize 语法验证脚本（最小测试）。

背景
----
项目之前主动排除 group_* 算子（见 alpha/operators.py 的
_UNSUPPORTED_PREFIXES）。本脚本用于验证 BRAIN 是否支持：

    group_neutralize(<expr>, <group>)

以及 group 参数支持哪些值（sector / industry / subindustry / ...）。

流程
----
    1. 登录 BRAIN
    2. 加载远程算子池（group_neutralize 不在本地 catalog）
    3. 构造表达式 group_neutralize(<expr>, <group>)
    4. dry-run：打印表达式 + validate_node 结果
    5. --submit：提交 1 条 sim，打印 BRAIN 是否接受 + 指标

用法
----
    # dry-run（默认，不提交）
    python tools/test_group_neutralize.py

    # 指定 group 参数
    python tools/test_group_neutralize.py --group industry

    # 指定内层表达式
    python tools/test_group_neutralize.py --expr "ts_delta(close,5)"

    # 真正提交（消耗 1 次 sim 配额）
    python tools/test_group_neutralize.py --submit

约束
----
- 只提交 1 条 sim
- 单次提交最多 5 分钟（超时保护）
- 不改其他文件
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让脚本在 tools/ 目录下也能 import 项目根下的包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


logger = logging.getLogger(__name__)


# ============================================================
# 常量
# ============================================================

DEFAULT_GROUP = "sector"
DEFAULT_EXPR = "ts_delta(close,5)"

# 单次提交超时（秒）
SUBMIT_TIMEOUT_SECONDS = 5 * 60

# 常见 group 候选值（供参考，脚本不自动遍历）
KNOWN_GROUPS = (
    "sector",
    "industry",
    "subindustry",
    "market",
    "country",
    "exchange",
)


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_group_neutralize",
        description="验证 BRAIN 是否支持 group_neutralize 语法",
    )
    parser.add_argument(
        "--group", default=DEFAULT_GROUP,
        help=f"group 参数值（默认 {DEFAULT_GROUP}）",
    )
    parser.add_argument(
        "--expr", default=DEFAULT_EXPR,
        help=f"内层表达式（默认 {DEFAULT_EXPR}）",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="真正提交 sim（默认 dry-run）",
    )
    return parser.parse_args(argv)


# ============================================================
# 登录 + 加载远程算子池
# ============================================================

def _login(settings):
    """登录 BRAIN，返回已认证 Session。"""
    from brain.authentication import create_authenticated_session

    return create_authenticated_session(
        settings.credentials,
        settings.endpoints,
        request_timeout=settings.polling.request_timeout,
    )


def _load_remote_operators(settings, session) -> bool:
    """
    加载远程算子池（合并进本地 catalog）。

    group_neutralize 不在本地 hardcoded catalog 里，
    必须加载远程算子池才能让 validate_node 认识它。

    返回 True 表示加载成功。
    """
    from alpha.operators import load_from_api

    try:
        catalog = load_from_api(
            session, settings.endpoints, settings.polling,
        )
    except Exception:
        logger.exception("加载远程算子池异常")
        return False

    if catalog is None:
        logger.warning("远程算子池为空")
        return False

    logger.info("远程算子池已加载: %s", catalog.summary())
    return True


# ============================================================
# 表达式构造 + 本地校验
# ============================================================

def _build_expression(inner_expr: str, group: str) -> str:
    """构造 group_neutralize(<expr>, <group>)。"""
    return f"group_neutralize({inner_expr},{group})"


def _local_validate(expr: str) -> None:
    """
    用 validate_expression 做本地校验，打印结果。

    注意：即使本地校验失败，BRAIN 仍可能接受该表达式
    （本地 catalog 可能不完整）。因此本步骤只做参考。
    """
    from alpha.validator import validate_expression

    ok, reason = validate_expression(expr)
    if ok:
        print(f"  [本地校验] PASS")
    else:
        print(f"  [本地校验] FAIL: {reason}")
        print(
            "  （本地 catalog 可能不含 group_neutralize，"
            "此结果仅供参考）"
        )


# ============================================================
# 提交
# ============================================================

def _submit_one(settings, expr: str) -> int:
    """
    提交 1 条 sim，打印结果。

    复用 search.miner.build_executor + Executor.run([task])。
    加 5 分钟超时保护（用线程 + join）。

    返回 0 成功，非 0 失败。
    """
    import threading

    from execution.task import Task, TaskState
    from search.miner import build_executor

    # -------- 1. 构建 executor --------
    try:
        executor = build_executor(settings)
    except Exception as e:
        print(f"❌ 构建 executor 失败: {e}", file=sys.stderr)
        return 2

    # -------- 2. 构造 task --------
    task = Task(expression=expr, source="GROUP_TEST")

    # -------- 3. 提交（带超时保护） --------
    print()
    print(f"=== 提交 {expr} ===")
    print("提交中 ...")

    error_holder: Dict[str, Any] = {}

    def _run() -> None:
        try:
            executor.run([task])
        except Exception as e:
            error_holder["error"] = repr(e)

    t = threading.Thread(target=_run, daemon=True)
    start = time.time()
    t.start()
    t.join(timeout=SUBMIT_TIMEOUT_SECONDS)
    elapsed = time.time() - start

    if t.is_alive():
        print(
            f"⚠️ 提交超时（>{SUBMIT_TIMEOUT_SECONDS}s），"
            f"已耗时 {elapsed:.0f}s"
        )
        print("（后台线程仍在运行，进程退出时会终止）")
        return 3

    if "error" in error_holder:
        print(f"❌ 提交异常: {error_holder['error']}")
        return 2

    # -------- 4. 打印结果 --------
    print(f"耗时: {elapsed:.1f}s")
    print(f"task.state: {task.state.value}")
    print(f"task.attempts: {task.attempts}")

    if task.state == TaskState.FAILED:
        print(f"task.error: {task.error}")
        print()
        print("→ BRAIN 拒绝了该表达式（或网络失败）")
        return 1

    if task.state != TaskState.COMPLETED or task.result is None:
        print("→ 未完成，无结果")
        return 1

    # -------- 5. 打印指标 --------
    result = task.result
    is_data = result.get("is", {}) or {}

    print()
    print("result:")
    print(f"  alpha_id: {result.get('_alpha_id', '')}")
    print(f"  sharpe:   {is_data.get('sharpe')}")
    print(f"  fitness:  {is_data.get('fitness')}")
    print(f"  turnover: {is_data.get('turnover')}")

    checks = is_data.get("checks", []) or []
    if checks:
        print("  checks:")
        for c in checks:
            if not isinstance(c, dict):
                continue
            name = c.get("name", "?")
            res = c.get("result", "?")
            print(f"    {name}: {res}")
    else:
        print("  checks: (无)")

    print()
    print("→ BRAIN 接受了该表达式")
    return 0


# ============================================================
# main
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args(argv)

    expr = _build_expression(args.expr, args.group)

    print()
    print("=" * 70)
    print("group_neutralize 语法验证")
    print("=" * 70)
    print(f"  内层表达式: {args.expr}")
    print(f"  group 参数: {args.group}")
    print(f"  完整表达式: {expr}")
    print(f"  模式:       {'提交' if args.submit else 'dry-run'}")
    print()

    # -------- 1. 加载配置 --------
    try:
        from config.settings import AppSettings
        settings = AppSettings.from_env()
    except Exception as e:
        print(f"❌ 加载配置失败: {e}", file=sys.stderr)
        return 2

    # -------- 2. 登录 --------
    try:
        session = _login(settings)
    except Exception as e:
        print(f"❌ BRAIN 登录失败: {e}", file=sys.stderr)
        return 2

    # -------- 3. 加载远程算子池 --------
    _load_remote_operators(settings, session)

    # -------- 4. 本地校验 --------
    _local_validate(expr)

    # -------- 5. dry-run --------
    if not args.submit:
        print()
        print("(dry-run) 未提交。加 --submit 真正提交。")
        print()
        print("常见 group 候选值（供参考）:")
        for g in KNOWN_GROUPS:
            print(f"  - {g}")
        try:
            session.close()
        except Exception:
            pass
        return 0

    # -------- 6. 提交 --------
    try:
        rc = _submit_one(settings, expr)
    finally:
        try:
            session.close()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
