# -*- coding: utf-8 -*-

"""
同步 BRAIN 已提交 alpha 到本地 DB。

功能
----
从 WorldQuant BRAIN 拉取所有已提交（SUBMITTED）的 alpha，
把本地 SQLite 数据库 ``data/wq.db`` 中对应记录的
``submitted`` 字段更新为 1。

用法
----
    python scripts/sync_submissions.py
    python scripts/sync_submissions.py --dry-run
    python scripts/sync_submissions.py --limit 500

注意
----
- 本脚本 **只查询** BRAIN，不调用任何提交接口。
- 单条 alpha_id 处理失败时打印 warning 并继续。
- 分页有最大页数保护，防止死循环。
"""

from __future__ import annotations

import argparse
import os
import sys

# 把项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brain import create_authenticated_session
from config.settings import AppSettings
from storage.database import WQDatabase


# 分页保护
_MAX_PAGES = 100
_PAGE_SIZE = 100


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sync_submissions.py",
        description="同步 BRAIN 已提交 alpha 到本地 DB",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印，不写 DB",
    )
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="最多拉多少条（默认 1000）",
    )
    return parser.parse_args(argv)


def _fetch_submitted_ids(
    session,
    base_url: str,
    limit: int,
) -> list:
    """分页拉取所有已提交 alpha 的 id。"""
    submitted_ids: list = []
    offset = 0
    page = 0

    while len(submitted_ids) < limit and page < _MAX_PAGES:
        page += 1
        url = (
            f"{base_url}/users/self/alphas"
            f"?limit={_PAGE_SIZE}&offset={offset}"
            f"&order=-dateSubmitted"
        )
        print(f"拉取 offset={offset}...")

        try:
            resp = session.get(url, timeout=30)
        except Exception as e:
            print(f"❌ 请求失败: {e!r}")
            break

        if resp.status_code != 200:
            print(
                f"❌ HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
            break

        try:
            data = resp.json()
        except Exception as e:
            print(f"❌ JSON 解析失败: {e!r}")
            break

        results = data.get("results", []) or []
        if not results:
            break

        for item in results:
            if not isinstance(item, dict):
                continue
            aid = item.get("id")
            if not aid:
                continue
            # 只看已提交的（dateSubmitted 有值）
            if item.get("dateSubmitted") is None:
                continue
            submitted_ids.append(aid)

        # 本页不满 page_size → 最后一页
        if len(results) < _PAGE_SIZE:
            break

        offset += _PAGE_SIZE

    return submitted_ids


def main(argv=None) -> None:
    args = _parse_args(argv)
    dry_run = bool(args.dry_run)
    limit = max(1, int(args.limit))

    # -------- 登录 BRAIN --------
    settings = AppSettings.from_env()
    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
    except Exception as e:
        print(f"❌ BRAIN 登录失败: {e!r}")
        sys.exit(2)

    base_url = settings.endpoints.base_url

    # -------- 拉取已提交 alpha --------
    submitted_ids = _fetch_submitted_ids(session, base_url, limit)
    print(f"从 BRAIN 拉到 {len(submitted_ids)} 个已提交 alpha")

    # -------- 更新 DB --------
    db = WQDatabase("data/wq.db")
    db.init_schema()

    updated = 0
    not_found = 0

    for aid in submitted_ids:
        if dry_run:
            print(f"  [dry-run] would mark {aid}")
            continue
        try:
            ok = db.mark_submitted(aid)
        except Exception as e:
            print(f"  warn: mark_submitted({aid}) 失败: {e!r}")
            continue
        if ok:
            updated += 1
        else:
            not_found += 1

    db.close()

    # -------- 汇总 --------
    print()
    print("=" * 60)
    print(f"BRAIN 已提交总数: {len(submitted_ids)}")
    print(f"DB 更新数:       {updated}")
    print(f"DB 未找到 (其他账号的): {not_found}")
    print("=" * 60)


if __name__ == "__main__":
    main()
