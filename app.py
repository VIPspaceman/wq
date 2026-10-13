# app.py
# -*- coding: utf-8 -*-

"""
WorldQuant BRAIN Alpha Mining — 命令行入口。

Step 10c
--------
- 根据 AgentSettings + 环境变量决定是否启用 AI Refine Agent
- AI 事件（AI_REFINE_STARTED / SUCCEEDED / FAILED / EXHAUSTED）
  通过 Telegram 通知（若 enabled）

用法
----
    python app.py
    python app.py --experiment-id exp_...
    python app.py --resume exp_...
    python app.py --list-experiments
    python app.py --base-dir ./runs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from config.logging import setup_logging
from config.settings import AppSettings
from search.coordinator import MiningConfig
from search.miner import build_agent, build_mining_config, mine
from search.resume import (
    build_state_from_checkpoint,
    build_state_from_db,
)
from storage.checkpoint import CheckpointStore
from storage.database import WQDatabase
from storage.experiment import (
    ExperimentLayout,
    ExperimentMeta,
    ExperimentRepository,
    new_experiment_id,
)
from storage.store import MiningStore


logger = logging.getLogger(__name__)


# ============================================================
# CLI
# ============================================================

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="WorldQuant BRAIN Alpha Mining",
    )
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="常驻模式：连续跑，自动 checkpoint",
    )
    return parser.parse_args(argv)


# ============================================================
# Capability Discovery
# ============================================================

def _maybe_load_remote_operators(settings: AppSettings) -> None:
    cap = settings.capability
    if not cap.use_remote_operators:
        logger.info("算子池来源: 硬编码默认值")
        return
    try:
        from brain import create_authenticated_session
        from alpha.operators import load_from_api
        logger.info("尝试从 BRAIN API 加载算子池 ...")
        session = create_authenticated_session(
            settings.credentials, settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
        catalog = load_from_api(
            session, settings.endpoints, settings.polling,
        )
        if catalog is not None:
            logger.info("✅ 远程算子池已启用: %s", catalog.summary())
        else:
            logger.warning("⚠️ 远程算子池加载失败")
    except Exception:
        logger.exception("⚠️ 加载远程算子池异常")


def _maybe_load_remote_data_fields(settings: AppSettings) -> None:
    cap = settings.capability
    if not cap.use_remote_data_fields:
        logger.info("字段池来源: 硬编码默认值")
        return
    datasets = list(cap.data_field_datasets or [])
    if not datasets:
        logger.warning("⚠️ data_field_datasets 为空")
        return
    try:
        from brain import (
            create_authenticated_session,
            fetch_data_fields,
        )
        from alpha.fields import (
            catalog_from_api,
            set_default_catalog,
        )
        logger.info(
            "尝试从 BRAIN API 加载字段池 (datasets=%s) ...",
            datasets,
        )
        session = create_authenticated_session(
            settings.credentials, settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
        all_fields = []
        seen_ids: set = set()
        for ds in datasets:
            fields = fetch_data_fields(
                session, settings.endpoints, settings.polling,
                region=settings.simulation.region,
                delay=settings.simulation.delay,
                universe=settings.simulation.universe,
                instrument_type=settings.simulation.instrument_type,
                dataset_id=ds,
                max_pages=cap.data_field_max_pages,
            )
            for f in fields:
                fid = str(f.get("id") or f.get("name") or "").strip()
                if not fid or fid in seen_ids:
                    continue
                seen_ids.add(fid)
                all_fields.append(f)
        if not all_fields:
            logger.warning("⚠️ 远程字段加载为空")
            return
        catalog = catalog_from_api(
            all_fields,
            max_fields=cap.data_field_max_fields,
            min_coverage=cap.data_field_min_coverage,
        )
        set_default_catalog(catalog)
        logger.info("✅ 远程字段池已启用: %s", catalog.summary())
    except Exception:
        logger.exception("⚠️ 加载远程字段池异常")


# ============================================================
# 通知
# ============================================================

def _build_event_hook(settings: AppSettings):
    if not settings.telegram.enabled:
        return None
    from notification.events import EventNotifier
    from notification.telegram import TelegramNotifier
    notifier = TelegramNotifier(
        bot_token=settings.telegram.bot_token or "",
        chat_id=settings.telegram.chat_id or "",
    )
    return EventNotifier(notifier, verbose=settings.telegram.verbose)


# ============================================================
# meta
# ============================================================

def _config_snapshot(cfg: MiningConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def _update_meta_from_state(meta: ExperimentMeta, state) -> None:
    meta.generation = state.generation
    meta.total_evaluated = len(state.evaluated_hashes)
    meta.archive_size = len(state.archive)
    meta.counts = state.archive.counts()
    best = state.archive.best_by_score(1)
    if best:
        r = best[0]
        meta.best_alpha = {
            "expression": r.expression,
            "alpha_id": r.alpha_id,
            "score": r.score,
            "sharpe": r.sharpe,
            "fitness": r.fitness,
            "turnover": r.turnover,
            "classification": r.classification.value,
            "tags": list(r.tags),
        }
    if state.final_concurrency is not None:
        meta.final_concurrency = state.final_concurrency


# ============================================================
# list
# ============================================================

def _cmd_list(repo: ExperimentRepository) -> None:
    metas = repo.list_metas()
    print()
    print("=" * 80)
    print(f"Experiments ({len(metas)})")
    print("=" * 80)
    for m in metas:
        best = m.best_alpha or {}
        best_str = ""
        if best:
            tags = best.get("tags", [])
            tag_str = f"[{','.join(tags)}] " if tags else ""
            best_str = (
                f"{tag_str}"
                f"best={best.get('classification', '?'):9s} "
                f"score={best.get('score', 0):+.4f} "
                f"{best.get('expression', '')}"
            )
        print(
            f"{m.experiment_id} | "
            f"{m.status:9s} | "
            f"gen={m.generation:>3d} | "
            f"archive={m.archive_size:>3d} | "
            f"cc={m.final_concurrency} | "
            f"{best_str}"
        )


# ============================================================
# resume
# ============================================================

def _cmd_resume(
    settings: AppSettings,
    repo: ExperimentRepository,
    layout: ExperimentLayout,
    experiment_id: str,
) -> None:
    meta = repo.load_meta(experiment_id)
    if meta is None:
        print(f"❌ 找不到 experiment: {experiment_id}", file=sys.stderr)
        sys.exit(2)
    exp_dir = layout.experiment_dir(experiment_id)
    ckpt_path = layout.checkpoint_path(experiment_id)

    # -------- 优先从 DB 恢复 --------
    initial_state = None
    concurrency_current = None
    try:
        db = WQDatabase("data/wq.db")
        db.init_schema()
        initial_state = build_state_from_db(db, experiment_id)
        if initial_state is not None:
            logger.info("从 DB 恢复 state")
            try:
                exp_row = db.get_experiment(experiment_id)
                if exp_row is not None:
                    concurrency_current = exp_row.get(
                        "concurrency_current"
                    )
            except Exception:
                logger.exception("读取 experiment 并发度失败")
        db.close()
    except Exception:
        logger.exception("从 DB 恢复失败，回退到 checkpoint 文件")
        initial_state = None

    # -------- 回退到文件 --------
    if initial_state is None:
        ckpt = CheckpointStore(ckpt_path).load()
        if ckpt is None:
            print(
                "❌ DB 和 checkpoint 都无法恢复",
                file=sys.stderr,
            )
            sys.exit(2)
        initial_state = build_state_from_checkpoint(ckpt)
        concurrency_current = ckpt.concurrency_current

    print()
    print("=" * 80)
    print(f"Resume experiment: {experiment_id}")
    print("=" * 80)
    print(f"  generation       : {initial_state.generation}")
    print(f"  population       : {len(initial_state.population)}")
    print(f"  archive          : {len(initial_state.archive)}")
    print(
        f"  evaluated_hashes : "
        f"{len(initial_state.evaluated_hashes)}"
    )
    print(f"  concurrency_curr : {concurrency_current}")

    settings.paths.base_dir = exp_dir
    hook = _build_event_hook(settings)
    agent = build_agent(settings)
    store = MiningStore(
        settings.paths,
        elite_size=settings.evolution.elite_size,
    )
    store.set_experiment(experiment_id)

    meta.status = "RUNNING"
    repo.save_meta(meta)

    try:
        state = mine(
            settings,
            on_event=hook,
            store=store,
            initial_state=initial_state,
            initial_concurrency=concurrency_current,
            agent=agent,
        )
        meta.status = "COMPLETED"
    except Exception:
        meta.status = "FAILED"
        _update_meta_from_state(meta, initial_state)
        repo.save_meta(meta)
        raise

    _update_meta_from_state(meta, state)
    repo.save_meta(meta)
    _print_summary(state, settings)


# ============================================================
# new
# ============================================================

def _cmd_new(
    settings: AppSettings,
    repo: ExperimentRepository,
    layout: ExperimentLayout,
    experiment_id: Optional[str],
) -> None:
    if experiment_id is None:
        experiment_id = new_experiment_id()
    exp_dir = layout.experiment_dir(experiment_id)
    settings.paths.base_dir = exp_dir

    cfg = build_mining_config(settings)
    meta = ExperimentMeta.new(
        experiment_id=experiment_id,
        random_seed=settings.evolution.random_seed,
        config_snapshot=_config_snapshot(cfg),
    )
    repo.save_meta(meta)

    print()
    print("=" * 80)
    print(f"New experiment: {experiment_id}")
    print("=" * 80)
    print(f"  dir: {exp_dir}")

    hook = _build_event_hook(settings)
    agent = build_agent(settings)
    store = MiningStore(
        settings.paths,
        elite_size=settings.evolution.elite_size,
    )
    store.set_experiment(experiment_id)

    try:
        state = mine(
            settings,
            on_event=hook,
            store=store,
            agent=agent,
        )
        meta.status = "COMPLETED"
    except Exception:
        meta.status = "FAILED"
        repo.save_meta(meta)
        raise

    _update_meta_from_state(meta, state)
    repo.save_meta(meta)
    _print_summary(state, settings)


# ============================================================
# 摘要
# ============================================================

def _print_summary(state, settings: AppSettings) -> None:
    print()
    print("=" * 80)
    print("Mining complete")
    print("=" * 80)
    print(f"generations      : {state.generation + 1}")
    print(
        f"total evaluated  : {len(state.evaluated_hashes)} "
        f"(真正提交 sim: {state.submitted_count})"
    )
    print(f"archive size     : {len(state.archive)}")
    print(f"final concurrency: {state.final_concurrency}")
    print()

    print("Counts:")
    for k, v in state.archive.counts().items():
        print(f"  {k:10s} : {v}")

    # AI_FIXED 统计
    all_records = state.archive.all()
    ai_refined = [r for r in all_records if r.has_tag("AI_REFINED")]
    ai_fixed = [r for r in all_records if r.has_tag("AI_FIXED")]
    print()
    print("AI Refine:")
    print(f"  AI_REFINED  : {len(ai_refined)}")
    print(f"  AI_FIXED    : {len(ai_fixed)}")

    top = state.archive.best_by_score(10)
    if top:
        print()
        print("Top 10 by score:")
        for i, r in enumerate(top, start=1):
            tag_str = f"[{','.join(r.tags)}] " if r.tags else ""
            print(
                f"  {i:02d}. {tag_str}"
                f"[{r.classification.value:9s}] "
                f"score={r.score:+.4f} "
                f"sharpe={r.sharpe:+.4f} "
                f"fitness={r.fitness:+.4f} "
                f"turnover={r.turnover:.4f} | "
                f"{r.expression}"
            )

    print()
    print("输出目录:")
    print(f"  {os.path.abspath(settings.paths.base_dir)}")


# ============================================================
# main
# ============================================================

def main(argv=None) -> None:
    args = _parse_args(argv)

    layout = ExperimentLayout(args.base_dir)
    repo = ExperimentRepository(layout)

    if args.list_experiments:
        # list 模式不需要 settings，用默认级别
        setup_logging(level=logging.INFO)
        _cmd_list(repo)
        return

    # 先构造 settings，再按 settings.logging.level 初始化日志
    settings = AppSettings.from_env()
    setup_logging(level=settings.logging.level)

    _maybe_load_remote_operators(settings)
    _maybe_load_remote_data_fields(settings)

    if args.daemon:
        from daemon.runner import Runner
        Runner(
            settings,
            base_dir=args.base_dir,
            experiment_id=args.experiment_id,
        ).run_forever()
        return

    if args.resume:
        _cmd_resume(settings, repo, layout, args.resume)
    else:
        _cmd_new(settings, repo, layout, args.experiment_id)


if __name__ == "__main__":
    main()
