# -*- coding: utf-8 -*-

"""
常驻调度器（Daemon Runner）。

目标
----
让系统能无人值守长时间运行：一次启动，连续跑几天。

行为
----
    - 每次 "cycle" 跑 generations_per_cycle 代（默认 3 代）
    - 每 cycle 结束都保存 checkpoint
    - 崩溃/重启后能从 checkpoint 继续
    - 全局异常不退出主循环（catch → 休眠 → 继续）
    - 停止条件：达到 max_cycles（默认 100）或按 Ctrl+C

职责边界
--------
只负责：
    - 常驻循环编排
    - 调用现有 search.miner.mine
    - 调用现有 storage.checkpoint.CheckpointStore
    - 时间窗口 / 配额 / 异常计数判定

不含：
    - 搜索逻辑（在 search/）
    - 提交配额管理（Step 2）
    - 候选池（Step 3）
    - 监控通知（Step 4）
    - 定时器（cron）

TODO
----
- 时间窗口（active_hours_start/end）当前只做简单判定，
  未处理跨天窗口（如 22:00-06:00）。
- max_daily_submissions 字段已留，但本模块未消费（Step 2 用）。
- 未做"每 cycle 之间检查 DB 里 CANDIDATE/STRONG/FINAL 数量"的
  自动提交触发（Step 2/3）。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from config.settings import AppSettings
from search.coordinator import MiningState
from search.miner import build_agent, build_mining_config, mine
from search.resume import build_state_from_checkpoint
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
# Runner
# ============================================================

class Runner:
    """
    常驻调度器。

    Parameters
    ----------
    settings : AppSettings
        应用配置（含 daemon 子配置）。
    base_dir : str
        实验根目录（experiments/ 的父目录）。
    experiment_id : str | None
        指定实验 ID；None 时自动生成。
    """

    def __init__(
        self,
        settings: AppSettings,
        base_dir: str = ".",
        experiment_id: Optional[str] = None,
    ) -> None:
        self._settings = settings
        self._base_dir = base_dir
        self._daemon = settings.daemon

        self._layout = ExperimentLayout(base_dir)
        self._repo = ExperimentRepository(self._layout)

        self._experiment_id = experiment_id or new_experiment_id()
        self._exp_dir = self._layout.experiment_dir(
            self._experiment_id
        )
        self._ckpt_path = self._layout.checkpoint_path(
            self._experiment_id
        )
        self._ckpt_store = CheckpointStore(self._ckpt_path)

        # 运行状态
        self._cycle_count = 0
        self._consecutive_errors = 0
        self._total_generations = 0
        self._stop_requested = False

    # --------------------------------------------------------
    # 主循环
    # --------------------------------------------------------

    def run_forever(self) -> None:
        """常驻循环。"""
        logger.info(
            "Daemon 启动 | experiment=%s | base_dir=%s | "
            "generations_per_cycle=%d | max_cycles=%d",
            self._experiment_id,
            self._base_dir,
            self._daemon.generations_per_cycle,
            self._daemon.max_cycles,
        )

        # 确保实验目录 + meta 存在
        self._ensure_experiment()

        while self._should_continue():
            cycle_start = time.time()
            self._cycle_count += 1

            logger.info(
                "===== Cycle %d/%d 开始 | 累计代数=%d =====",
                self._cycle_count,
                self._daemon.max_cycles,
                self._total_generations,
            )

            try:
                self._run_one_cycle()
                self._consecutive_errors = 0
            except KeyboardInterrupt:
                logger.info("收到 Ctrl+C，保存 checkpoint 后退出")
                self._save_checkpoint()
                break
            except Exception as e:
                self._consecutive_errors += 1
                logger.exception(
                    "Cycle %d 异常 (%d/%d): %s",
                    self._cycle_count,
                    self._consecutive_errors,
                    self._daemon.max_consecutive_errors,
                    e,
                )
                if (
                    self._consecutive_errors
                    >= self._daemon.max_consecutive_errors
                ):
                    logger.error(
                        "连续异常 %d 次，退出主循环",
                        self._consecutive_errors,
                    )
                    self._save_checkpoint()
                    break
                logger.info(
                    "休眠 %d 秒后重试",
                    self._daemon.error_sleep_seconds,
                )
                time.sleep(self._daemon.error_sleep_seconds)
                continue

            cycle_elapsed = time.time() - cycle_start
            logger.info(
                "===== Cycle %d 完成 | 本 cycle 用时 %s | "
                "累计代数=%d =====",
                self._cycle_count,
                _fmt_duration(cycle_elapsed),
                self._total_generations,
            )

            self._log_db_counts()

            # self-corr 检查（不阻塞 sim，不自动提交）
            self._check_self_corr()

            # 是否还有下一个 cycle
            if not self._should_continue():
                break

            # 时间窗口检查
            if not self._in_active_window():
                sleep_s = self._seconds_until_active()
                logger.info(
                    "当前不在活跃时间窗口，休眠 %s",
                    _fmt_duration(sleep_s),
                )
                self._sleep_interruptible(sleep_s)
                continue

            # cycle 间隔
            if self._daemon.cycle_sleep_seconds > 0:
                logger.info(
                    "休眠 %d 秒后进入下一个 cycle",
                    self._daemon.cycle_sleep_seconds,
                )
                self._sleep_interruptible(
                    self._daemon.cycle_sleep_seconds
                )

        logger.info(
            "Daemon 退出 | 共 %d 个 cycle | 累计代数=%d",
            self._cycle_count,
            self._total_generations,
        )

    # --------------------------------------------------------
    # 单 cycle
    # --------------------------------------------------------

    def _run_one_cycle(self) -> None:
        """
        跑一个 cycle：加载 checkpoint → 跑 N 代 → 保存 checkpoint。
        """
        # -------- 1. 加载上次 checkpoint --------
        initial_state = self._load_checkpoint()
        initial_concurrency = None
        if initial_state is not None:
            initial_concurrency = initial_state.final_concurrency
            logger.info(
                "从 checkpoint 恢复: generation=%d population=%d "
                "archive=%d",
                initial_state.generation,
                len(initial_state.population),
                len(initial_state.archive),
            )

        # -------- 2. 计算本 cycle 的目标代数 --------
        # mine() 会从 start_gen 跑到 settings.evolution.generations。
        # 因此把 generations 设为"当前代数 + 本 cycle 代数"。
        start_gen = (
            initial_state.generation + 1
            if initial_state is not None else 0
        )
        target_gen = (
            start_gen + self._daemon.generations_per_cycle
        )

        # 临时覆盖 generations（不改全局配置对象语义）
        original_generations = self._settings.evolution.generations
        self._settings.evolution.generations = target_gen
        try:
            # -------- 3. 跑 mine --------
            settings = self._settings
            settings.paths.base_dir = self._exp_dir

            hook = self._build_event_hook()
            agent = build_agent(settings)
            store = MiningStore(
                settings.paths,
                elite_size=settings.evolution.elite_size,
            )
            store.set_experiment(self._experiment_id)

            state = mine(
                settings,
                on_event=hook,
                store=store,
                initial_state=initial_state,
                initial_concurrency=initial_concurrency,
                agent=agent,
            )
        finally:
            self._settings.evolution.generations = original_generations

        # -------- 4. 保存 checkpoint --------
        self._save_checkpoint(state)

        # -------- 5. 更新累计代数 --------
        self._total_generations = state.generation + 1

        # -------- 6. 更新 meta --------
        self._update_meta(state)

    # --------------------------------------------------------
    # checkpoint
    # --------------------------------------------------------

    def _load_checkpoint(self) -> Optional[MiningState]:
        """从 checkpoint 文件恢复 MiningState；不存在返回 None。"""
        ckpt = self._ckpt_store.load()
        if ckpt is None:
            return None
        try:
            return build_state_from_checkpoint(ckpt)
        except Exception:
            logger.exception("从 checkpoint 恢复失败，忽略")
            return None

    def _save_checkpoint(
        self,
        state: Optional[MiningState] = None,
    ) -> None:
        """保存 checkpoint。state 为 None 时尝试从文件重读。"""
        try:
            if state is None:
                # 无 state 时，仅确保文件存在（若已有则不动）
                if self._ckpt_store.load() is not None:
                    return
                state = MiningState()
            self._ckpt_store.save(
                state,
                concurrency_current=state.final_concurrency,
                experiment_id=self._experiment_id,
            )
            logger.info(
                "Checkpoint 已保存: %s (generation=%d)",
                self._ckpt_path, state.generation,
            )
        except Exception:
            logger.exception("保存 checkpoint 失败")

    # --------------------------------------------------------
    # 停止条件
    # --------------------------------------------------------

    def _should_continue(self) -> bool:
        """检查是否应该继续跑。"""
        if self._stop_requested:
            return False
        if self._cycle_count >= self._daemon.max_cycles:
            logger.info(
                "达到 max_cycles=%d，停止",
                self._daemon.max_cycles,
            )
            return False
        return True

    # --------------------------------------------------------
    # 时间窗口
    # --------------------------------------------------------

    def _in_active_window(self) -> bool:
        """
        当前是否在活跃时间窗口内。

        TODO: 当前只支持 start < end 的简单窗口（如 0-24、9-18）。
              跨天窗口（如 22-6）未处理。
        """
        start = self._daemon.active_hours_start
        end = self._daemon.active_hours_end
        if start <= 0 and end >= 24:
            return True
        hour = datetime.now().hour
        return start <= hour < end

    def _seconds_until_active(self) -> float:
        """距离下一个活跃窗口开始的秒数（近似）。"""
        start = self._daemon.active_hours_start
        now = datetime.now()
        target = now.replace(
            hour=start, minute=0, second=0, microsecond=0,
        )
        if now.hour >= start:
            target = target + timedelta(days=1)
        return max(0.0, (target - now).total_seconds())

    def _sleep_interruptible(self, seconds: float) -> None:
        """可被 Ctrl+C 打断的休眠。"""
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            logger.info("休眠中被 Ctrl+C 打断")
            self._stop_requested = True

    # --------------------------------------------------------
    # self-corr 检查
    # --------------------------------------------------------

    def _check_self_corr(self) -> None:
        """
        每 cycle 结束后，检查 7 项 PASS 但 self_corr 未知的候选。

        每次最多查 10 条，避免占用太多 API 配额。
        不阻塞 sim，不自动提交。
        """
        max_per_cycle = 10
        sleep_between = 5.0

        try:
            db = WQDatabase("data/wq.db")
            db.init_schema()
        except Exception:
            logger.exception("self-corr 检查：打开 DB 失败")
            return

        try:
            candidates = db.get_pending_self_corr(limit=max_per_cycle)
        except Exception:
            logger.exception("self-corr 检查：读取候选失败")
            db.close()
            return

        if not candidates:
            logger.info("self-corr 检查：无待查候选")
            db.close()
            return

        logger.info(
            "self-corr 检查：本 cycle 待查 %d 条", len(candidates),
        )

        # 登录（复用 authentication）
        try:
            from brain.authentication import (
                create_authenticated_session,
            )
            from brain.client import BrainClient
            session = create_authenticated_session(
                self._settings.credentials,
                self._settings.endpoints,
                request_timeout=(
                    self._settings.polling.request_timeout
                ),
            )
            client = BrainClient(
                session,
                self._settings.endpoints,
                self._settings.polling,
            )
        except Exception:
            logger.exception("self-corr 检查：登录失败")
            db.close()
            return

        passed = 0
        failed = 0
        pending = 0
        errors = 0

        try:
            for i, c in enumerate(candidates, start=1):
                alpha_id = c["alpha_id"]
                if not alpha_id:
                    continue

                data = client.check_alpha(alpha_id)
                if not data:
                    result = "ERROR"
                    errors += 1
                else:
                    checks = _extract_checks_map(data)
                    sc = checks.get("SELF_CORRELATION")
                    if sc == "PASS":
                        result = "PASS"
                        passed += 1
                    elif sc == "FAIL":
                        result = "FAIL"
                        failed += 1
                    elif sc == "PENDING":
                        result = "PENDING"
                        pending += 1
                    else:
                        result = "ERROR"
                        errors += 1

                db.update_self_corr(alpha_id, result)
                logger.info(
                    "self-corr [%d/%d] %s → %s",
                    i, len(candidates), alpha_id, result,
                )

                if i < len(candidates):
                    time.sleep(sleep_between)
        finally:
            try:
                session.close()
            except Exception:
                pass
            db.close()

        logger.info(
            "self-corr 检查完成：PASS=%d FAIL=%d PENDING=%d ERROR=%d",
            passed, failed, pending, errors,
        )

    # --------------------------------------------------------
    # 日志辅助
    # --------------------------------------------------------

    def _log_db_counts(self) -> None:
        """打印 DB 里 CANDIDATE / STRONG / FINAL 数量。"""
        try:
            db = WQDatabase("data/wq.db")
            db.init_schema()
            counts = {}
            for cls in ("CANDIDATE", "STRONG", "FINAL"):
                rows = db.top_alphas(n=100000, classification=cls)
                counts[cls] = len(rows)
            db.close()
            logger.info(
                "DB 统计: CANDIDATE=%d STRONG=%d FINAL=%d",
                counts.get("CANDIDATE", 0),
                counts.get("STRONG", 0),
                counts.get("FINAL", 0),
            )
        except Exception:
            logger.exception("读取 DB 统计失败")

    # --------------------------------------------------------
    # 实验 meta
    # --------------------------------------------------------

    def _ensure_experiment(self) -> None:
        """确保实验目录 + meta 存在。"""
        try:
            self._layout.ensure()
        except Exception:
            logger.exception("创建实验目录失败")

        meta = self._repo.load_meta(self._experiment_id)
        if meta is None:
            cfg = build_mining_config(self._settings)
            from dataclasses import asdict
            meta = ExperimentMeta.new(
                experiment_id=self._experiment_id,
                random_seed=self._settings.evolution.random_seed,
                config_snapshot=asdict(cfg),
            )
            meta.status = "RUNNING"
            self._repo.save_meta(meta)
            logger.info("创建新实验: %s", self._experiment_id)

    def _update_meta(self, state: MiningState) -> None:
        """更新实验 meta。"""
        try:
            meta = self._repo.load_meta(self._experiment_id)
            if meta is None:
                return
            meta.generation = state.generation
            meta.total_evaluated = len(state.evaluated_hashes)
            meta.archive_size = len(state.archive)
            meta.counts = state.archive.counts()
            if state.final_concurrency is not None:
                meta.final_concurrency = state.final_concurrency
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
            self._repo.save_meta(meta)
        except Exception:
            logger.exception("更新 meta 失败")

    # --------------------------------------------------------
    # 事件钩子
    # --------------------------------------------------------

    def _build_event_hook(self):
        """构造事件钩子（Telegram 通知，若启用）。"""
        if not self._settings.telegram.enabled:
            return None
        try:
            from notification.events import EventNotifier
            from notification.telegram import TelegramNotifier
            notifier = TelegramNotifier(
                bot_token=self._settings.telegram.bot_token or "",
                chat_id=self._settings.telegram.chat_id or "",
            )
            return EventNotifier(
                notifier,
                verbose=self._settings.telegram.verbose,
            )
        except Exception:
            logger.exception("构造事件钩子失败")
            return None


# ============================================================
# 辅助
# ============================================================

def _fmt_duration(sec: float) -> str:
    """把秒数格式化为人类可读的时长。"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}h{m}m"
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


def _extract_checks_map(raw_json) -> dict:
    """
    从 raw_json 中提取 {check_name: result}。

    支持 {"is_raw": {...}} / {"is": {...}} / {"checks": [...]} / JSON 字符串。
    不抛异常；无数据返回 {}。
    """
    import json as _json

    if raw_json is None:
        return {}
    if isinstance(raw_json, str):
        if not raw_json.strip():
            return {}
        try:
            raw_json = _json.loads(raw_json)
        except (ValueError, TypeError):
            return {}
    if not isinstance(raw_json, dict):
        return {}

    checks_list = None
    for key in ("is_raw", "is"):
        block = raw_json.get(key)
        if isinstance(block, dict):
            c = block.get("checks")
            if isinstance(c, list):
                checks_list = c
                break
    if checks_list is None:
        c = raw_json.get("checks")
        if isinstance(c, list):
            checks_list = c

    if not checks_list:
        return {}

    out = {}
    for c in checks_list:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "") or "")
        if not name:
            continue
        out[name] = str(c.get("result", "UNKNOWN") or "UNKNOWN")
    return out
