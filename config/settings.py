# config/settings.py
# -*- coding: utf-8 -*-

"""
应用级配置。

配置策略
--------
**敏感信息**（不进代码）→ 环境变量：
    WQ_BRAIN_USERNAME
    WQ_BRAIN_PASSWORD
    WQ_BRAIN_BASE_URL           （可选）
    WQ_TELEGRAM_BOT_TOKEN       （可选）
    WQ_TELEGRAM_CHAT_ID         （可选）
    DEEPSEEK_API_KEY / OPENAI_API_KEY （可选）

**其余所有参数** → 本文件的 dataclass 默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from brain.models import SimulationSettings


# ============================================================
# .env 加载
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ============================================================
# 环境变量键名（仅敏感信息）
# ============================================================

ENV_USERNAME = "WQ_BRAIN_USERNAME"
ENV_PASSWORD = "WQ_BRAIN_PASSWORD"
ENV_BASE_URL = "WQ_BRAIN_BASE_URL"

ENV_TG_BOT_TOKEN = "WQ_TELEGRAM_BOT_TOKEN"
ENV_TG_CHAT_ID = "WQ_TELEGRAM_CHAT_ID"

ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY_WQ"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY_WQ"


# ============================================================
# BRAIN 连接
# ============================================================

@dataclass(frozen=True)
class BrainEndpoints:
    base_url: str = "https://api.worldquantbrain.com"
    authentication: str = "/authentication"
    simulations: str = "/simulations"
    alphas: str = "/alphas"

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"


@dataclass
class BrainCredentials:
    username: str
    password: str


# ============================================================
# 轮询 / 重试
# ============================================================

@dataclass
class PollingSettings:
    poll_interval: float = 3.0
    request_timeout: float = 30.0
    http_retry: int = 3
    simulation_timeout: float = 300.0
    warning_timeout: float = 60.0


# ============================================================
# 进化参数
# ============================================================

@dataclass
class EvolutionSettings:
    population_size: int = 20
    generations: int = 3
    elite_size: int = 8
    mutation_rate: float = 0.75
    crossover_rate: float = 0.50
    reverse_probability: float = 0.15
    max_depth: int = 6
    max_nodes: int = 20
    max_same_family_elite: int = 3
    random_seed: int = 20261006

    # ---- AB：Sign-Flip ----
    enable_sign_flip: bool = False
    max_sign_flips_per_gen: int = 5

    # ---- AB：Combine ----
    enable_combine: bool = True
    # T-4：每代 Combine 次数从 3 提升到 8
    max_combines_per_gen: int = 8
    combine_size: int = 3
    combine_max_depth: int = 10
    combine_max_nodes: int = 30

    # ---- BRAIN 饱和过滤 ----
    enable_brain_saturation_filter: bool = True

    # ---- 字段配额（FIELD-QUOTA）----
    # 单字段在一代种群中最多占的比例（防止单一字段垄断）
    max_single_field_ratio: float = 0.30

    # ---- 初始种群 family 配额（FAMILY-QUOTA）----
    # 初始种群中单 family 最多占的比例（防止初始种群被单一 family 垄断）
    max_same_family_initial: float = 0.30


# ============================================================
# 执行并发
# ============================================================

@dataclass
class ExecutionSettings:
    max_workers: int = 2
    adaptive: bool = True
    min_workers: int = 1
    max_workers_cap: int = 3
    adaptive_success_threshold: int = 0
    max_retries: int = 3
    retry_backoff: float = 1.0
    concurrency_backoff: float = 30.0
    rate_limit_per_second: Optional[float] = None


# ============================================================
# Capability Discovery
# ============================================================

@dataclass
class CapabilitySettings:
    use_remote_operators: bool = True
    use_remote_data_fields: bool = True
    data_field_datasets: List[str] = field(
        default_factory=lambda: ["pv1", "fundamental6"]
    )
    data_field_max_pages: int = 1
    data_field_max_fields: int = 500
    data_field_min_coverage: float = 0.0


# ============================================================
# Agent
# ============================================================

@dataclass
class AgentSettings:
    enabled: bool = True
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout: float = 30.0
    max_retries: int = 2
    max_attempts_per_expression: int = 3
    api_key: Optional[str] = None


# ============================================================
# 自动提交（Auto-Submit）
# ============================================================

@dataclass
class AutoSubmitSettings:
    """
    自动提交框架配置。

    ⚠️ 默认关闭（enabled=False），默认演练模式（dry_run=True）。
    当前阶段只做框架，不做定时器，不自动启用。
    """
    # 总开关（默认关闭）
    enabled: bool = False
    # 演练模式（只筛选不提交，默认开启）
    dry_run: bool = True
    # 每日配额
    max_daily_submissions: int = 1
    # 筛选门槛（与 BRAIN 对齐）
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    min_turnover: float = 0.01
    max_turnover: float = 0.70
    # 结构去重
    structure_dedup: bool = True
    max_per_structure: int = 1
    # 排除已提交的
    exclude_submitted: bool = True


# ============================================================
# 常驻调度器（Daemon）
# ============================================================

@dataclass
class DaemonSettings:
    """
    常驻调度器配置。

    ⚠️ 默认关闭（enabled=False），需手动启用。
    """
    enabled: bool = False              # 默认关闭，需手动启用
    generations_per_cycle: int = 3     # 每个 cycle 跑几代
    max_cycles: int = 100              # 最多跑多少个 cycle
    cycle_sleep_seconds: int = 30      # 每 cycle 之间的间隔
    error_sleep_seconds: int = 60      # 异常后休眠时间
    max_consecutive_errors: int = 5    # 连续异常多少次后退出

    # 时间窗口（可选，默认全天）
    active_hours_start: int = 0        # 0-23
    active_hours_end: int = 24         # 0-24

    # 配额（Step 2 用，先留字段）
    max_daily_submissions: int = 2


# ============================================================
# 日志
# ============================================================

@dataclass
class LoggingSettings:
    """
    日志级别配置。

    level 取值：DEBUG / INFO / WARNING / ERROR / CRITICAL
    由 app.py 读取后传给 config.logging.setup_logging()。
    """
    level: str = "INFO"


# ============================================================
# Telegram 通知
# ============================================================

@dataclass
class TelegramSettings:
    enabled: bool = False
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    verbose: bool = False


# ============================================================
# 输出路径
# ============================================================

@dataclass
class OutputPaths:
    base_dir: str = "."

    results_csv: str = "alpha_results.csv"
    results_json: str = "alpha_results.json"
    candidate_json: str = "candidate_alphas.json"
    strong_json: str = "strong_alphas.json"
    final_json: str = "final_alphas.json"
    elite_json: str = "elite_alphas.json"

    # Step 10d：AI 修复单独存档
    ai_fixed_json: str = "ai_fixed_alphas.json"

    checkpoint: str = "alpha_checkpoint.json"
    pass_txt: str = "pass_alphas.txt"
    error_log: str = "alpha_errors.log"

    def path(self, name: str) -> str:
        return os.path.join(self.base_dir, name)


# ============================================================
# 顶层聚合
# ============================================================

@dataclass
class AppSettings:
    credentials: BrainCredentials
    endpoints: BrainEndpoints
    simulation: SimulationSettings
    polling: PollingSettings
    evolution: EvolutionSettings
    execution: ExecutionSettings
    capability: CapabilitySettings
    agent: AgentSettings
    telegram: TelegramSettings
    paths: OutputPaths
    logging: LoggingSettings = field(
        default_factory=LoggingSettings
    )
    auto_submit: AutoSubmitSettings = field(
        default_factory=AutoSubmitSettings
    )
    daemon: DaemonSettings = field(
        default_factory=DaemonSettings
    )

    @classmethod
    def from_env(cls) -> "AppSettings":
        username = os.environ.get(ENV_USERNAME)
        password = os.environ.get(ENV_PASSWORD)

        if not username or not password:
            raise RuntimeError(
                f"缺少 BRAIN 凭据。请设置环境变量：\n"
                f"  {ENV_USERNAME}\n"
                f"  {ENV_PASSWORD}"
            )

        base_url = os.environ.get(
            ENV_BASE_URL, BrainEndpoints.base_url,
        )

        tg_token = os.environ.get(ENV_TG_BOT_TOKEN)
        tg_chat = os.environ.get(ENV_TG_CHAT_ID)
        telegram = TelegramSettings(
            enabled=bool(tg_token and tg_chat),
            bot_token=tg_token,
            chat_id=tg_chat,
            verbose=False,
        )

        # LLM API Key（可选，缺省时由 agent 自己从环境变量读）
        llm_api_key = (
            os.environ.get(ENV_DEEPSEEK_API_KEY)
            or os.environ.get(ENV_OPENAI_API_KEY)
        )
        agent = AgentSettings(api_key=llm_api_key)

        return cls(
            credentials=BrainCredentials(
                username=username,
                password=password,
            ),
            endpoints=BrainEndpoints(base_url=base_url),
            simulation=SimulationSettings(),
            polling=PollingSettings(),
            evolution=EvolutionSettings(),
            execution=ExecutionSettings(),
            capability=CapabilitySettings(),
            agent=agent,
            telegram=telegram,
            paths=OutputPaths(),
            logging=LoggingSettings(),
        )
