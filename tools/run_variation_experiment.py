# -*- coding: utf-8 -*-

"""
变体生成实验入口。

流程
----
    1. 从 DB 拉 gold 种子（WQDatabase.get_gold_seeds）
    2. 用 VariationGenerator 从每个种子生成变体
    3. dry-run：输出到 stdout
       非 dry-run：写入 /tmp/variation_candidates.json（暂不提交 sim）

用法
----
    python tools/run_variation_experiment.py
    python tools/run_variation_experiment.py --limit 25
    python tools/run_variation_experiment.py --dry-run
    python tools/run_variation_experiment.py --seed 42
    python tools/run_variation_experiment.py --db data/wq.db

约束
----
- 不提交 sim（非 dry-run 也只写 JSON）
- 不改 variation_generator.py
"""


import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让脚本在 tools/ 目录下也能 import 项目根下的包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from storage.database import WQDatabase
from variation_generator import (
    STATUS_OK,
    STATUS_REJECTED,
    Seed,
    VariationGenerator,
)


def _load_remote_catalog() -> bool:
    """
    加载远程算子池（合并进本地 catalog）。

    必须在解析种子之前调用，否则含远程算子
    （ts_av_diff / ts_decay_linear 等）的种子会解析失败。

    失败时返回 False（不中断，降级用本地 catalog）。
    """
    try:
        from config.settings import AppSettings
        from brain.authentication import create_authenticated_session
        from alpha.operators import load_from_api
    except ImportError:
        logging.getLogger(__name__).warning(
            "无法导入远程 catalog 依赖，使用本地 catalog"
        )
        return False

    try:
        settings = AppSettings.from_env()
    except Exception as e:
        logging.getLogger(__name__).warning(
            "加载配置失败，使用本地 catalog: %s", e
        )
        return False

    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "BRAIN 登录失败，使用本地 catalog: %s", e
        )
        return False

    try:
        catalog = load_from_api(
            session,
            settings.endpoints,
            settings.polling,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "加载远程算子池异常，使用本地 catalog: %s", e
        )
        return False

    if catalog is None:
        logging.getLogger(__name__).warning(
            "远程算子池为空，使用本地 catalog"
        )
        return False

    logging.getLogger(__name__).info(
        "远程算子池已加载: %s", catalog.summary()
    )
    return True


DEFAULT_DB = "data/wq.db"
DEFAULT_LIMIT = 25
DEFAULT_SEED = 20261006
DEFAULT_OUTPUT = "/tmp/variation_candidates.json"

# 提交模式常量
SELF_CORR_SLEEP = 5.0
SELF_CORR_MAX = 50


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_variation_experiment",
        description="从 gold 种子生成变体",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"生成 N 个变体（默认 {DEFAULT_LIMIT}）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只生成不写文件",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"随机种子（默认 {DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"数据库路径（默认 {DEFAULT_DB}）",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"输出 JSON 路径（默认 {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="真正提交 sim",
    )
    parser.add_argument(
        "--experiment-id", type=str, default=None,
        help="实验 ID（默认 exp_variation_<时间戳>）",
    )
    return parser.parse_args(argv)


def _load_seeds(db_path: str) -> List[Seed]:
    """从 DB 拉 gold 种子，转成 Seed 列表。"""
    db = WQDatabase(db_path)
    try:
        rows = db.get_gold_seeds()
    finally:
        db.close()

    out: List[Seed] = []
    for r in rows:
        expr = str(r.get("expression") or "")
        if not expr:
            continue
        out.append(Seed(
            alpha_id=str(r.get("alpha_id") or ""),
            expression=expr,
            seed_tier="gold",
        ))
    return out


def _new_experiment_id() -> str:
    """生成默认实验 ID。"""
    return "exp_variation_" + time.strftime("%Y%m%d_%H%M%S")


def _submit_and_collect(
    variants: List[Any],
    args: argparse.Namespace,
) -> int:
    """
    提交变体到 BRAIN，写 DB，查 self-corr，打印统计。

    复用：
        - search.miner.build_executor
        - storage.store.MiningStore
        - evaluation.record.build_record
        - brain.client.BrainClient.check_alpha
    """
    from config.settings import AppSettings
    from execution.task import Task, TaskState
    from search.miner import build_executor
    from storage.store import MiningStore

    # -------- 1. experiment_id --------
    experiment_id = args.experiment_id or _new_experiment_id()
    print(f"实验 ID: {experiment_id}")

    # -------- 2. settings --------
    try:
        settings = AppSettings.from_env()
    except Exception as e:
        print(f"❌ 加载配置失败: {e}", file=sys.stderr)
        return 2

    exp_dir = os.path.join("experiments", experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    settings.paths.base_dir = exp_dir

    # -------- 3. MiningStore --------
    store = MiningStore(
        settings.paths,
        elite_size=settings.evolution.elite_size,
    )
    store.set_experiment(experiment_id)

    # -------- 4. executor --------
    try:
        executor = build_executor(
            settings,
            on_task_complete=store.on_task_complete,
        )
    except Exception as e:
        print(f"❌ 构建 executor 失败: {e}", file=sys.stderr)
        return 2

    # -------- 5. variants → tasks --------
    variant_by_expr: Dict[str, Any] = {}
    tasks: List[Task] = []
    for v in variants:
        t = Task(expression=v.expression, source="VARIATION")
        tasks.append(t)
        variant_by_expr[v.expression] = v

    print(f"提交 {len(tasks)} 个变体 ...")

    # -------- 6. 提交 --------
    try:
        executor.run(tasks)
    except Exception:
        logging.getLogger(__name__).exception("executor.run 异常")

    completed = [t for t in tasks if t.state == TaskState.COMPLETED]
    failed = [t for t in tasks if t.state == TaskState.FAILED]

    # -------- 7. self-corr 检查 --------
    self_corr_stats = _check_self_corr_for_variants(
        settings, completed, variant_by_expr,
    )

    # -------- 8. 统计 --------
    print()
    print("=" * 70)
    print("Variation Experiment Results")
    print("=" * 70)
    print(f"Total submitted:  {len(tasks)}")
    print(f"Completed:        {len(completed)}")
    print(f"Failed:           {len(failed)}")
    print()

    _print_grouped_stats(
        tasks, variant_by_expr, self_corr_stats,
    )

    print("=" * 70)
    return 0


def _check_self_corr_for_variants(
    settings,
    completed: List[Any],
    variant_by_expr: Dict[str, Any],
) -> Dict[str, str]:
    """
    对已完成的变体调 /check 拿 self-corr。

    返回 {expression: self_corr_result}。
    """
    from brain.authentication import create_authenticated_session
    from brain.client import BrainClient
    from evaluation.official_checks import parse_official_checks

    out: Dict[str, str] = {}

    targets = []
    for t in completed:
        if not t.result:
            continue
        alpha_id = str(t.result.get("_alpha_id") or "")
        if not alpha_id:
            continue
        targets.append((t.expression, alpha_id))

    if not targets:
        return out

    targets = targets[:SELF_CORR_MAX]

    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
        client = BrainClient(
            session, settings.endpoints, settings.polling,
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "self-corr 检查：登录失败"
        )
        return out

    try:
        for i, (expr, alpha_id) in enumerate(targets, start=1):
            data = client.check_alpha(alpha_id)
            if not data:
                result = "ERROR"
            else:
                checks = parse_official_checks(data)
                result = checks.get("SELF_CORRELATION", "UNKNOWN")
            out[expr] = result
            print(
                f"  self-corr [{i}/{len(targets)}] "
                f"{alpha_id} → {result}"
            )
            if i < len(targets):
                time.sleep(SELF_CORR_SLEEP)
    finally:
        try:
            session.close()
        except Exception:
            pass

    return out


def _print_grouped_stats(
    tasks: List[Any],
    variant_by_expr: Dict[str, Any],
    self_corr_stats: Dict[str, str],
) -> None:
    """按变异类型分组打印统计。"""
    from collections import defaultdict
    from evaluation.official_checks import all_checks_pass

    groups: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"submit": 0, "seven": 0, "selfcorr": 0}
    )

    for t in tasks:
        v = variant_by_expr.get(t.expression)
        if v is None:
            continue
        mut = v.mutation_type
        groups[mut]["submit"] += 1

        if t.result and all_checks_pass(
            t.result, ignore_pending=True,
        ):
            groups[mut]["seven"] += 1

        sc = self_corr_stats.get(t.expression)
        if sc == "PASS":
            groups[mut]["selfcorr"] += 1

    print("按变异类型分组：")
    for mut in sorted(groups.keys()):
        g = groups[mut]
        print(
            f"  {mut:14s}: submit={g['submit']} / "
            f"7check={g['seven']} / selfcorr={g['selfcorr']}"
        )
    print()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args(argv)

    if not os.path.exists(args.db):
        print(f"❌ 数据库不存在: {args.db}", file=sys.stderr)
        return 2

    # -------- 0. 加载远程算子池（必须在解析种子之前） --------
    _load_remote_catalog()

    # -------- 1. 拉 gold 种子 --------
    print("开始解析种子 ...")
    try:
        seeds = _load_seeds(args.db)
    except Exception as e:
        print(f"❌ 读取 gold 种子失败: {e}", file=sys.stderr)
        return 2

    if not seeds:
        print("❌ 没有 gold 种子（seed_tier='gold'）")
        print("提示：先运行 update_seed_tiers() 或 baseline_stats.py")
        return 1

    print(f"加载 gold 种子: {len(seeds)} 条")

    # -------- 2. 生成变体 --------
    gen = VariationGenerator(random_seed=args.seed)
    ok, rejected = gen.generate_with_rejects(
        seeds, max_total=args.limit,
    )

    total = len(ok) + len(rejected)

    # -------- 3. 输出 --------
    print()
    print("=" * 70)
    print("VARIATION EXPERIMENT")
    print("=" * 70)
    print(f"通过 validate_node 的变体数: {len(ok)} / {total}")
    print(f"被拒: {len(rejected)}")
    print()

    if ok:
        print("[有效变体]")
        for i, v in enumerate(ok, start=1):
            print(
                f"  {i:02d}. {v.variant_id[:8]} | "
                f"parent={v.parent_alpha_id} | "
                f"{v.mutation_type} | "
                f"{v.expression}"
            )
        print()

    if rejected:
        print("[被拒变体]")
        for i, v in enumerate(rejected, start=1):
            print(
                f"  {i:02d}. {v.mutation_type} | "
                f"{v.reject_reason} | "
                f"{v.expression}"
            )
        print()

    # -------- 4. 提交模式 --------
    if args.submit:
        return _submit_and_collect(ok, args)

    # -------- 5. 写 JSON（非 dry-run） --------
    if args.dry_run:
        print("(dry-run) 未写文件")
        return 0

    payload: Dict[str, Any] = {
        "random_seed": args.seed,
        "seeds": [
            {
                "alpha_id": s.alpha_id,
                "expression": s.expression,
            }
            for s in seeds
        ],
        "variants": [v.to_dict() for v in ok],
        "rejected": [v.to_dict() for v in rejected],
    }

    try:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)),
            exist_ok=True,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"JSON 已写入: {args.output}")
    except Exception as e:
        print(f"❌ 写 JSON 失败: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
# -*- coding: utf-8 -*-

"""
变体生成实验入口。

流程
----
    1. 从 DB 拉 gold 种子（WQDatabase.get_gold_seeds）
    2. 用 VariationGenerator 从每个种子生成变体
    3. dry-run：输出到 stdout
       非 dry-run：写入 /tmp/variation_candidates.json（暂不提交 sim）
       --submit：真正提交 sim，写 DB，查 self-corr

用法
----
    python tools/run_variation_experiment.py
    python tools/run_variation_experiment.py --limit 25
    python tools/run_variation_experiment.py --dry-run
    python tools/run_variation_experiment.py --seed 42
    python tools/run_variation_experiment.py --db data/wq.db
    python tools/run_variation_experiment.py --submit
    python tools/run_variation_experiment.py --submit --experiment-id exp_variation_test1

约束
----
- 默认不提交 sim（非 dry-run 也只写 JSON）
- --submit 才真正提交
- 不改 variation_generator.py
"""


import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让脚本在 tools/ 目录下也能 import 项目根下的包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from storage.database import WQDatabase
from variation_generator import (
    STATUS_OK,
    STATUS_REJECTED,
    Seed,
    VariationGenerator,
)


# ============================================================
# 提交模式常量
# ============================================================

SUBMIT_TIMEOUT_SECONDS = 60 * 60   # 单次实验最多 60 分钟
SELF_CORR_SLEEP = 5.0              # self-corr 检查间隔
SELF_CORR_MAX = 50                 # 单次实验最多查多少条 self-corr


def _load_remote_catalog() -> bool:
    """
    加载远程算子池（合并进本地 catalog）。

    必须在解析种子之前调用，否则含远程算子
    （ts_av_diff / ts_decay_linear 等）的种子会解析失败。

    失败时返回 False（不中断，降级用本地 catalog）。
    """
    try:
        from config.settings import AppSettings
        from brain.authentication import create_authenticated_session
        from alpha.operators import load_from_api
    except ImportError:
        logging.getLogger(__name__).warning(
            "无法导入远程 catalog 依赖，使用本地 catalog"
        )
        return False

    try:
        settings = AppSettings.from_env()
    except Exception as e:
        logging.getLogger(__name__).warning(
            "加载配置失败，使用本地 catalog: %s", e
        )
        return False

    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "BRAIN 登录失败，使用本地 catalog: %s", e
        )
        return False

    try:
        catalog = load_from_api(
            session,
            settings.endpoints,
            settings.polling,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "加载远程算子池异常，使用本地 catalog: %s", e
        )
        return False

    if catalog is None:
        logging.getLogger(__name__).warning(
            "远程算子池为空，使用本地 catalog"
        )
        return False

    logging.getLogger(__name__).info(
        "远程算子池已加载: %s", catalog.summary()
    )
    return True


DEFAULT_DB = "data/wq.db"
DEFAULT_LIMIT = 25
DEFAULT_SEED = 20261006
DEFAULT_OUTPUT = "/tmp/variation_candidates.json"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_variation_experiment",
        description="从 gold 种子生成变体",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"生成 N 个变体（默认 {DEFAULT_LIMIT}）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只生成不写文件",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"随机种子（默认 {DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"数据库路径（默认 {DEFAULT_DB}）",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"输出 JSON 路径（默认 {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="真正提交 sim（默认 dry-run 只输出）",
    )
    parser.add_argument(
        "--experiment-id", default=None,
        help="实验 ID（默认 exp_variation_<日期时间>）",
    )
    return parser.parse_args(argv)


def _load_seeds(db_path: str) -> List[Seed]:
    """从 DB 拉 gold 种子，转成 Seed 列表。"""
    db = WQDatabase(db_path)
    try:
        rows = db.get_gold_seeds()
    finally:
        db.close()

    out: List[Seed] = []
    for r in rows:
        expr = str(r.get("expression") or "")
        if not expr:
            continue
        out.append(Seed(
            alpha_id=str(r.get("alpha_id") or ""),
            expression=expr,
            seed_tier="gold",
        ))
    return out


def _new_experiment_id() -> str:
    """生成默认实验 ID。"""
    return "exp_variation_" + time.strftime("%Y%m%d_%H%M%S")


def args_db_path() -> str:
    """返回 DB 路径（与主流程一致）。"""
    return os.environ.get("WQ_DB_PATH", "data/wq.db")


def _run_submit(
    seeds: List[Seed],
    ok: List[Any],
    experiment_id: str,
) -> int:
    """
    提交模式：把变体包装成 Task，用现有 Executor 提交，写 DB。

    直接用 WQDatabase（不走 MiningStore），以便：
        - 写 parent_structure
        - 写 tags（VARIATION / variation_type:XXX / variant_of:XXX）
        - 写 self_corr

    返回 0 成功，非 0 失败。
    """
    from config.settings import AppSettings
    from execution.task import Task, TaskState
    from search.miner import build_executor
    from evaluation.record import build_record
    from alpha.parser import parse_expression
    from core.signature import structure_signature

    # -------- 1. 加载配置 --------
    try:
        settings = AppSettings.from_env()
    except Exception as e:
        print(f"❌ 加载配置失败: {e}", file=sys.stderr)
        return 2

    # -------- 2. 打开 DB --------
    try:
        db = WQDatabase(args_db_path())
        db.init_schema()
        device_id = db.get_or_create_device("variation-tool")
    except Exception as e:
        print(f"❌ 打开 DB 失败: {e}", file=sys.stderr)
        return 2

    # -------- 3. 构建 executor --------
    try:
        executor = build_executor(settings)
    except Exception as e:
        print(f"❌ 构建 executor 失败: {e}", file=sys.stderr)
        db.close()
        return 2

    # -------- 4. 变体 → Task --------
    variant_by_expr: Dict[str, Any] = {}
    tasks: List[Task] = []
    for v in ok:
        t = Task(expression=v.expression, source="VARIATION")
        tasks.append(t)
        variant_by_expr[v.expression] = v

    print(f"提交 {len(tasks)} 个变体 ...")

    # -------- 5. 提交（带超时保护） --------
    start = time.time()
    try:
        executor.run(tasks)
    except Exception:
        logging.getLogger(__name__).exception("executor.run 异常")
    elapsed = time.time() - start

    if elapsed > SUBMIT_TIMEOUT_SECONDS:
        print(
            f"⚠️ 提交耗时 {elapsed:.0f}s 超过 "
            f"{SUBMIT_TIMEOUT_SECONDS}s 上限"
        )

    # -------- 6. 写 DB --------
    completed = [t for t in tasks if t.state == TaskState.COMPLETED]
    failed = [t for t in tasks if t.state == TaskState.FAILED]

    written = 0
    for t in completed:
        v = variant_by_expr.get(t.expression)
        if v is None or t.result is None:
            continue
        try:
            node = parse_expression(t.expression)
            if node is None:
                continue
            record = build_record(node, t.result)

            # ---- 打 tags ----
            record.add_tag("VARIATION")
            record.add_tag(f"variation_type:{v.mutation_type}")
            if v.parent_alpha_id:
                record.add_tag(f"variant_of:{v.parent_alpha_id}")

            # ---- 父种子结构签名 ----
            parent_sig = None
            try:
                pnode = parse_expression(v.parent_expression)
                if pnode is not None:
                    parent_sig = structure_signature(pnode)
            except Exception:
                parent_sig = None

            ok_db = db.save_record(
                record=record.to_dict(),
                experiment_id=experiment_id,
                device_id=device_id,
                generation=0,
                parent_structure=parent_sig,
            )
            if ok_db:
                written += 1
                print(
                    f"  DB 写入 {record.alpha_id} "
                    f"(parent={v.parent_alpha_id}, "
                    f"{v.mutation_type})"
                )
        except Exception:
            logging.getLogger(__name__).exception(
                "写 DB 失败: %s", t.expression[:60]
            )

    print()
    print("=" * 70)
    print("Variation Experiment Results")
    print("=" * 70)
    print(f"Total submitted:  {len(tasks)}")
    print(f"Completed:        {len(completed)}")
    print(f"Failed:           {len(failed)}")
    print(f"DB written:       {written}")
    print()

    # -------- 7. self-corr 检查 --------
    self_corr_stats = _check_self_corr_for_variants(
        settings, completed, variant_by_expr, db,
    )

    # -------- 8. 按变异类型分组统计 --------
    _print_grouped_stats(
        tasks, variant_by_expr, self_corr_stats,
    )

    db.close()
    print("=" * 70)
    return 0


def _check_self_corr_for_variants(
    settings,
    completed: List[Any],
    variant_by_expr: Dict[str, Any],
    db: Any,
) -> Dict[str, str]:
    """
    对已完成的变体调 /check 拿 self-corr，并写回 DB。

    返回 {expression: self_corr_result}。
    """
    from brain.authentication import create_authenticated_session
    from brain.client import BrainClient
    from evaluation.official_checks import parse_official_checks

    out: Dict[str, str] = {}

    # 收集有 alpha_id 的
    targets = []
    for t in completed:
        if not t.result:
            continue
        alpha_id = str(t.result.get("_alpha_id") or "")
        if not alpha_id:
            continue
        targets.append((t.expression, alpha_id))

    if not targets:
        return out

    targets = targets[:SELF_CORR_MAX]

    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
        client = BrainClient(
            session, settings.endpoints, settings.polling,
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "self-corr 检查：登录失败"
        )
        return out

    try:
        for i, (expr, alpha_id) in enumerate(targets, start=1):
            data = client.check_alpha(alpha_id)
            if not data:
                result = "ERROR"
            else:
                checks = parse_official_checks(data)
                result = checks.get("SELF_CORRELATION", "UNKNOWN")
            out[expr] = result

            # ---- 写回 DB ----
            try:
                db.update_self_corr(alpha_id, result)
            except Exception:
                logging.getLogger(__name__).exception(
                    "update_self_corr 失败: %s", alpha_id
                )

            print(
                f"  self-corr [{i}/{len(targets)}] "
                f"{alpha_id} → {result}"
            )
            if i < len(targets):
                time.sleep(SELF_CORR_SLEEP)
    finally:
        try:
            session.close()
        except Exception:
            pass

    return out


def _print_grouped_stats(
    tasks: List[Any],
    variant_by_expr: Dict[str, Any],
    self_corr_stats: Dict[str, str],
) -> None:
    """按变异类型分组打印统计。"""
    from collections import defaultdict

    groups: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"submit": 0, "seven": 0, "selfcorr": 0}
    )

    for t in tasks:
        v = variant_by_expr.get(t.expression)
        if v is None:
            continue
        mut = v.mutation_type
        groups[mut]["submit"] += 1

        # 7 项 PASS 判断
        if t.result:
            from evaluation.official_checks import all_checks_pass
            if all_checks_pass(t.result, ignore_pending=True):
                groups[mut]["seven"] += 1

        # self-corr
        sc = self_corr_stats.get(t.expression)
        if sc == "PASS":
            groups[mut]["selfcorr"] += 1

    print("按变异类型分组：")
    for mut in sorted(groups.keys()):
        g = groups[mut]
        print(
            f"  {mut:14s}: submit={g['submit']} / "
            f"7check={g['seven']} / selfcorr={g['selfcorr']}"
        )
    print()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args(argv)

    if not os.path.exists(args.db):
        print(f"❌ 数据库不存在: {args.db}", file=sys.stderr)
        return 2

    # -------- 0. 加载远程算子池（必须在解析种子之前） --------
    _load_remote_catalog()

    # -------- 1. 拉 gold 种子 --------
    print("开始解析种子 ...")
    try:
        seeds = _load_seeds(args.db)
    except Exception as e:
        print(f"❌ 读取 gold 种子失败: {e}", file=sys.stderr)
        return 2

    if not seeds:
        print("❌ 没有 gold 种子（seed_tier='gold'）")
        print("提示：先运行 update_seed_tiers() 或 baseline_stats.py")
        return 1

    print(f"加载 gold 种子: {len(seeds)} 条")

    # -------- 2. 生成变体 --------
    gen = VariationGenerator(random_seed=args.seed)
    ok, rejected = gen.generate_with_rejects(
        seeds, max_total=args.limit,
    )

    total = len(ok) + len(rejected)

    # -------- 3. 输出 --------
    print()
    print("=" * 70)
    print("VARIATION EXPERIMENT")
    print("=" * 70)
    print(f"通过 validate_node 的变体数: {len(ok)} / {total}")
    print(f"被拒: {len(rejected)}")
    print()

    if ok:
        print("[有效变体]")
        for i, v in enumerate(ok, start=1):
            print(
                f"  {i:02d}. {v.variant_id[:8]} | "
                f"parent={v.parent_alpha_id} | "
                f"{v.mutation_type} | "
                f"{v.expression}"
            )
        print()

    if rejected:
        print("[被拒变体]")
        for i, v in enumerate(rejected, start=1):
            print(
                f"  {i:02d}. {v.mutation_type} | "
                f"{v.reject_reason} | "
                f"{v.expression}"
            )
        print()

    # -------- 4. 提交模式 --------
    if args.submit:
        experiment_id = args.experiment_id or _new_experiment_id()
        print(f"实验 ID: {experiment_id}")
        return _run_submit(seeds, ok, experiment_id)

    # -------- 5. 写 JSON（非 dry-run） --------
    if args.dry_run:
        print("(dry-run) 未写文件")
        return 0

    payload: Dict[str, Any] = {
        "random_seed": args.seed,
        "seeds": [
            {
                "alpha_id": s.alpha_id,
                "expression": s.expression,
            }
            for s in seeds
        ],
        "variants": [v.to_dict() for v in ok],
        "rejected": [v.to_dict() for v in rejected],
    }

    try:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)),
            exist_ok=True,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"JSON 已写入: {args.output}")
    except Exception as e:
        print(f"❌ 写 JSON 失败: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
# -*- coding: utf-8 -*-

"""
变体生成实验入口。

流程
----
    1. 从 DB 拉 gold 种子（WQDatabase.get_gold_seeds）
    2. 用 VariationGenerator 从每个种子生成变体
    3. dry-run：输出到 stdout
       非 dry-run：写入 /tmp/variation_candidates.json（暂不提交 sim）
       --submit：真正提交 sim，写 DB，查 self-corr

用法
----
    python tools/run_variation_experiment.py
    python tools/run_variation_experiment.py --limit 25
    python tools/run_variation_experiment.py --dry-run
    python tools/run_variation_experiment.py --seed 42
    python tools/run_variation_experiment.py --db data/wq.db
    python tools/run_variation_experiment.py --submit
    python tools/run_variation_experiment.py --submit --experiment-id exp_variation_test1

约束
----
- 默认不提交 sim（非 dry-run 也只写 JSON）
- --submit 才真正提交
- 不改 variation_generator.py
"""


import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让脚本在 tools/ 目录下也能 import 项目根下的包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from storage.database import WQDatabase
from variation_generator import (
    STATUS_OK,
    STATUS_REJECTED,
    Seed,
    VariationGenerator,
)


# ============================================================
# 提交模式常量
# ============================================================

SUBMIT_TIMEOUT_SECONDS = 60 * 60   # 单次实验最多 60 分钟
SELF_CORR_SLEEP = 5.0              # self-corr 检查间隔
SELF_CORR_MAX = 50                 # 单次实验最多查多少条 self-corr


def _load_remote_catalog() -> bool:
    """
    加载远程算子池（合并进本地 catalog）。

    必须在解析种子之前调用，否则含远程算子
    （ts_av_diff / ts_decay_linear 等）的种子会解析失败。

    失败时返回 False（不中断，降级用本地 catalog）。
    """
    try:
        from config.settings import AppSettings
        from brain.authentication import create_authenticated_session
        from alpha.operators import load_from_api
    except ImportError:
        logging.getLogger(__name__).warning(
            "无法导入远程 catalog 依赖，使用本地 catalog"
        )
        return False

    try:
        settings = AppSettings.from_env()
    except Exception as e:
        logging.getLogger(__name__).warning(
            "加载配置失败，使用本地 catalog: %s", e
        )
        return False

    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "BRAIN 登录失败，使用本地 catalog: %s", e
        )
        return False

    try:
        catalog = load_from_api(
            session,
            settings.endpoints,
            settings.polling,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "加载远程算子池异常，使用本地 catalog: %s", e
        )
        return False

    if catalog is None:
        logging.getLogger(__name__).warning(
            "远程算子池为空，使用本地 catalog"
        )
        return False

    logging.getLogger(__name__).info(
        "远程算子池已加载: %s", catalog.summary()
    )
    return True


DEFAULT_DB = "data/wq.db"
DEFAULT_LIMIT = 25
DEFAULT_SEED = 20261006
DEFAULT_OUTPUT = "/tmp/variation_candidates.json"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_variation_experiment",
        description="从 gold 种子生成变体",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"生成 N 个变体（默认 {DEFAULT_LIMIT}）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只生成不写文件",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"随机种子（默认 {DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"数据库路径（默认 {DEFAULT_DB}）",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"输出 JSON 路径（默认 {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="真正提交 sim（默认 dry-run 只输出）",
    )
    parser.add_argument(
        "--experiment-id", default=None,
        help="实验 ID（默认 exp_variation_<日期时间>）",
    )
    return parser.parse_args(argv)


def _load_seeds(db_path: str) -> List[Seed]:
    """从 DB 拉 gold 种子，转成 Seed 列表。"""
    db = WQDatabase(db_path)
    try:
        rows = db.get_gold_seeds()
    finally:
        db.close()

    out: List[Seed] = []
    for r in rows:
        expr = str(r.get("expression") or "")
        if not expr:
            continue
        out.append(Seed(
            alpha_id=str(r.get("alpha_id") or ""),
            expression=expr,
            seed_tier="gold",
        ))
    return out


def _new_experiment_id() -> str:
    """生成默认实验 ID。"""
    return "exp_variation_" + time.strftime("%Y%m%d_%H%M%S")


def args_db_path() -> str:
    """返回 DB 路径（与主流程一致）。"""
    return os.environ.get("WQ_DB_PATH", "data/wq.db")


def _run_submit(
    seeds: List[Seed],
    ok: List[Any],
    experiment_id: str,
) -> int:
    """
    提交模式：把变体包装成 Task，用现有 Executor 提交，写 DB。

    直接用 WQDatabase（不走 MiningStore），以便：
        - 写 parent_structure
        - 写 tags（VARIATION / variation_type:XXX / variant_of:XXX）
        - 写 self_corr

    返回 0 成功，非 0 失败。
    """
    from config.settings import AppSettings
    from execution.task import Task, TaskState
    from search.miner import build_executor
    from evaluation.record import build_record
    from alpha.parser import parse_expression
    from core.signature import structure_signature

    # -------- 1. 加载配置 --------
    try:
        settings = AppSettings.from_env()
    except Exception as e:
        print(f"❌ 加载配置失败: {e}", file=sys.stderr)
        return 2

    # -------- 2. 打开 DB --------
    try:
        db = WQDatabase(args_db_path())
        db.init_schema()
        device_id = db.get_or_create_device("variation-tool")
    except Exception as e:
        print(f"❌ 打开 DB 失败: {e}", file=sys.stderr)
        return 2

    # -------- 3. 构建 executor --------
    try:
        executor = build_executor(settings)
    except Exception as e:
        print(f"❌ 构建 executor 失败: {e}", file=sys.stderr)
        db.close()
        return 2

    # -------- 4. 变体 → Task --------
    variant_by_expr: Dict[str, Any] = {}
    tasks: List[Task] = []
    for v in ok:
        t = Task(expression=v.expression, source="VARIATION")
        tasks.append(t)
        variant_by_expr[v.expression] = v

    print(f"提交 {len(tasks)} 个变体 ...")

    # -------- 5. 提交（带超时保护） --------
    start = time.time()
    try:
        executor.run(tasks)
    except Exception:
        logging.getLogger(__name__).exception("executor.run 异常")
    elapsed = time.time() - start

    if elapsed > SUBMIT_TIMEOUT_SECONDS:
        print(
            f"⚠️ 提交耗时 {elapsed:.0f}s 超过 "
            f"{SUBMIT_TIMEOUT_SECONDS}s 上限"
        )

    # -------- 6. 写 DB --------
    completed = [t for t in tasks if t.state == TaskState.COMPLETED]
    failed = [t for t in tasks if t.state == TaskState.FAILED]

    written = 0
    for t in completed:
        v = variant_by_expr.get(t.expression)
        if v is None or t.result is None:
            continue
        try:
            node = parse_expression(t.expression)
            if node is None:
                continue
            record = build_record(node, t.result)

            # ---- 打 tags ----
            record.add_tag("VARIATION")
            record.add_tag(f"variation_type:{v.mutation_type}")
            if v.parent_alpha_id:
                record.add_tag(f"variant_of:{v.parent_alpha_id}")

            # ---- 父种子结构签名 ----
            parent_sig = None
            try:
                pnode = parse_expression(v.parent_expression)
                if pnode is not None:
                    parent_sig = structure_signature(pnode)
            except Exception:
                parent_sig = None

            ok_db = db.save_record(
                record=record.to_dict(),
                experiment_id=experiment_id,
                device_id=device_id,
                generation=0,
                parent_structure=parent_sig,
            )
            if ok_db:
                written += 1
                print(
                    f"  DB 写入 {record.alpha_id} "
                    f"(parent={v.parent_alpha_id}, "
                    f"{v.mutation_type})"
                )
        except Exception:
            logging.getLogger(__name__).exception(
                "写 DB 失败: %s", t.expression[:60]
            )

    print()
    print("=" * 70)
    print("Variation Experiment Results")
    print("=" * 70)
    print(f"Total submitted:  {len(tasks)}")
    print(f"Completed:        {len(completed)}")
    print(f"Failed:           {len(failed)}")
    print(f"DB written:       {written}")
    print()

    # -------- 7. self-corr 检查 --------
    self_corr_stats = _check_self_corr_for_variants(
        settings, completed, variant_by_expr, db,
    )

    # -------- 8. 按变异类型分组统计 --------
    _print_grouped_stats(
        tasks, variant_by_expr, self_corr_stats,
    )

    db.close()
    print("=" * 70)
    return 0


def _check_self_corr_for_variants(
    settings,
    completed: List[Any],
    variant_by_expr: Dict[str, Any],
    db: Any,
) -> Dict[str, str]:
    """
    对已完成的变体调 /check 拿 self-corr，并写回 DB。

    返回 {expression: self_corr_result}。
    """
    from brain.authentication import create_authenticated_session
    from brain.client import BrainClient
    from evaluation.official_checks import parse_official_checks

    out: Dict[str, str] = {}

    # 收集有 alpha_id 的
    targets = []
    for t in completed:
        if not t.result:
            continue
        alpha_id = str(t.result.get("_alpha_id") or "")
        if not alpha_id:
            continue
        targets.append((t.expression, alpha_id))

    if not targets:
        return out

    targets = targets[:SELF_CORR_MAX]

    try:
        session = create_authenticated_session(
            settings.credentials,
            settings.endpoints,
            request_timeout=settings.polling.request_timeout,
        )
        client = BrainClient(
            session, settings.endpoints, settings.polling,
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "self-corr 检查：登录失败"
        )
        return out

    try:
        for i, (expr, alpha_id) in enumerate(targets, start=1):
            data = client.check_alpha(alpha_id)
            if not data:
                result = "ERROR"
            else:
                checks = parse_official_checks(data)
                result = checks.get("SELF_CORRELATION", "UNKNOWN")
            out[expr] = result

            # ---- 写回 DB ----
            try:
                db.update_self_corr(alpha_id, result)
            except Exception:
                logging.getLogger(__name__).exception(
                    "update_self_corr 失败: %s", alpha_id
                )

            print(
                f"  self-corr [{i}/{len(targets)}] "
                f"{alpha_id} → {result}"
            )
            if i < len(targets):
                time.sleep(SELF_CORR_SLEEP)
    finally:
        try:
            session.close()
        except Exception:
            pass

    return out


def _print_grouped_stats(
    tasks: List[Any],
    variant_by_expr: Dict[str, Any],
    self_corr_stats: Dict[str, str],
) -> None:
    """按变异类型分组打印统计。"""
    from collections import defaultdict

    groups: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"submit": 0, "seven": 0, "selfcorr": 0}
    )

    for t in tasks:
        v = variant_by_expr.get(t.expression)
        if v is None:
            continue
        mut = v.mutation_type
        groups[mut]["submit"] += 1

        # 7 项 PASS 判断
        if t.result:
            from evaluation.official_checks import all_checks_pass
            if all_checks_pass(t.result, ignore_pending=True):
                groups[mut]["seven"] += 1

        # self-corr
        sc = self_corr_stats.get(t.expression)
        if sc == "PASS":
            groups[mut]["selfcorr"] += 1

    print("按变异类型分组：")
    for mut in sorted(groups.keys()):
        g = groups[mut]
        print(
            f"  {mut:14s}: submit={g['submit']} / "
            f"7check={g['seven']} / selfcorr={g['selfcorr']}"
        )
    print()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args(argv)

    if not os.path.exists(args.db):
        print(f"❌ 数据库不存在: {args.db}", file=sys.stderr)
        return 2

    # -------- 0. 加载远程算子池（必须在解析种子之前） --------
    _load_remote_catalog()

    # -------- 1. 拉 gold 种子 --------
    print("开始解析种子 ...")
    try:
        seeds = _load_seeds(args.db)
    except Exception as e:
        print(f"❌ 读取 gold 种子失败: {e}", file=sys.stderr)
        return 2

    if not seeds:
        print("❌ 没有 gold 种子（seed_tier='gold'）")
        print("提示：先运行 update_seed_tiers() 或 baseline_stats.py")
        return 1

    print(f"加载 gold 种子: {len(seeds)} 条")

    # -------- 2. 生成变体 --------
    gen = VariationGenerator(random_seed=args.seed)
    ok, rejected = gen.generate_with_rejects(
        seeds, max_total=args.limit,
    )

    total = len(ok) + len(rejected)

    # -------- 3. 输出 --------
    print()
    print("=" * 70)
    print("VARIATION EXPERIMENT")
    print("=" * 70)
    print(f"通过 validate_node 的变体数: {len(ok)} / {total}")
    print(f"被拒: {len(rejected)}")
    print()

    if ok:
        print("[有效变体]")
        for i, v in enumerate(ok, start=1):
            print(
                f"  {i:02d}. {v.variant_id[:8]} | "
                f"parent={v.parent_alpha_id} | "
                f"{v.mutation_type} | "
                f"{v.expression}"
            )
        print()

    if rejected:
        print("[被拒变体]")
        for i, v in enumerate(rejected, start=1):
            print(
                f"  {i:02d}. {v.mutation_type} | "
                f"{v.reject_reason} | "
                f"{v.expression}"
            )
        print()

    # -------- 4. 提交模式 --------
    if args.submit:
        experiment_id = args.experiment_id or _new_experiment_id()
        print(f"实验 ID: {experiment_id}")
        return _run_submit(seeds, ok, experiment_id)

    # -------- 5. 写 JSON（非 dry-run） --------
    if args.dry_run:
        print("(dry-run) 未写文件")
        return 0

    payload: Dict[str, Any] = {
        "random_seed": args.seed,
        "seeds": [
            {
                "alpha_id": s.alpha_id,
                "expression": s.expression,
            }
            for s in seeds
        ],
        "variants": [v.to_dict() for v in ok],
        "rejected": [v.to_dict() for v in rejected],
    }

    try:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)),
            exist_ok=True,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"JSON 已写入: {args.output}")
    except Exception as e:
        print(f"❌ 写 JSON 失败: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
