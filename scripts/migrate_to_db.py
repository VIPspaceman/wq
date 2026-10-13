# -*- coding: utf-8 -*-

"""
历史实验数据迁移脚本。

功能
----
将 ``experiments/`` 目录下所有历史实验的 ``alpha_results.json``
导入 SQLite 数据库 ``data/wq.db``。

- 每个子目录视为一个 experiment。
- 读取 ``meta.json`` 获取 random_seed / config_snapshot（可选）。
- 逐条调用 ``WQDatabase.save_record``（内部 INSERT OR IGNORE，天然幂等）。
- 单个坏文件不会终止整个导入流程。

注意
----
本脚本 **只** import 标准库与 ``storage.database``，
不 import 任何其它项目模块，以避免循环导入。
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

# 把项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from storage.database import WQDatabase


def main() -> None:
    db = WQDatabase("data/wq.db")
    db.init_schema()

    try:
        device_name = socket.gethostname() or "unknown-device"
    except Exception:
        device_name = "unknown-device"
    device_id = db.get_or_create_device(device_name)

    exp_root = Path("experiments")
    if not exp_root.exists():
        print("experiments/ 目录不存在")
        db.close()
        return

    for exp_dir in sorted(exp_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_id = exp_dir.name
        results_json = exp_dir / "alpha_results.json"

        if not results_json.exists():
            print(f"  skip {exp_id}: 无 alpha_results.json")
            continue

        # -------- 读 meta.json --------
        meta_path = exp_dir / "meta.json"
        random_seed = 0
        config_snapshot = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                random_seed = meta.get("random_seed", 0)
                config_snapshot = meta.get("config_snapshot", {})
            except Exception as e:
                print(f"  warn: 读 meta 失败 {e}")

        # -------- 创建 experiment 记录 --------
        try:
            db.create_experiment(
                experiment_id=exp_id,
                device_id=device_id,
                random_seed=random_seed,
                config_snapshot=config_snapshot,
            )
        except Exception as e:
            print(f"  warn: create_experiment 失败: {e}")

        # -------- 读 records --------
        try:
            records = json.loads(
                results_json.read_text(encoding="utf-8")
            )
        except Exception as e:
            print(f"  error: 读 records 失败: {e}")
            continue

        if not isinstance(records, list):
            print(f"  skip {exp_id}: records 不是 list")
            continue

        # -------- 逐条导入 --------
        ok = 0
        err = 0
        for rec in records:
            if not isinstance(rec, dict):
                err += 1
                continue
            try:
                db.save_record(
                    record=rec,
                    experiment_id=exp_id,
                    device_id=device_id,
                    generation=-1,  # 历史数据无法定位到具体代
                )
                ok += 1
            except Exception as e:
                err += 1
                print(f"    error: {e}")

        print(f"  {exp_id}: 导入 {ok} 条（失败 {err}）")

    # -------- 总结 --------
    hashes = db.load_all_expr_hashes()
    print()
    print("=" * 60)
    print(f"总 expr_hash 数: {len(hashes)}")
    print(f"总 experiments: {len(db.list_experiments())}")
    print(f"总 evaluations: {len(db.top_alphas(99999))}")
    print("=" * 60)

    db.close()


if __name__ == "__main__":
    main()
