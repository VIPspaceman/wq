# -*- coding: utf-8 -*-

"""
种子变体生成器（独立模块）。

职责边界
--------
只负责：从金种子表达式生成 **可追溯的 AST 级变体**。

不含：
    - BRAIN 模拟（由 execution/ 执行）
    - 提交（由 tools/auto_submit.py 执行）
    - 进化循环（由 evolution/ 执行）
    - 数据库写入（由调用方决定）

变异策略（第一轮）
------------------
    L1_WINDOW   : 替换 ts / ts_binary 节点的 window
    L2_FIELD    : 同类型字段替换
    L3_OPERATOR : 同 arity 算子替换
    L4_SUBTREE  : 替换 ts 节点的 window 子树（保守版）

每个变体只执行一种变异，保留完整变异记录。

质量控制流水线
--------------
    AST 变异 → node_to_expr → parse_expression 复验
      → validate_node（结构/运算符/字段/类型/语义）
      → normalize_expression → expression_hash
      → 过滤父种子 → 过滤 existing_hashes
      → 长度/节点数检查 → 输出 Variant

⚠️ 本地检查通过只代表候选可进入后续流程，
   不代表 BRAIN 会接受，也不代表质量/自相关检查通过。

用法
----
    from variation_generator import (
        Seed, Variant, VariationGenerator, load_gold_seeds,
    )

    seeds = load_gold_seeds("data/wq.db", limit=10)
    gen = VariationGenerator(random_seed=42)
    variants = gen.generate(seeds, max_total=25)
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from alpha.ast import (
    Node,
    clone_node,
    collect_nodes,
    node_count,
    node_depth,
    node_to_expr,
)
from alpha.fields import default_catalog as _field_catalog
from alpha.operators import default_catalog as _op_catalog
from alpha.parser import parse_expression
from alpha.validator import validate_node
from core.hashing import expression_hash, normalize_expression


logger = logging.getLogger(__name__)


# ============================================================
# 常量
# ============================================================

MUT_L1_WINDOW = "L1_WINDOW"
MUT_L2_FIELD = "L2_FIELD"
MUT_L3_OPERATOR = "L3_OPERATOR"
MUT_L4_SUBTREE = "L4_SUBTREE"

ALL_MUTATIONS = (
    MUT_L2_FIELD,
    MUT_L3_OPERATOR,
    MUT_L1_WINDOW,
    MUT_L4_SUBTREE,
)

STATUS_OK = "OK"
STATUS_REJECTED = "REJECTED"

# 默认窗口候选（会与 catalog.windows() 求交集）
_DEFAULT_WINDOW_CANDIDATES: Tuple[int, ...] = (
    2, 3, 5, 10, 15, 20, 30, 60, 120,
)

# L4 子树替换的最大深度（保守）
_L4_MAX_SUBTREE_DEPTH = 3


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Seed:
    """金种子。"""
    alpha_id: str
    expression: str
    expr_hash: str = ""
    seed_tier: str = "gold"
    parent_structure: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.expr_hash and self.expression:
            self.expr_hash = expression_hash(self.expression)


@dataclass
class Variant:
    """一个变体。"""
    variant_id: str
    parent_alpha_id: str
    parent_expression: str
    mutation_type: str
    mutation_params: Dict[str, Any]
    expression: str
    expr_hash: str
    generated_at: str
    status: str = STATUS_OK
    reject_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "parent_alpha_id": self.parent_alpha_id,
            "parent_expression": self.parent_expression,
            "mutation_type": self.mutation_type,
            "mutation_params": dict(self.mutation_params),
            "expression": self.expression,
            "expr_hash": self.expr_hash,
            "generated_at": self.generated_at,
            "status": self.status,
            "reject_reason": self.reject_reason,
        }


# ============================================================
# 生成器
# ============================================================

class VariationGenerator:
    """
    种子变体生成器。

    Parameters
    ----------
    random_seed : int
        固定随机种子，保证可复现。
    max_variants_per_seed : int
        单个种子最多产出多少个变体（含被拒）。
    max_expr_length : int
        表达式最大字符长度。
    max_expr_nodes : int
        AST 最大节点数（默认 20，与 EvolutionSettings.max_nodes 对齐）。
    max_depth : int
        AST 最大深度（默认 6，与 EvolutionSettings.max_depth 对齐）。
    mutation_types : list[str] | None
        启用的变异策略；None 时用 ALL_MUTATIONS。
    """

    def __init__(
        self,
        *,
        random_seed: int = 20261006,
        max_variants_per_seed: int = 5,
        max_expr_length: int = 200,
        max_expr_nodes: int = 20,
        max_depth: int = 6,
        mutation_types: Optional[List[str]] = None,
    ) -> None:
        self._rng = random.Random(random_seed)
        self._random_seed = random_seed
        self._max_per_seed = max(1, max_variants_per_seed)
        self._max_len = max(10, max_expr_length)
        self._max_nodes = max(3, max_expr_nodes)
        self._max_depth = max(1, max_depth)
        self._mutations = list(
            mutation_types if mutation_types else ALL_MUTATIONS
        )

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def generate(
        self,
        seeds: List[Seed],
        *,
        max_total: int = 25,
        existing_hashes: Optional[Set[str]] = None,
    ) -> List[Variant]:
        """
        生成变体。

        Parameters
        ----------
        seeds : list[Seed]
            金种子列表。
        max_total : int
            最多返回多少个 **有效** 变体（status=OK）。
        existing_hashes : set[str] | None
            数据库中已存在的 expr_hash，用于去重。

        Returns
        -------
        list[Variant]
            仅含 status=OK 的变体；被拒的变体不返回
            （如需审计，用 generate_with_rejects）。
        """
        ok, _ = self.generate_with_rejects(
            seeds, max_total=max_total,
            existing_hashes=existing_hashes,
        )
        return ok

    def generate_with_rejects(
        self,
        seeds: List[Seed],
        *,
        max_total: int = 25,
        existing_hashes: Optional[Set[str]] = None,
    ) -> Tuple[List[Variant], List[Variant]]:
        """
        生成变体，同时返回被拒的变体（供审计）。

        Returns
        -------
        (ok_variants, rejected_variants)
        """
        if not seeds:
            logger.warning("generate: 种子列表为空")
            return [], []

        existing = set(existing_hashes or set())
        # 父种子 hash 也加入去重集
        parent_hashes = {s.expr_hash for s in seeds if s.expr_hash}

        ok: List[Variant] = []
        rejected: List[Variant] = []
        seen_hashes: Set[str] = set()

        for seed in seeds:
            if len(ok) >= max_total:
                break

            seed_variants = self._mutate_seed(seed)

            for v in seed_variants:
                if len(ok) >= max_total:
                    break

                # ---- 去重 ----
                if v.expr_hash in seen_hashes:
                    v.status = STATUS_REJECTED
                    v.reject_reason = "duplicate_in_batch"
                    rejected.append(v)
                    continue
                if v.expr_hash in parent_hashes:
                    v.status = STATUS_REJECTED
                    v.reject_reason = "same_as_parent"
                    rejected.append(v)
                    continue
                if v.expr_hash in existing:
                    v.status = STATUS_REJECTED
                    v.reject_reason = "already_in_db"
                    rejected.append(v)
                    continue

                seen_hashes.add(v.expr_hash)
                ok.append(v)

        total_candidates = len(ok) + len(rejected)
        logger.info(
            "生成 %d 个候选，其中 %d 个通过全部校验，"
            "实际变体数 %d（被拒 %d）",
            total_candidates, len(ok), len(ok), len(rejected),
        )
        return ok, rejected

    # --------------------------------------------------------
    # 单种子变异
    # --------------------------------------------------------

    def _mutate_seed(self, seed: Seed) -> List[Variant]:
        """对单个种子执行所有启用的变异策略。"""
        node = parse_expression(seed.expression)
        if node is None:
            logger.warning(
                "无法解析种子表达式: %s", seed.expression[:80],
            )
            return []

        out: List[Variant] = []
        per_seed_count = 0

        for mut in self._mutations:
            if per_seed_count >= self._max_per_seed:
                break
            try:
                if mut == MUT_L1_WINDOW:
                    vs = self.mutate_window(seed, node)
                elif mut == MUT_L2_FIELD:
                    vs = self.mutate_field(seed, node)
                elif mut == MUT_L3_OPERATOR:
                    vs = self.mutate_operator(seed, node)
                elif mut == MUT_L4_SUBTREE:
                    vs = self.mutate_subtree(seed, node)
                else:
                    continue
            except Exception:
                logger.exception("变异 %s 失败: %s", mut, seed.alpha_id)
                continue

            for v in vs:
                if per_seed_count >= self._max_per_seed:
                    break
                out.append(v)
                per_seed_count += 1

        return out

    # --------------------------------------------------------
    # L1：窗口变化
    # --------------------------------------------------------

    def mutate_window(
        self, seed: Seed, node: Node,
    ) -> List[Variant]:
        """替换 ts / ts_binary 节点的 window。"""
        catalog = _op_catalog()
        valid_windows = set(catalog.windows()) or set(
            _DEFAULT_WINDOW_CANDIDATES
        )

        targets = [
            n for n in collect_nodes(node)
            if n.kind in ("ts", "ts_binary")
            and n.window is not None
        ]
        if not targets:
            return []

        out: List[Variant] = []
        for target in targets:
            old_w = target.window
            candidates = [
                w for w in valid_windows if w != old_w
            ]
            if not candidates:
                continue
            new_w = self._rng.choice(candidates)

            new_node = clone_node(node)
            # 在克隆树里找到对应节点（按结构路径）
            path = _find_path(node, target)
            if path is None:
                continue
            new_target = _node_at_path(new_node, path)
            if new_target is None:
                continue
            new_target.window = new_w

            v = self._build_variant(
                seed, new_node, MUT_L1_WINDOW,
                {"old_window": old_w, "new_window": new_w},
            )
            if v is not None:
                out.append(v)

        return out

    # --------------------------------------------------------
    # L2：字段替换
    # --------------------------------------------------------

    def mutate_field(
        self, seed: Seed, node: Node,
    ) -> List[Variant]:
        """同类型字段替换。"""
        catalog = _field_catalog()
        field_types = catalog.field_types()

        targets = [
            n for n in collect_nodes(node)
            if n.kind == "field"
        ]
        if not targets:
            return []

        out: List[Variant] = []
        for target in targets:
            old_name = target.value
            old_type = field_types.get(old_name)
            if old_type is None:
                continue

            same_type = [
                f for f in catalog.by_type(old_type)
                if f != old_name
            ]
            if not same_type:
                continue
            new_name = self._rng.choice(same_type)

            new_node = clone_node(node)
            path = _find_path(node, target)
            if path is None:
                continue
            new_target = _node_at_path(new_node, path)
            if new_target is None:
                continue
            new_target.value = new_name
            new_target.field_type = old_type

            v = self._build_variant(
                seed, new_node, MUT_L2_FIELD,
                {"old_field": old_name, "new_field": new_name,
                 "field_type": old_type},
            )
            if v is not None:
                out.append(v)

        return out

    # --------------------------------------------------------
    # L3：算子替换
    # --------------------------------------------------------

    def mutate_operator(
        self, seed: Seed, node: Node,
    ) -> List[Variant]:
        """同 arity 算子替换。"""
        catalog = _op_catalog()

        out: List[Variant] = []
        for target in collect_nodes(node):
            if target.kind == "unary":
                pool = list(catalog.unary())
            elif target.kind == "ts":
                pool = list(catalog.ts())
            elif target.kind == "binary":
                pool = list(catalog.binary())
            elif target.kind == "ts_binary":
                pool = list(catalog.ts_binary())
            else:
                continue

            old_op = target.value
            candidates = [op for op in pool if op != old_op]
            if not candidates:
                continue
            new_op = self._rng.choice(candidates)

            new_node = clone_node(node)
            path = _find_path(node, target)
            if path is None:
                continue
            new_target = _node_at_path(new_node, path)
            if new_target is None:
                continue
            new_target.value = new_op

            v = self._build_variant(
                seed, new_node, MUT_L3_OPERATOR,
                {"old_op": old_op, "new_op": new_op,
                 "kind": target.kind},
            )
            if v is not None:
                out.append(v)

        return out

    # --------------------------------------------------------
    # L4：子树替换（保守版）
    # --------------------------------------------------------

    def mutate_subtree(
        self, seed: Seed, node: Node,
    ) -> List[Variant]:
        """
        保守版子树替换：仅替换 ts 节点的 window 子树。

        即：把 ts(op, child, w) 的 child 换成另一个同类型子树。
        第一版限制替换深度 <= _L4_MAX_SUBTREE_DEPTH。
        """
        catalog = _op_catalog()
        ts_ops = list(catalog.ts())
        if not ts_ops:
            return []

        targets = [
            n for n in collect_nodes(node)
            if n.kind == "ts"
            and n.child is not None
            and node_depth(n.child) <= _L4_MAX_SUBTREE_DEPTH
        ]
        if not targets:
            return []

        out: List[Variant] = []
        for target in targets:
            old_child = target.child
            old_op = target.value
            # 用同 op 的另一个窗口构造新子树
            valid_windows = set(catalog.windows()) or set(
                _DEFAULT_WINDOW_CANDIDATES
            )
            candidates = [
                w for w in valid_windows if w != target.window
            ]
            if not candidates:
                continue
            new_w = self._rng.choice(candidates)

            new_node = clone_node(node)
            path = _find_path(node, target)
            if path is None:
                continue
            new_target = _node_at_path(new_node, path)
            if new_target is None:
                continue
            # 替换 child 为同 op 不同窗口的子树
            new_target.child = clone_node(old_child)
            new_target.window = new_w

            v = self._build_variant(
                seed, new_node, MUT_L4_SUBTREE,
                {"old_op": old_op, "old_window": target.window,
                 "new_window": new_w},
            )
            if v is not None:
                out.append(v)

        return out

    # --------------------------------------------------------
    # 变体构造 + 质量控制
    # --------------------------------------------------------

    def _build_variant(
        self,
        seed: Seed,
        new_node: Node,
        mutation_type: str,
        params: Dict[str, Any],
    ) -> Optional[Variant]:
        """从变异后的 AST 构造 Variant，并做质量控制。"""
        try:
            expr = node_to_expr(new_node)
        except Exception:
            logger.exception("node_to_expr 失败")
            return None

        # ---- 长度检查 ----
        if len(expr) > self._max_len:
            return self._reject(
                seed, expr, mutation_type, params,
                "expr_too_long",
            )

        # ---- 节点数检查 ----
        if node_count(new_node) > self._max_nodes:
            return self._reject(
                seed, expr, mutation_type, params,
                "too_many_nodes",
            )

        # ---- 语法复验 ----
        reparsed = parse_expression(expr)
        if reparsed is None:
            return self._reject(
                seed, expr, mutation_type, params,
                "reparse_failed",
            )

        # ---- 结构 / 运算符 / 字段 / 类型 / 语义 校验 ----
        # 与 validator 用同一套上限，避免误杀。
        # semantic_filter 的拒绝规则（sign(volume) / rank(rank(x)) 等）
        # 对变体生成是好事，保持开启。
        ok, reason = validate_node(
            reparsed,
            max_depth=self._max_depth,
            max_nodes=self._max_nodes,
        )
        if not ok:
            return self._reject(
                seed, expr, mutation_type, params,
                f"validate_failed: {reason}",
            )

        # ---- 规范化 + hash ----
        normalized = normalize_expression(expr)
        h = expression_hash(normalized)

        return Variant(
            variant_id=str(uuid.uuid4()),
            parent_alpha_id=seed.alpha_id,
            parent_expression=seed.expression,
            mutation_type=mutation_type,
            mutation_params=dict(params),
            expression=expr,
            expr_hash=h,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            status=STATUS_OK,
            reject_reason=None,
        )

    def _reject(
        self,
        seed: Seed,
        expr: str,
        mutation_type: str,
        params: Dict[str, Any],
        reason: str,
    ) -> Variant:
        return Variant(
            variant_id=str(uuid.uuid4()),
            parent_alpha_id=seed.alpha_id,
            parent_expression=seed.expression,
            mutation_type=mutation_type,
            mutation_params=dict(params),
            expression=expr,
            expr_hash=expression_hash(expr),
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            status=STATUS_REJECTED,
            reject_reason=reason,
        )


# ============================================================
# AST 路径工具
# ============================================================

def _find_path(
    root: Node, target: Node,
) -> Optional[List[str]]:
    """
    找到 target 在 root 中的路径。

    路径元素："child" / "left" / "right"。
    找不到返回 None。
    """
    if root is target:
        return []

    for attr in ("child", "left", "right"):
        sub = getattr(root, attr, None)
        if sub is None:
            continue
        p = _find_path(sub, target)
        if p is not None:
            return [attr] + p
    return None


def _node_at_path(
    root: Node, path: List[str],
) -> Optional[Node]:
    """按路径取节点。"""
    cur = root
    for attr in path:
        cur = getattr(cur, attr, None)
        if cur is None:
            return None
    return cur


# ============================================================
# 种子加载
# ============================================================

def load_gold_seeds(
    db_path: str = "data/wq.db",
    limit: int = 20,
) -> List[Seed]:
    """
    从 DB 加载金种子。

    依赖 storage.database.WQDatabase.get_gold_seeds()。
    """
    from storage.database import WQDatabase

    db = WQDatabase(db_path)
    try:
        rows = db.get_gold_seeds(limit=limit)
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
            expr_hash=str(r.get("expr_hash") or ""),
            seed_tier=str(r.get("seed_tier") or "gold"),
            parent_structure=r.get("parent_structure"),
        ))
    return out


# ============================================================
# self-test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 用一条已知可解析的表达式做自测
    test_seed = Seed(
        alpha_id="TEST001",
        expression="rank(ts_delta(close,5))",
    )
    gen = VariationGenerator(random_seed=42)
    ok, rejected = gen.generate_with_rejects(
        [test_seed], max_total=25,
    )
    print(f"OK: {len(ok)}")
    for v in ok:
        print(f"  [{v.mutation_type}] {v.expression}")
    print(f"REJECTED: {len(rejected)}")
    for v in rejected:
        print(f"  [{v.mutation_type}] {v.reject_reason}: {v.expression}")
    print("self-test OK")
