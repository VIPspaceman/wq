# evolution/population.py
# -*- coding: utf-8 -*-

"""
种群管理。

职责边界
--------
    - create_initial_population(size)
    - create_next_generation(elites, size, ...)
    - record_to_node(record)  内部 helper

对齐 V3 对应函数；参数显式传入。
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Dict, List, Optional, Set

from alpha.ast import Node, clone_node, node_to_expr
from alpha.families import detect_family
from alpha.generator import generate_initial_alpha
from alpha.parser import parse_expression
from alpha.validator import validate_node
from core.hashing import expression_hash
from evaluation.record import AlphaRecord
from evolution.crossover import crossover
from evolution.mutation import mutate
from evolution.reverse import reverse_alpha


logger = logging.getLogger(__name__)


# ============================================================
# record -> node
# ============================================================

def record_to_node(record: AlphaRecord) -> Optional[Node]:
    return parse_expression(record.expression)


# ============================================================
# 初始种群
# ============================================================

def create_initial_population(
    size: int,
    evaluated_hashes: Optional[Set[str]] = None,
    max_attempts_factor: int = 500,
    max_same_family_ratio: float = 0.30,
) -> List[Node]:
    """
    与 V3 行为一致：
        - 去重（基于 expression hash）
        - 跳过 evaluated_hashes 里已有的表达式
        - 上限尝试 size * max_attempts_factor

    FAMILY-QUOTA（新增）
    --------------------
    为避免初始种群被单一 family 垄断（例如历史 hash 去重对
    价量模板命中率高、对基本面模板命中率低，导致剩下的都是
    基本面），对每个 family 施加配额：

        quota = max(1, int(size * max_same_family_ratio))

    超过配额的节点被跳过（不加入 population），继续重试。
    重试 max_attempts 次后仍不足 size 时，返回已生成的
    （不死循环）。

    Parameters
    ----------
    max_same_family_ratio : float
        单个 family 最多占的比例（默认 0.30）。
    """
    population: List[Node] = []
    seen: Set[str] = set()
    attempts = 0
    max_attempts = size * max_attempts_factor

    if evaluated_hashes is None:
        evaluated_hashes = set()

    # FAMILY-QUOTA：单 family 配额（至少 1，避免 size 小时配额为 0）
    family_quota = max(1, int(size * max_same_family_ratio))
    family_counts: Counter = Counter()

    # ========================================================
    # DIAG：诊断日志（临时，定位 0/20 问题）
    # ========================================================
    from alpha.operators import default_catalog as _diag_catalog
    _diag_cat = _diag_catalog()
    logger.debug(
        "[DIAG create_initial_population] "
        "size=%d max_attempts=%d family_quota=%d "
        "evaluated_hashes=%d",
        size, max_attempts, family_quota, len(evaluated_hashes),
    )
    logger.debug(
        "[DIAG create_initial_population] catalog.unary=%s",
        _diag_cat.unary(),
    )
    logger.debug(
        "[DIAG create_initial_population] catalog.ts=%s",
        _diag_cat.ts(),
    )
    logger.debug(
        "[DIAG create_initial_population] catalog.ts_binary=%s",
        _diag_cat.ts_binary(),
    )
    logger.debug(
        "[DIAG create_initial_population] catalog.windows=%s",
        _diag_cat.windows(),
    )
    logger.debug(
        "[DIAG create_initial_population] catalog.summary=%s",
        _diag_cat.summary(),
    )

    # 拒绝原因计数
    _diag_reasons: Counter = Counter()
    _diag_none_examples: List[str] = []
    _diag_dup_seen_examples: List[str] = []
    _diag_dup_eval_examples: List[str] = []
    _diag_quota_examples: List[str] = []
    _diag_ok_examples: List[str] = []

    while len(population) < size and attempts < max_attempts:
        attempts += 1

        node = generate_initial_alpha()
        if node is None:
            _diag_reasons["generate_none"] += 1
            if attempts <= 10:
                logger.debug(
                    "[DIAG attempt %d] "
                    "generate_initial_alpha -> None",
                    attempts,
                )
            continue

        expr = node_to_expr(node)
        h = expression_hash(expr)

        # DIAG：前 10 次尝试打印详情
        if attempts <= 10:
            logger.debug(
                "[DIAG attempt %d] node=%r hash=%s family=%r",
                attempts, expr, h[:12], node.family,
            )

        if h in seen:
            _diag_reasons["dup_seen"] += 1
            if len(_diag_dup_seen_examples) < 5:
                _diag_dup_seen_examples.append(expr)
            if attempts <= 10:
                logger.debug(
                    "[DIAG attempt %d] REJECT: dup_seen",
                    attempts,
                )
            continue
        if h in evaluated_hashes:
            _diag_reasons["dup_evaluated"] += 1
            if len(_diag_dup_eval_examples) < 5:
                _diag_dup_eval_examples.append(expr)
            if attempts <= 10:
                logger.debug(
                    "[DIAG attempt %d] "
                    "REJECT: dup_evaluated (历史 hash)",
                    attempts,
                )
            continue  # ★ 跳过已评估

        # FAMILY-QUOTA：family 配额检查
        family = node.family or detect_family(expr) or "UNKNOWN"
        if family_counts[family] >= family_quota:
            _diag_reasons["family_quota"] += 1
            if len(_diag_quota_examples) < 5:
                _diag_quota_examples.append(
                    f"{family}: {expr}"
                )
            if attempts <= 10:
                logger.debug(
                    "[DIAG attempt %d] REJECT: family_quota "
                    "(family=%s count=%d quota=%d)",
                    attempts, family,
                    family_counts[family], family_quota,
                )
            continue

        seen.add(h)
        population.append(node)
        family_counts[family] += 1
        _diag_reasons["accepted"] += 1
        if len(_diag_ok_examples) < 5:
            _diag_ok_examples.append(expr)
        if attempts <= 10:
            logger.debug(
                "[DIAG attempt %d] ACCEPT: family=%s",
                attempts, family,
            )

    # ========================================================
    # DIAG：汇总
    # ========================================================
    logger.debug(
        "[DIAG create_initial_population] END attempts=%d "
        "accepted=%d/%d reasons=%s",
        attempts, len(population), size, dict(_diag_reasons),
    )
    logger.debug(
        "[DIAG create_initial_population] family_counts=%s",
        dict(family_counts),
    )
    if _diag_none_examples:
        logger.debug(
            "[DIAG create_initial_population] none_examples=%s",
            _diag_none_examples,
        )
    if _diag_dup_seen_examples:
        logger.debug(
            "[DIAG create_initial_population] dup_seen_examples=%s",
            _diag_dup_seen_examples,
        )
    if _diag_dup_eval_examples:
        logger.debug(
            "[DIAG create_initial_population] "
            "dup_evaluated_examples=%s",
            _diag_dup_eval_examples,
        )
    if _diag_quota_examples:
        logger.debug(
            "[DIAG create_initial_population] "
            "family_quota_examples=%s",
            _diag_quota_examples,
        )
    if _diag_ok_examples:
        logger.debug(
            "[DIAG create_initial_population] accepted_examples=%s",
            _diag_ok_examples,
        )

    logger.info(
        "初始种群生成: %d/%d 条（尝试 %d 次）",
        len(population), size, attempts,
    )
    logger.debug(
        "create_initial_population: size=%d/%d family_quota=%d "
        "distribution=%s",
        len(population), size, family_quota,
        dict(family_counts),
    )

    if len(population) < size:
        logger.warning(
            "初始种群未凑够 size: %d/%d（尝试 %d 次后放弃）",
            len(population), size, attempts,
        )

    return population


# ============================================================
# 下一代种群
# ============================================================

def create_next_generation(
    elites: List[AlphaRecord],
    size: int,
    *,
    mutation_rate: float,
    crossover_rate: float,
    reverse_probability: float,
    evaluated_hashes: Optional[Set[str]] = None,
) -> List[Node]:
    """
    与 V3 行为一致：
        1) 保留精英（去重）
        2) 迭代生成子代：mutation + crossover + reverse
        3) 子代去重
        4) 若不足，则用 generate_initial_alpha 补足
    """
    next_population: List[Node] = []
    seen = set()
    elite_nodes: List[Node] = []

    # -------- 保留精英 --------
    for record in elites:
        node = record_to_node(record)
        if node is None:
            continue
        h = expression_hash(node_to_expr(node))
        if h in seen:
            continue
        seen.add(h)
        elite_nodes.append(node)
        next_population.append(node)

    if not elite_nodes:
        # 极端情况：所有 elite 都无法 parse。
        # 至少尝试用 generate_initial_alpha 补足，避免完全丢失。
        logger.warning(
            "create_next_generation: 无可用 elite，回退到初始种群"
        )
        return create_initial_population(size)

    # -------- 迭代生成子代 --------
    attempts = 0
    max_attempts = size * 200

    while len(next_population) < size and attempts < max_attempts:
        attempts += 1

        parent = random.choice(elite_nodes)
        child: Optional[Node] = clone_node(parent)

        # mutation
        if random.random() < mutation_rate:
            mutated = mutate(child)
            if mutated is not None:
                child = mutated

        # crossover
        if (
            len(elite_nodes) >= 2
            and random.random() < crossover_rate
        ):
            other = random.choice(elite_nodes)
            crossed = crossover(child, other)
            if crossed is not None:
                child = crossed

        # reverse
        if random.random() < reverse_probability:
            reversed_node = reverse_alpha(child)
            if reversed_node is not None:
                child = reversed_node

        if child is None:
            continue

        ok, _ = validate_node(child)
        if not ok:
            continue

        expr = node_to_expr(child)
        h = expression_hash(expr)
        if h in seen:
            continue
        if evaluated_hashes and h in evaluated_hashes:
            continue
        seen.add(h)
        next_population.append(child)

    # -------- 补足 --------
    # Step 5a：加最大尝试上限，避免 generate_initial_alpha 持续返回
    # 重复表达式时无限自旋（原实现无上限）。
    fill_attempts = 0
    max_fill_attempts = size * 100

    while len(next_population) < size and fill_attempts < max_fill_attempts:
        fill_attempts += 1

        node = generate_initial_alpha()
        if node is None:
            break

        expr = node_to_expr(node)
        h = expression_hash(expr)
        if h in seen:
            continue
        if evaluated_hashes and h in evaluated_hashes:
            continue
        seen.add(h)
        next_population.append(node)

    if len(next_population) < size:
        logger.warning(
            "create_next_generation: 补足未达 size: %d/%d "
            "（尝试 %d 次后放弃）",
            len(next_population), size, fill_attempts,
        )

    return next_population
