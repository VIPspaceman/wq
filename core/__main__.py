# -*- coding: utf-8 -*-

"""
core 模块自检（L1 验证）。

运行：
    python -m core

不依赖外部 BRAIN / 网络 / 数据库。
只验证 core/ 内部逻辑自洽。
"""

from __future__ import annotations

import sys


def _check(cond: bool, msg: str) -> bool:
    if cond:
        print(f"  [OK]   {msg}")
        return True
    print(f"  [FAIL] {msg}")
    return False


def main() -> int:
    print("=" * 60)
    print("core 模块自检")
    print("=" * 60)

    ok = True

    # -------- 1. import --------
    print("\n[1] import")
    try:
        from core import (
            AlphaCandidate,
            expression_hash,
            structure_signature,
            get_field_registry,
            get_operator_registry,
        )
        ok &= _check(True, "core 导出全部可用")
    except Exception as e:
        print(f"  [FAIL] import 失败: {e!r}")
        return 1

    # -------- 2. expression_hash --------
    print("\n[2] expression_hash")
    h1 = expression_hash("rank(ts_delta(close,5))")
    h2 = expression_hash("rank(ts_delta(close, 5))")   # 带空格
    h3 = expression_hash("RANK(TS_DELTA(CLOSE,5))")    # 大写
    ok &= _check(h1 == h2, "空格不影响 hash")
    ok &= _check(h1 == h3, "大小写不影响 hash")
    ok &= _check(len(h1) == 32, "hash 为 32 位十六进制")

    # -------- 3. structure_signature --------
    print("\n[3] structure_signature")
    from alpha.parser import parse_expression

    n1 = parse_expression("rank(ts_delta(close,30)/volume)")
    n2 = parse_expression("rank(ts_delta(close,20)/volume)")
    n3 = parse_expression("rank(ts_mean(close,30)/volume)")

    sig1 = structure_signature(n1)
    sig2 = structure_signature(n2)
    sig3 = structure_signature(n3)

    ok &= _check(sig1 != "", "签名非空")
    ok &= _check(sig1 == sig2, "同结构不同窗口 → 同签名")
    ok &= _check(sig1 != sig3, "不同算子 → 不同签名")
    ok &= _check(structure_signature(None) == "", "None → 空签名")

    # -------- 4. AlphaCandidate.from_expression --------
    print("\n[4] AlphaCandidate.from_expression")
    cand = AlphaCandidate.from_expression(
        "rank(ts_delta(close,5))",
        family="MOMENTUM",
        generation=3,
        source="generator",
    )
    ok &= _check(cand.expression == "rank(ts_delta(close,5))", "expression")
    ok &= _check(cand.expr_hash == expression_hash(cand.expression), "expr_hash")
    ok &= _check(cand.candidate_id == cand.expr_hash, "candidate_id 默认=expr_hash")
    ok &= _check(cand.structure_sig != "", "structure_sig 已填")
    ok &= _check(cand.ast is not None, "ast 已解析")
    ok &= _check(cand.family == "MOMENTUM", "family")
    ok &= _check(cand.generation == 3, "generation")
    ok &= _check(cand.stage == "generated", "stage 默认")
    ok &= _check(cand.status == "PENDING", "status 默认")

    # -------- 5. 序列化 round-trip --------
    print("\n[5] to_dict / from_dict round-trip")
    cand.add_tag("AI_FIXED")
    cand.sharpe = 1.42
    cand.fitness = 0.88
    cand.turnover = 0.55
    cand.score = 0.91
    cand.status = "CANDIDATE"
    cand.stage = "classified"

    d = cand.to_dict()
    cand2 = AlphaCandidate.from_dict(d)

    ok &= _check(cand2.expression == cand.expression, "expression 一致")
    ok &= _check(cand2.expr_hash == cand.expr_hash, "expr_hash 一致")
    ok &= _check(cand2.structure_sig == cand.structure_sig, "structure_sig 一致")
    ok &= _check(cand2.family == cand.family, "family 一致")
    ok &= _check(cand2.generation == cand.generation, "generation 一致")
    ok &= _check(cand2.status == cand.status, "status 一致")
    ok &= _check(cand2.stage == cand.stage, "stage 一致")
    ok &= _check(cand2.sharpe == cand.sharpe, "sharpe 一致")
    ok &= _check(cand2.tags == cand.tags, "tags 一致")
    ok &= _check(cand2.ast is None, "ast 延迟解析（反序列化后为 None）")

    # -------- 6. 延迟解析 --------
    print("\n[6] ensure_ast 延迟解析")
    node = cand2.ensure_ast()
    ok &= _check(node is not None, "ensure_ast 成功")
    ok &= _check(cand2.structure_sig != "", "ensure_ast 补填 structure_sig")

    # -------- 7. registry --------
    print("\n[7] registry")
    fr = get_field_registry()
    orr = get_operator_registry()

    fields = fr.names()
    ok &= _check(len(fields) > 0, f"字段池非空 ({len(fields)} 个)")
    ok &= _check("close" in fields, "close 在字段池")
    ok &= _check(fr.contains("close"), "contains('close')")
    ok &= _check(not fr.contains("__nope__"), "contains('__nope__') == False")

    ops = orr.unary()
    ok &= _check(len(ops) > 0, f"一元算子非空 ({len(ops)} 个)")
    ok &= _check("rank" in ops, "rank 在一元算子池")
    ok &= _check(orr.contains("rank"), "contains('rank')")
    ok &= _check(not orr.contains("__nope__"), "contains('__nope__') == False")

    # -------- 8. 兼容层 --------
    print("\n[8] AlphaRecord <-> AlphaCandidate")
    try:
        rec = cand.to_record()
        ok &= _check(rec.expression == cand.expression, "to_record expression")
        ok &= _check(rec.sharpe == cand.sharpe, "to_record sharpe")
        ok &= _check(rec.classification.value == cand.status, "to_record classification")

        cand3 = AlphaCandidate.from_record(rec)
        ok &= _check(cand3.expression == cand.expression, "from_record expression")
        ok &= _check(cand3.status == cand.status, "from_record status")
        ok &= _check(cand3.sharpe == cand.sharpe, "from_record sharpe")
    except Exception as e:
        ok &= _check(False, f"兼容层失败: {e!r}")

    # -------- 结果 --------
    print()
    print("=" * 60)
    if ok:
        print("AlphaCandidate OK")
        print("=" * 60)
        return 0
    print("AlphaCandidate FAILED")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
