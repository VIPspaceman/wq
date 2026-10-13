# -*- coding: utf-8 -*-

"""
search/coordinator.py 单元测试。

覆盖：
    - MiningConfig 默认值
    - MiningState 默认值
    - _maybe_persist 的异常处理（P2 修复验证）
"""

from __future__ import annotations

import pytest

from search.coordinator import Coordinator, MiningConfig, MiningState


# ============================================================
# MiningConfig
# ============================================================

def test_config_defaults() -> None:
    cfg = MiningConfig()
    assert cfg.population_size == 30
    assert cfg.generations == 10
    assert cfg.elite_size == 8
    assert cfg.enable_sign_flip is True
    assert cfg.enable_combine is True


# ============================================================
# MiningState
# ============================================================

def test_state_defaults() -> None:
    s = MiningState()
    assert s.generation == -1
    assert s.evaluated_hashes == set()
    assert s.population == []
    assert s.elites == []
    assert s.final_concurrency is None


# ============================================================
# _maybe_persist 异常处理（P2 修复验证）
# ============================================================

class _FakeExecutor:
    class _Ctrl:
        def current(self):
            raise RuntimeError("boom")

    @property
    def concurrency_controller(self):
        return self._Ctrl()


def test_maybe_persist_swallows_controller_error() -> None:
    """
    concurrency_controller.current() 抛异常时，
    _maybe_persist 不应向上传播。
    """
    coord = Coordinator(
        executor=_FakeExecutor(),
        config=MiningConfig(),
    )
    state = MiningState()
    # 不应抛异常
    coord._maybe_persist(state, on_generation=None)


def test_maybe_persist_swallows_callback_error() -> None:
    """
    on_generation 回调抛异常时，不应向上传播。
    """
    class _OkExecutor:
        class _Ctrl:
            def current(self):
                return 2

        @property
        def concurrency_controller(self):
            return self._Ctrl()

    coord = Coordinator(
        executor=_OkExecutor(),
        config=MiningConfig(),
    )
    state = MiningState()

    def bad_callback(s):
        raise RuntimeError("callback boom")

    # 不应抛异常
    coord._maybe_persist(state, on_generation=bad_callback)
    assert state.final_concurrency == 2
