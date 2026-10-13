# execution/concurrency.py
# -*- coding: utf-8 -*-

"""
并发度决策。
（其余注释同前）
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class ConcurrencySettings:
    max_workers: int = 1
    adaptive: bool = False
    min_workers: int = 1
    max_workers_cap: int = 8
    adaptive_success_threshold: int = 5


class ConcurrencyController:
    def __init__(self, settings: ConcurrencySettings) -> None:
        if settings.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._settings = settings

    def effective_workers(self) -> int:
        return max(1, self._settings.max_workers)

    def current(self) -> int:
        return self.effective_workers()

    def on_success(self) -> None:
        return

    def on_concurrency_limit(self) -> None:
        return


class AdaptiveConcurrencyController:
    def __init__(
        self,
        settings: ConcurrencySettings,
        initial_current: int | None = None,
    ) -> None:
        if settings.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if settings.min_workers < 1:
            raise ValueError("min_workers must be >= 1")
        if settings.max_workers_cap < settings.min_workers:
            raise ValueError(
                "max_workers_cap must be >= min_workers"
            )

        self._settings = settings
        base = (
            initial_current
            if initial_current is not None
            else settings.max_workers
        )
        self._current = max(
            settings.min_workers,
            min(base, settings.max_workers_cap),
        )
        self._success_streak = 0
        self._lock = threading.Lock()

    def effective_workers(self) -> int:
        with self._lock:
            return self._current

    def current(self) -> int:
        return self.effective_workers()

    def on_concurrency_limit(self) -> None:
        with self._lock:
            self._current = max(
                self._settings.min_workers,
                self._current - 1,
            )
            self._success_streak = 0

    def on_success(self) -> None:
        with self._lock:
            self._success_streak += 1
            t = self._settings.adaptive_success_threshold
            if t > 0 and self._success_streak >= t:
                self._current = min(
                    self._settings.max_workers_cap,
                    self._current + 1,
                )
                self._success_streak = 0


def build_concurrency_controller(
    settings: ConcurrencySettings,
    initial_current: int | None = None,
):
    if settings.adaptive:
        return AdaptiveConcurrencyController(
            settings, initial_current=initial_current,
        )
    return ConcurrencyController(settings)