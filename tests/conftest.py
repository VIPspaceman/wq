# -*- coding: utf-8 -*-

"""
pytest 全局配置。

职责
----
    - 把项目根目录加入 sys.path，使 `import alpha.*` 等可用
    - 提供共享 fixture
"""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def project_root() -> str:
    return PROJECT_ROOT
