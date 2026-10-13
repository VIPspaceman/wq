# brain/__init__.py
# -*- coding: utf-8 -*-

"""
brain：WorldQuant BRAIN 通信层。

子模块职责：
    authentication : 登录 + Session
    client         : HTTP + retry
    polling        : Simulation 轮询
    simulation     : 提交 + 等待
    capabilities   : Capability Discovery（operators / data-fields）
    models         : 数据结构
"""

from brain.authentication import (
    AuthenticationError,
    create_authenticated_session,
)
from brain.capabilities import (
    Capabilities,
    discover,
    fetch_data_fields,
    fetch_operators,
)
from brain.client import BrainClient
from brain.models import (
    SimulationRequest,
    SimulationSettings,
)
from brain.polling import SimulationPoller
from brain.simulation import (
    ConcurrencyLimitExceeded,
    SimulationError,
    SimulationRunner,
)

__all__ = [
    # authentication
    "AuthenticationError",
    "create_authenticated_session",
    # capabilities
    "Capabilities",
    "fetch_operators",
    "fetch_data_fields",
    "discover",
    # client
    "BrainClient",
    # models
    "SimulationRequest",
    "SimulationSettings",
    # polling
    "SimulationPoller",
    # simulation
    "SimulationError",
    "ConcurrencyLimitExceeded",
    "SimulationRunner",
]
