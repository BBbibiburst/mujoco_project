"""
策略模块入口.

导出:
    TaskStrategy, PhaseResult, PhaseContext, ActionContext  (基类与数据结构)
    BlockLiftingStrategy                                    (具体实现)
    create_strategy, STRATEGY_REGISTRY                      (注册表)
"""

from .base import TaskStrategy, PhaseResult, PhaseContext, ActionContext
from .block_lifting import BlockLiftingStrategy
from ..registry import load_strategy as create_strategy, has_strategy