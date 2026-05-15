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

# TaskStrategy (ABC)                    # 抽象策略基类
# │
# ├── 状态属性
# │   ├── phase_idx: int                # 当前阶段索引
# │   ├── phase_step: int               # 当前阶段内步数
# │   ├── memory: Dict                  # 跨阶段记忆存储
# │   ├── finished: bool                # 是否结束
# │   └── success: bool                 # 是否成功
# │
# ├── 抽象接口（子类必须实现）
# │   ├── @property phases() -> List[str]
# │   └── execute_phase(idx, ctx) -> (PhaseResult, ActionContext)
# │
# ├── 主循环引擎（基类已实现）
# │   └── tick(obs, info, step, env)
# │       ├── 构造 PhaseContext
# │       ├── 调用 execute_phase()
# │       ├── 处理阶段结果 → CONTINUE / NEXT / RETRY / ABORT
# │       └── 调用 _build_action() → 归一化动作 [-1, 1]
# │
# └── 生命周期
#     ├── reset()                       # 重置所有状态
#     └── get_status_dict()             # 导出状态字典