"""
任务注册表.

集中管理所有任务的元信息，新增任务只需在此添加一条记录。
"""

from typing import Any, Dict, Type, Optional
import importlib

from ..env.base_env import RobotArmEnvBase
from ..env.env_config import RobotConfig


# ====================== 注册表 ======================

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "block_lifting": {
        "module":        "source.env.block_lifting_env",
        "env_class":     "BlockLiftingEnv",
        "cfg_class":     "BlockLiftingConfig",
        "display_name":  "Block Lifting",
        "strategy_class": "BlockLiftingStrategy",
        "strategy_module": "source.env_demos.strategies.block_lifting",
    },
}


# ====================== 环境加载 ======================

def load_task(task_name: str, robot_cfg: RobotConfig) -> RobotArmEnvBase:
    """
    按名称动态加载并实例化任务环境.

    Args:
        task_name: TASK_REGISTRY 中的键名。
        robot_cfg: 机器人配置。

    Returns:
        初始化完成的环境实例（未调用 reset）。
    """
    if task_name not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys())
        raise ValueError(f"未知任务: '{task_name}'。可用任务: {available}")

    reg = TASK_REGISTRY[task_name]
    mod      = importlib.import_module(reg["module"])
    EnvClass = getattr(mod, reg["env_class"])
    return EnvClass(robot_config=robot_cfg)


# ====================== 策略加载 ======================

def load_strategy(task_name: str):
    """
    按名称动态加载并实例化任务策略.

    Args:
        task_name: TASK_REGISTRY 中的键名。

    Returns:
        策略实例（未调用 reset）。
    """
    if task_name not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys())
        raise ValueError(f"未知任务: '{task_name}'。可用策略: {available}")

    reg = TASK_REGISTRY[task_name]
    strategy_class = reg.get("strategy_class")
    if strategy_class is None:
        raise ValueError(f"任务 '{task_name}' 未注册策略。")

    mod = importlib.import_module(reg["strategy_module"])
    StrategyClass = getattr(mod, strategy_class)
    return StrategyClass()


def has_strategy(task_name: str) -> bool:
    """检查任务是否注册了策略."""
    return task_name in TASK_REGISTRY and TASK_REGISTRY[task_name].get("strategy_class") is not None


# ====================== 便捷函数 ======================

def get_display_name(task_name: str) -> str:
    """获取任务显示名称."""
    return TASK_REGISTRY[task_name]["display_name"]


def list_tasks() -> list:
    """返回所有注册的任务名称."""
    return list(TASK_REGISTRY.keys())


def list_strategies() -> list:
    """返回所有注册了策略的任务名称."""
    return [k for k, v in TASK_REGISTRY.items() if v.get("strategy_class") is not None]