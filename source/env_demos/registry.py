"""
任务注册表.

集中管理所有任务的元信息，新增任务只需在此添加一条记录。
"""

from typing import Any, Dict, Type
import importlib

from ..env.base_env import RobotArmEnvBase
from ..env.env_config import RobotConfig


# ====================== 注册表 ======================

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "block_lifting": {
        "module":       "source.env.block_lifting_env",
        "env_class":    "BlockLiftingEnv",
        "cfg_class":    "BlockLiftingConfig",
        "display_name": "Block Lifting",
        "default_cfg_kwargs": {},
    },
}


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
    mod       = importlib.import_module(reg["module"])
    EnvClass  = getattr(mod, reg["env_class"])
    CfgClass  = getattr(mod, reg["cfg_class"])
    task_cfg  = CfgClass(**reg["default_cfg_kwargs"])
    return EnvClass(robot_config=robot_cfg, task_config=task_cfg)