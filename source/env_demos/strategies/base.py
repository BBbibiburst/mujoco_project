"""
策略基类与核心数据结构.

定义 TaskStrategy 抽象基类，以及 PhaseResult / PhaseContext / ActionContext。

设计原则：策略只负责生成动作序列，成功 / 失败完全由环境的
terminated / truncated / success 信号判断，策略内部不维护重复的
finished / success 状态。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PhaseResult(Enum):
    """阶段执行结果.

    CONTINUE : 当前阶段继续执行。
    NEXT     : 当前阶段完成，推进到下一阶段。
    RETRY    : 重置当前阶段的步数计数器并重试（阶段不变）。
    RESTART  : 阶段完成，但策略需要回到第一个阶段重新开始（例如判定失败后重试）。
    ABORT    : 策略放弃执行，后续步骤输出零动作。

    """
    CONTINUE = auto()
    NEXT = auto()
    RETRY = auto()
    RESTART = auto()
    ABORT = auto()


@dataclass
class PhaseContext:
    """阶段执行上下文."""
    obs: Dict[str, Any]
    info: Dict[str, Any]
    step: int
    phase_step: int
    memory: Dict[str, Any]
    env: Any


@dataclass
class ActionContext:
    """策略输出的原始控制目标（未归一化）."""
    ee_delta_pos: Optional[np.ndarray] = None
    ee_delta_rot: Optional[np.ndarray] = None
    hand_target: Optional[np.ndarray] = None
    gripper_cmd: Optional[str] = None
    # 绝对目标位姿，供可视化使用
    ee_target_pos: Optional[np.ndarray] = None   # 目标 ee 位置（世界坐标）
    ee_target_quat: Optional[np.ndarray] = None  # 目标 ee 姿态四元数 [w,x,y,z]


class TaskStrategy(ABC):
    """任务策略抽象基类.

    策略只负责生成动作；回合的成功 / 失败 / 终止完全依赖环境信号

    属性:
        phase_idx  : 当前阶段索引。
        phase_step : 当前阶段内的步数。
        aborted    : 策略主动放弃执行时置 True（ABORT 结果）。
        memory     : 跨阶段共享的键值存储，由子类自由读写。
    """

    def __init__(self):
        self.phase_idx: int = 0
        self.phase_step: int = 0
        self.aborted: bool = False
        self.memory: Dict[str, Any] = {}

    @property
    @abstractmethod
    def phases(self) -> List[str]:
        """阶段名称列表."""
        pass

    @abstractmethod
    def execute_phase(self, phase_idx: int, ctx: PhaseContext) -> Tuple[PhaseResult, ActionContext]:
        """执行当前阶段，返回结果与动作上下文."""
        pass

    def reset(self) -> None:
        """重置策略状态."""
        self.phase_idx = 0
        self.phase_step = 0
        self.aborted = False
        self.memory.clear()

    def tick(self, obs, info, step, env) -> Tuple[np.ndarray, ActionContext]:
        """
        策略主循环调用，每个环境步骤调用一次。

        Returns:
            (action, action_context)

            action 始终是合法的归一化动作向量。策略放弃（ABORT）或
            所有阶段完成后，返回零动作——调用方应继续让环境自然结束
            （等待 terminated / truncated），而非依赖策略状态判断终止。
        """
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        if self.aborted or self.phase_idx >= len(self.phases):
            return zero_action, ActionContext()

        ctx = PhaseContext(
            obs=obs, info=info, step=step,
            phase_step=self.phase_step,
            memory=self.memory,
            env=env,
        )

        result, act_ctx = self.execute_phase(self.phase_idx, ctx)
        self.phase_step += 1
        action = self._build_action(act_ctx, env)

        if result == PhaseResult.NEXT:
            self.phase_idx += 1
            self.phase_step = 0
        elif result == PhaseResult.RETRY:
            self.phase_step = 0
        elif result == PhaseResult.RESTART:
            # 回到第一个阶段，步数清零，触发子类的重计算逻辑
            self.phase_idx = 0
            self.phase_step = 0
        elif result == PhaseResult.ABORT:
            self.aborted = True

        return action, act_ctx

    @property
    def all_phases_done(self) -> bool:
        """所有阶段已依次完成（不含 ABORT）."""
        return self.phase_idx >= len(self.phases)

    def _build_action(self, act_ctx: ActionContext, env) -> np.ndarray:
        """
        将 ActionContext 转换为 env.action_space 格式的归一化动作 [-1, 1].
        """
        cfg = env.cfg

        if cfg.action_mode == "ee":
            if act_ctx.ee_delta_pos is not None:
                delta_pos = act_ctx.ee_delta_pos / cfg.action_scale
            else:
                delta_pos = np.zeros(3)

            if act_ctx.ee_delta_rot is not None:
                angle = np.linalg.norm(act_ctx.ee_delta_rot)
                if angle > 1e-6:
                    max_rot_step = cfg.action_scale_rot or cfg.action_scale
                    if angle > max_rot_step:
                        act_ctx.ee_delta_rot = act_ctx.ee_delta_rot / angle * max_rot_step
                        angle = max_rot_step
                delta_rot = act_ctx.ee_delta_rot / (cfg.action_scale_rot or cfg.action_scale)
            else:
                delta_rot = np.zeros(3)

            if act_ctx.hand_target is not None:
                current_hand = env.get_hand_qpos()
                hand_delta = act_ctx.hand_target - current_hand
                scale_hand = cfg.action_scale_hand or cfg.action_scale
                hand_action = hand_delta / scale_hand
            else:
                hand_action = np.zeros(env.HAND_DOF)

            action = np.concatenate([delta_pos, delta_rot, hand_action]).astype(np.float32)

        elif cfg.action_mode == "joint":
            if act_ctx.hand_target is not None:
                current = np.concatenate([env.get_arm_qpos(), env.get_hand_qpos()])
                target = current.copy()
                target[env.ARM_DOF:] = act_ctx.hand_target
                delta = target - current
                scale_hand = cfg.action_scale_hand or cfg.action_scale
                delta[env.ARM_DOF:] /= scale_hand
                delta[:env.ARM_DOF] /= cfg.action_scale
                action = delta.astype(np.float32)
            else:
                action = np.zeros(env.TOTAL_DOF, dtype=np.float32)
        else:
            raise ValueError(f"Unknown action_mode: {cfg.action_mode}")

        return np.clip(action, -1.0, 1.0)

    def get_status_dict(self) -> dict:
        """
        返回通用状态字典，供外部显示使用.

        子类可重写以添加任务特定信息。
        注：成功 / 失败状态由环境信号决定，此处不再重复报告。
        """
        return {
            "phase_idx": self.phase_idx,
            "phase_name": self.phases[self.phase_idx] if self.phase_idx < len(self.phases) else "done",
            "phase_step": self.phase_step,
            "total_phases": len(self.phases),
            "all_phases_done": self.all_phases_done,
            "aborted": self.aborted,
            "memory_keys": list(self.memory.keys()),
        }