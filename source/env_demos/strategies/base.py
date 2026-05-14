"""
策略基类与核心数据结构.

定义 TaskStrategy 抽象基类，以及 PhaseResult / PhaseContext / ActionContext。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PhaseResult(Enum):
    """阶段执行结果."""
    CONTINUE = auto()
    NEXT = auto()
    RETRY = auto()
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
    """任务策略抽象基类."""

    def __init__(self):
        self.phase_idx: int = 0
        self.phase_step: int = 0
        self.memory: Dict[str, Any] = {}
        self.finished: bool = False
        self.success: bool = False

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
        self.memory.clear()
        self.finished = False
        self.success = False

    def tick(self, obs, info, step, env) -> Tuple[bool, np.ndarray, bool, ActionContext]:
        """
        策略主循环调用.

        Returns:
            (running, action, terminated_by_strategy, action_context)
        """
        if self.finished:
            return False, np.zeros(env.action_space.shape), False, ActionContext()

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
            if self.phase_idx >= len(self.phases):
                self.finished = True
                self.success = True
        elif result == PhaseResult.RETRY:
            self.phase_step = 0
        elif result == PhaseResult.ABORT:
            self.finished = True
            self.success = False

        terminated = self.finished and self.success
        return not self.finished, action, terminated, act_ctx

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
        """
        return {
            "phase_idx": self.phase_idx,
            "phase_name": self.phases[self.phase_idx] if self.phase_idx < len(self.phases) else "done",
            "phase_step": self.phase_step,
            "total_phases": len(self.phases),
            "finished": self.finished,
            "success": self.success,
            "memory_keys": list(self.memory.keys()),
        }