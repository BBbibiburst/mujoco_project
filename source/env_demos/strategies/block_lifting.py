"""
BlockLifting 任务策略实现.

阶段序列: open_hand → approach → align → descend → grasp → lift → hold
"""

from typing import Tuple

import mujoco
import numpy as np

from .base import TaskStrategy, PhaseResult, PhaseContext, ActionContext


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """四元数差异（弧度），输入 [w,x,y,z]."""
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2.0 * np.arccos(abs(dot))


class BlockLiftingStrategy(TaskStrategy):
    """
    BlockLifting 流程化策略.

    阶段:
      0. open_hand : 张开手
      1. approach  : 移动到方块上方
      2. align     : 调整末端姿态（垂直向下）
      3. descend   : 下降接触方块
      4. grasp     : 闭合手指抓取
      5. lift      : 垂直提升
      6. hold      : 保持验证
    """

    PRE_GRASP_HEIGHT: float = 0.15
    GRASP_HEIGHT: float = 0.03
    LIFT_HEIGHT: float = 0.20
    APPROACH_SPEED: float = 0.03
    DESCEND_SPEED: float = 0.015
    LIFT_SPEED: float = 0.02
    MIN_PHASE_STEPS: int = 5

    GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    HAND_OPEN = np.zeros(6, dtype=np.float64)
    HAND_CLOSE = np.full(6, 0.0095, dtype=np.float64)

    @property
    def phases(self) -> list:
        return ["open_hand", "approach", "align", "descend", "grasp", "lift", "hold"]

    def execute_phase(self, phase_idx: int, ctx: PhaseContext) -> Tuple[PhaseResult, ActionContext]:
        phase = self.phases[phase_idx]
        env = ctx.env
        act = ActionContext()

        if "obj_pos" not in ctx.memory:
            try:
                obj_pos = env.get_body_pos("target_object")
                ctx.memory["obj_pos"] = obj_pos.copy()
                ctx.memory["table_height"] = getattr(env, '_table_height', 0.55)
            except ValueError:
                ctx.memory["obj_pos"] = np.array([0.45, 0.0, 0.58])
                ctx.memory["table_height"] = 0.55

        obj_pos = ctx.memory["obj_pos"]
        table_z = ctx.memory["table_height"]
        ee_pos, ee_quat = env.get_ee_pose()
        hand_qpos = env.get_hand_qpos()

        if phase == "open_hand":
            act.hand_target = self.HAND_OPEN.copy()
            act.ee_delta_pos = np.zeros(3)
            if ctx.phase_step > self.MIN_PHASE_STEPS and np.all(hand_qpos < 0.001):
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "approach":
            target_pos = np.array([obj_pos[0], obj_pos[1], table_z + self.PRE_GRASP_HEIGHT])
            delta = target_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist > self.APPROACH_SPEED:
                delta = delta / dist * self.APPROACH_SPEED
            act.ee_delta_pos = delta
            act.hand_target = self.HAND_OPEN.copy()
            if ctx.phase_step > self.MIN_PHASE_STEPS and distance(ee_pos, target_pos) < 0.02:
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "align":
            target_pos = np.array([obj_pos[0], obj_pos[1], table_z + self.PRE_GRASP_HEIGHT])
            pos_delta = target_pos - ee_pos
            if np.linalg.norm(pos_delta) > 0.01:
                pos_delta = pos_delta / np.linalg.norm(pos_delta) * 0.01
            act.ee_delta_pos = pos_delta

            rel_quat = np.zeros(4, dtype=np.float64)
            inv_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_negQuat(inv_quat, ee_quat)
            mujoco.mju_mulQuat(rel_quat, self.GRASP_QUAT, inv_quat)

            axis_angle = np.zeros(3, dtype=np.float64)
            mujoco.mju_quat2Vel(axis_angle, rel_quat, 1.0)

            angle = np.linalg.norm(axis_angle)
            max_rot = 0.1
            if angle > max_rot:
                axis_angle = axis_angle / angle * max_rot
            act.ee_delta_rot = axis_angle

            quat_err = quat_distance(ee_quat, self.GRASP_QUAT)
            pos_err = distance(ee_pos, target_pos)
            if ctx.phase_step > self.MIN_PHASE_STEPS and quat_err < 0.15 and pos_err < 0.025:
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "descend":
            target_pos = np.array([obj_pos[0], obj_pos[1], table_z + self.GRASP_HEIGHT])
            delta = target_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist > self.DESCEND_SPEED:
                delta = delta / dist * self.DESCEND_SPEED
            act.ee_delta_pos = delta
            act.ee_delta_rot = np.zeros(3)
            act.hand_target = self.HAND_OPEN.copy()
            if ctx.phase_step > self.MIN_PHASE_STEPS and distance(ee_pos, target_pos) < 0.02:
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "grasp":
            target_pos = np.array([obj_pos[0], obj_pos[1], table_z + self.GRASP_HEIGHT])
            pos_delta = target_pos - ee_pos
            if np.linalg.norm(pos_delta) > 0.005:
                pos_delta = pos_delta / np.linalg.norm(pos_delta) * 0.005
            act.ee_delta_pos = pos_delta
            act.hand_target = self.HAND_CLOSE.copy()

            hand_closed = np.all(hand_qpos > 0.008)
            if ctx.phase_step > 20 and hand_closed:
                return PhaseResult.NEXT, act
            if ctx.phase_step > 50:
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "lift":
            target_pos = np.array([obj_pos[0], obj_pos[1], table_z + self.LIFT_HEIGHT])
            delta = target_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist > self.LIFT_SPEED:
                delta = delta / dist * self.LIFT_SPEED
            act.ee_delta_pos = delta
            act.ee_delta_rot = np.zeros(3)
            act.hand_target = self.HAND_CLOSE.copy()
            if ctx.phase_step > self.MIN_PHASE_STEPS and ee_pos[2] >= target_pos[2] - 0.02:
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "hold":
            target_pos = np.array([obj_pos[0], obj_pos[1], table_z + self.LIFT_HEIGHT])
            pos_delta = target_pos - ee_pos
            if np.linalg.norm(pos_delta) > 0.005:
                pos_delta = pos_delta / np.linalg.norm(pos_delta) * 0.005
            act.ee_delta_pos = pos_delta
            act.hand_target = self.HAND_CLOSE.copy()
            if ctx.phase_step > 30:
                self.success = True
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        return PhaseResult.ABORT, act