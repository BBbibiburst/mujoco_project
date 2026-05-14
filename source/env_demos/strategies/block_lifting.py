"""
BlockLifting 任务策略实现.

阶段序列: make_gripper_hand_form → approach → align → descend → grasp → lift → hold
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
      0. make_gripper_hand_form : 张成夹爪形准备抓取
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

    _HAND_MAX: float = 0.0095
    _HAND_MIN: float = 0.0
    GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    HAND_OPEN = np.full(6, _HAND_MIN, dtype=np.float64)
    HAND_CLOSE = np.full(6, _HAND_MAX, dtype=np.float64)
    HAND_GRIPPER = np.array([_HAND_MAX, _HAND_MAX, _HAND_MIN, _HAND_MIN, _HAND_MIN, _HAND_MAX])

    @property
    def phases(self) -> list:
        return ["make_gripper_hand_form", "approach", "align", "descend", "grasp", "lift", "hold"]

    def execute_phase(self, phase_idx: int, ctx: PhaseContext) -> Tuple[PhaseResult, ActionContext]:
        phase = self.phases[phase_idx]
        env = ctx.env
        act = ActionContext()

        # 获取方块位置和桌面高度
        obj_pos = env.get_block_position()  # [x, y, z]
        table_z = env._table_height
        ee_pos, ee_quat = env.get_ee_pose()
        hand_qpos = env.get_hand_qpos()

        if phase == "make_gripper_hand_form":
            act.hand_target = self.HAND_GRIPPER.copy()
            act.ee_delta_pos = np.zeros(3)
            act.ee_delta_rot = np.zeros(3)
            max_error = np.max(np.abs(hand_qpos - act.hand_target))
            if ctx.phase_step > self.MIN_PHASE_STEPS and max_error < 0.001:
                # ===== 标定：计算 d_hand（mid-point 在 ee 坐标系下的固定偏移）=====
                if not hasattr(self, '_d_hand'):
                    ee_pos_cal, ee_quat_cal = env.get_ee_pose()
                    mid_cal = env.get_mid_point_position()
                    
                    # 构建 ee 旋转矩阵 R_ee (3x3)
                    R_ee_flat = np.zeros(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(R_ee_flat, ee_quat_cal)
                    R_ee_cal = R_ee_flat.reshape(3, 3)
                    
                    # d_hand = R_ee^T @ (p_mid - p_ee)   [转到 ee 坐标系]
                    self._d_hand = R_ee_cal.T @ (mid_cal - ee_pos_cal)
                    print(f"标定 d_hand: {self._d_hand}")
                
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "approach":
            # ========== 1. 目标 mid-point 位置 ==========
            target_mid = np.array([
                obj_pos[0], 
                obj_pos[1], 
                env._table_height + self.PRE_GRASP_HEIGHT
            ])
            
            # ========== 2. 确定目标手指连线方向（水平） ==========
            # 获取当前手指位置，提取水平方向
            thumb_pos = env.get_site_pos("inspirehand_fingertip_thumb")
            finger3_pos = env.get_site_pos("inspirehand_fingertip_3")
            
            current_axis = thumb_pos - finger3_pos
            current_axis[2] = 0.0  # 强制水平（投影到xy平面）
            
            axis_norm = np.linalg.norm(current_axis)
            if axis_norm > 1e-6:
                target_finger_axis = current_axis / axis_norm
            else:
                target_finger_axis = np.array([1.0, 0.0, 0.0])
            
            # ========== 3. 构建目标 ee 旋转矩阵 ==========
            # 
            # 关键约束：手指连线在 ee 的 x-z 平面内（由模型结构决定）
            # 要让手指连线水平（世界 z=0），需要让 ee 的 x-z 平面与世界 xy 平面平行
            # 即 ee_x 和 ee_z 都在水平面内，ee_y 垂直向上
            #
            # 设目标手指连线方向为 f（水平单位向量）
            # 手指连线在 ee 坐标系下为 [a, 0, b]（a²+b²=1），在世界下为 f
            # 即 f = a * R_ee[:,0] + b * R_ee[:,2]
            #
            # 我们需要 ee_y 垂直向上（世界 +z），这样 ee_x 和 ee_z 就在水平面内
            # 然后让手指连线（ee_x 和 ee_z 的线性组合）对准 f
            
            # 方案：令 ee_y = 世界 +z（向上），则 ee_x, ee_z 在水平面
            ee_y = np.array([0.0, 0.0, 1.0])
            
            # ee_z 在水平面内，且与手指连线方向有关
            # 由 d_hand = [0.277, 0.030, -0.041]，ee_z 分量较小
            # 为简化：让 ee_z 垂直于手指连线（在水平面内）
            # 即 ee_z = normalize([f_y, -f_x, 0])
            ee_z = np.array([target_finger_axis[1], -target_finger_axis[0], 0.0])
            z_norm = np.linalg.norm(ee_z)
            if z_norm < 1e-6:
                ee_z = np.array([0.0, -1.0, 0.0])
            else:
                ee_z = ee_z / z_norm
            
            # ee_x = ee_y × ee_z（确保右手系）
            ee_x = np.cross(ee_y, ee_z)
            ee_x = ee_x / np.linalg.norm(ee_x)
            
            # 验证：重新计算 ee_z = ee_x × ee_y
            ee_z = np.cross(ee_x, ee_y)
            ee_z = ee_z / np.linalg.norm(ee_z)
            
            # 构建旋转矩阵（列向量）
            R_target = np.column_stack([ee_x, ee_y, ee_z])
            
            # 转成四元数
            target_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(target_quat, R_target.flatten('F'))
            
            # ========== 4. 反推目标 ee 位置 ==========
            target_ee_pos = target_mid - R_target @ self._d_hand
            
            # ========== 5. 填充绝对目标（供可视化使用）==========
            act.ee_target_pos = target_ee_pos.copy()
            act.ee_target_quat = target_quat.copy()
            
            # ========== 6. 执行控制 ==========
            pos_delta = target_ee_pos - ee_pos
            dist = np.linalg.norm(pos_delta)
            if dist > self.APPROACH_SPEED:
                pos_delta = pos_delta / dist * self.APPROACH_SPEED
            act.ee_delta_pos = pos_delta
            
            # 姿态控制
            rel_quat = np.zeros(4, dtype=np.float64)
            inv_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_negQuat(inv_quat, ee_quat)
            mujoco.mju_mulQuat(rel_quat, target_quat, inv_quat)
            
            axis_angle = np.zeros(3, dtype=np.float64)
            mujoco.mju_quat2Vel(axis_angle, rel_quat, 1.0)
            
            angle = np.linalg.norm(axis_angle)
            max_rot = 0.1
            if angle > max_rot:
                axis_angle = axis_angle / angle * max_rot
            act.ee_delta_rot = axis_angle
            
            act.hand_target = self.HAND_GRIPPER.copy()
            
            # ========== 7. 收敛判断 ==========
            pos_err = distance(ee_pos, target_ee_pos)
            quat_err = quat_distance(ee_quat, target_quat)
            
            if ctx.phase_step > self.MIN_PHASE_STEPS and pos_err < 0.02 and quat_err < 0.15:
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
            act.hand_target = self.HAND_GRIPPER.copy()
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