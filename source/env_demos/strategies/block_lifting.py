"""
BlockLifting 任务策略实现.

阶段序列: make_gripper_hand_form → approach → descend → adjust → grasp → lift
"""

import time
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
      1. approach  : 移动到方块上方预抓取位置，保持夹爪形
      2. descend   : 下降接触方块
      3. adjust    : 如果下降后末端位置在水平面上偏离了预定的抓取位置，进行微调回预定位置
      4. grasp     : 闭合手指抓取
      5. lift      : 垂直提升
    """
    
    # PRE_GRASP_HEIGHT 等同于环境中可视化目标高度marker平面的高度
    # 确保在 approach 阶段末端在方块上方且不太远
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
        return ["make_gripper_hand_form", "approach", "descend", "adjust", "grasp", "lift"]

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
                # ===== 标定：计算 _d_hand 和 _finger_in_local =====
                # _d_hand       : mid-point 在 ee 局部坐标系下的固定偏移
                # _finger_in_local : thumb→finger3 连线在 ee 局部坐标系下的单位方向
                if not hasattr(self, '_d_hand'):
                    ee_pos_cal, ee_quat_cal = env.get_ee_pose()
                    mid_cal = env.get_mid_point_position()

                    R_ee_flat = np.zeros(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(R_ee_flat, ee_quat_cal)
                    R_ee_cal = R_ee_flat.reshape(3, 3)

                    # mid-point 在 ee 局部的偏移
                    self._d_hand = R_ee_cal.T @ (mid_cal - ee_pos_cal)

                    # thumb→finger3 在 ee 局部的单位方向
                    thumb_cal   = env.get_site_pos("inspirehand_fingertip_thumb")
                    finger3_cal = env.get_site_pos("inspirehand_fingertip_3")
                    finger_vec_world = finger3_cal - thumb_cal
                    finger_vec_local = R_ee_cal.T @ finger_vec_world
                    self._finger_in_local = finger_vec_local / np.linalg.norm(finger_vec_local)

                    # print(f"标定 _d_hand: {self._d_hand}")
                    # print(f"标定 _finger_in_local: {self._finger_in_local}")

                    # ===== 预计算固定目标旋转矩阵（只算一次）=====
                    # 目标姿态：手指连线（_finger_in_local 方向）在世界中平行于 X 轴，
                    #           同时手心尽量朝下（在手指连线水平的几何约束下能达到的最低）。
                    #
                    # palm 在 ee 局部方向 = [0, 0, -1]（-ee_z_local，由诊断确认）
                    # 构造正交基 {f, p_perp_hat, h}（ee 局部）并映射到目标世界方向：
                    #   f           → [1, 0, 0]   手指连线平行世界 X
                    #   p_perp_hat  → [0, 0, -1]  手心法线在 f 正交补内尽量朝 -Z
                    #   h = f×p_perp_hat → 由右手系唯一确定
                    f_local  = self._finger_in_local
                    p_local  = np.array([0.0, 0.0, -1.0])   # palm 方向在 ee 局部

                    p_perp   = p_local - np.dot(p_local, f_local) * f_local
                    p_perp  /= np.linalg.norm(p_perp)
                    h_local  = np.cross(f_local, p_perp)
                    h_local /= np.linalg.norm(h_local)

                    target_f = np.array([1.0, 0.0, 0.0])
                    target_p = np.array([0.0, 0.0, -1.0])
                    target_h = np.cross(target_f, target_p)   # [0, 1, 0]

                    src_basis = np.column_stack([f_local, p_perp, h_local])
                    dst_basis = np.column_stack([target_f, target_p, target_h])
                    self._approach_R_target = dst_basis @ src_basis.T   # (3×3)

                    # 转为四元数（mujoco [w,x,y,z]）
                    q = np.zeros(4, dtype=np.float64)
                    mujoco.mju_mat2Quat(q, self._approach_R_target.flatten('C'))
                    self._approach_quat_target = q
                    # print(f"目标旋转矩阵:\n{np.round(self._approach_R_target, 4)}")
                    # print(f"目标四元数 (wxyz): {self._approach_quat_target}")

                # ===== 缓存 approach 阶段的固定目标 mid-point 位置 =====
                obj_pos_now = env.get_block_position()
                self._approach_target_mid = np.array([
                    obj_pos_now[0],
                    obj_pos_now[1],
                    env._table_height + self.PRE_GRASP_HEIGHT,
                ])

                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "approach":
            # ========== 目标姿态（固定，阶段0已预计算）==========
            R_target    = self._approach_R_target
            target_quat = self._approach_quat_target

            # ========== 目标 ee 位置：由目标 mid-point 反推 ==========
            # p_mid = p_ee + R_target @ _d_hand  =>  p_ee = p_mid - R_target @ _d_hand
            target_ee_pos = self._approach_target_mid - R_target @ self._d_hand

            # ========== 供可视化使用 ==========
            act.ee_target_pos  = target_ee_pos.copy()
            act.ee_target_quat = target_quat.copy()

            # ========== 位置控制 ==========
            pos_delta = target_ee_pos - ee_pos
            dist = np.linalg.norm(pos_delta)
            if dist > self.APPROACH_SPEED:
                pos_delta = pos_delta / dist * self.APPROACH_SPEED
            act.ee_delta_pos = pos_delta

            # ========== 姿态控制 ==========
            inv_quat = np.zeros(4, dtype=np.float64)
            rel_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_negQuat(inv_quat, ee_quat)
            mujoco.mju_mulQuat(rel_quat, target_quat, inv_quat)

            axis_angle = np.zeros(3, dtype=np.float64)
            mujoco.mju_quat2Vel(axis_angle, rel_quat, 1.0)
            angle = np.linalg.norm(axis_angle)
            if angle > 0.1:
                axis_angle = axis_angle / angle * 0.1
            act.ee_delta_rot = axis_angle

            act.hand_target = self.HAND_GRIPPER.copy()

            # ========== 收敛判断 ==========
            pos_err  = distance(ee_pos, target_ee_pos)
            quat_err = quat_distance(ee_quat, target_quat)
            if ctx.phase_step > self.MIN_PHASE_STEPS and pos_err < 0.02 and quat_err < 0.15:
                
                # 缓存approach阶段末端位置用于 descend 阶段使用
                self._descend_target_pos = ee_pos.copy()
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "descend":
            # approach阶段末端位置在方块上方的marker平面处，高度为 table_z + PRE_GRASP_HEIGHT
            # 下降的距离应该为marker平面高度减去一定比例的方块中心高度
            descend_length = self.PRE_GRASP_HEIGHT + table_z - 0.25 * obj_pos[2]
            x_delta = self._descend_target_pos[0] - ee_pos[0]
            y_delta = self._descend_target_pos[1] - ee_pos[1]
            if abs(x_delta) > 0.01 or abs(y_delta) > 0.01:
                # 如果末端在水平面上偏离了预定的下降位置，优先调整回预定位置再下降
                target_pos = self._descend_target_pos.copy()
                target_pos[2] = ee_pos[2]  # 保持当前高度不变
            else: 
                target_pos = self._descend_target_pos.copy()
                target_pos[2] -= descend_length
            delta = target_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist > self.DESCEND_SPEED:
                delta = delta / dist * self.DESCEND_SPEED
            act.ee_delta_pos = delta
            act.ee_delta_rot = self._rot_correction(ee_quat, self._approach_quat_target)
            act.hand_target = self.HAND_GRIPPER.copy()
            if ctx.phase_step > self.MIN_PHASE_STEPS and ee_pos[2] <= target_pos[2] + 0.01:
                # 存储目前的末端位置用于 grasp 阶段使用
                self._grasp_target_pos = ee_pos.copy()
                # 记录一个开始时间戳用于 grasp 阶段的时间判断（确保有足够时间进行接触和挤压）
                self._grasp_start_time = time.time()
                self._cube_pos_at_descend = obj_pos.copy()  # 记录下降阶段末端位置对应的方块位置，用于后续调整阶段使用
                self._mid_point_at_descend = env.get_mid_point_position()  # 记录下降阶段末端位置对应的mid-point位置，用于后续调整阶段使用
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act
        
        elif phase == "adjust":
        # 初始化：只在进入 adjust 第一步时锁定目标
            if not hasattr(self, '_adjust_step'):
                self._adjust_step = 0
                # 锁定目标：当前方块位置 + 当前指尖中点高度（只调水平面，不调高度）
                cube_now = env.get_block_position()
                # 关键修复：目标是指尖中点的xy，而不是ee_pos的xy
                self._adjust_target_mid = np.array([
                    cube_now[0],
                    cube_now[1],
                    self._mid_point_at_descend[2]  # 保持下降后的指尖高度
                ])
            
            self._adjust_step += 1
            
            # ========== 关键修复：由目标 mid-point 反推目标 ee 位置 ==========
            # 当前姿态下，mid-point = ee_pos + R_ee @ _d_hand
            # 所以 ee_pos = target_mid - R_ee @ _d_hand
            R_ee_flat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(R_ee_flat, ee_quat)
            R_ee = R_ee_flat.reshape(3, 3)
            
            target_ee_pos = self._adjust_target_mid - R_ee @ self._d_hand
            
            # 向目标 ee 位置移动
            delta = target_ee_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist > self.APPROACH_SPEED:
                delta = delta / dist * self.APPROACH_SPEED
            act.ee_delta_pos = delta
            
            act.ee_delta_rot = self._rot_correction(ee_quat, self._approach_quat_target)
            act.hand_target = self.HAND_GRIPPER.copy()
            
            # 收敛判断：检查指尖中点是否对准（而不是 ee_pos）
            current_mid = env.get_mid_point_position()
            mid_err = np.linalg.norm(current_mid[:2] - self._adjust_target_mid[:2])
            
            # 固定50步或中点对准后进入 grasp
            if self._adjust_step >= 50 or mid_err < 0.005:
                delattr(self, '_adjust_step')
                delattr(self, '_adjust_target_mid')
                self._grasp_target_pos = ee_pos.copy()
                return PhaseResult.NEXT, act
            
            return PhaseResult.CONTINUE, act

        elif phase == "grasp":
            ee_delta_pos = self._grasp_target_pos - ee_pos
            ee_delta_rot = self._rot_correction(ee_quat, self._approach_quat_target)
            # grasp 阶段保持末端位置不变，逐渐闭合手指到一个合适的程度（不完全闭合以避免过大的抓取力导致方块被挤出）
            act.ee_delta_pos = ee_delta_pos
            act.ee_delta_rot = ee_delta_rot
            # close_value 是一个介于 0 和 1 之间的值，表示从完全张开到完全闭合的程度。
            # 需要在成功抓取和避免过大挤压力之间权衡
            # 0.4 是根据经验调整的一个较合适的抓取程度，可以根据实际情况微调
            finger_close_value = 0.6
            thumb_close_value = 0.2
            finger_close_value = finger_close_value*self._HAND_MAX+(1-finger_close_value)*self._HAND_MIN
            thumb_close_value = thumb_close_value*self._HAND_MAX+(1-thumb_close_value)*self._HAND_MIN
            
            HAND_GRIPPER_CLOSE = np.array([self._HAND_MAX, self._HAND_MAX, finger_close_value, finger_close_value, thumb_close_value, self._HAND_MAX])
            act.hand_target = HAND_GRIPPER_CLOSE.copy()
            # 增加最小步数限制，确保手指有足够时间闭合、接触并稳定
            if ctx.phase_step > self.MIN_PHASE_STEPS and ctx.phase_step >= 100:
                # 存储目前的末端位置用于 lift 阶段使用
                self._lift_target_pos = ee_pos.copy()
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "lift":
            # ========== 目标位置：上升到 lift 高度 ==========
            target_pos = self._lift_target_pos.copy()
            target_pos[2] += self.LIFT_HEIGHT + self.PRE_GRASP_HEIGHT
            
            # ========== 目标姿态 ==========
            target_quat = self._approach_quat_target

            # ========== 位置控制 ==========
            delta_pos = target_pos - ee_pos
            dist_pos = np.linalg.norm(delta_pos)
            if dist_pos > self.LIFT_SPEED:
                delta_pos = delta_pos / dist_pos * self.LIFT_SPEED
            act.ee_delta_pos = delta_pos

            # ========== 姿态控制（与 approach 阶段相同逻辑）==========
            inv_quat = np.zeros(4, dtype=np.float64)
            rel_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_negQuat(inv_quat, ee_quat)
            mujoco.mju_mulQuat(rel_quat, target_quat, inv_quat)

            axis_angle = np.zeros(3, dtype=np.float64)
            mujoco.mju_quat2Vel(axis_angle, rel_quat, 1.0)
            angle = np.linalg.norm(axis_angle)
            if angle > 0.1:
                axis_angle = axis_angle / angle * 0.1
            act.ee_delta_rot = axis_angle

            # 可视化目标
            act.ee_target_pos = target_pos.copy()
            act.ee_target_quat = target_quat.copy()

            act.hand_target = self.HAND_CLOSE.copy()

            # ========== 收敛判断（位置 + 姿态）==========
            pos_err = np.linalg.norm(ee_pos - target_pos)
            quat_err = quat_distance(ee_quat, target_quat)
            if ctx.phase_step > self.MIN_PHASE_STEPS and pos_err < 0.02 and quat_err < 0.15:
                self._hold_target_pos = ee_pos.copy()
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        return PhaseResult.ABORT, act
    
    def _rot_correction(self, ee_quat: np.ndarray, target_quat: np.ndarray,
                    max_step: float = 0.1) -> np.ndarray:
        """计算从当前姿态朝目标姿态的旋转修正量（axis-angle），限幅 max_step。"""
        inv_quat = np.zeros(4, dtype=np.float64)
        rel_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_negQuat(inv_quat, ee_quat)
        mujoco.mju_mulQuat(rel_quat, target_quat, inv_quat)
        axis_angle = np.zeros(3, dtype=np.float64)
        mujoco.mju_quat2Vel(axis_angle, rel_quat, 1.0)
        angle = np.linalg.norm(axis_angle)
        if angle > max_step:
            axis_angle = axis_angle / angle * max_step
        return axis_angle