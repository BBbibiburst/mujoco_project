"""
BlockLifting 任务策略实现.

阶段序列: make_gripper_hand_form → approach → descend → adjust → grasp → lift → check
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
      6. check     : 判定方块与 mid_point 位置误差，误差过大则判定抓取失败，从阶段0重新开始
    """

    # PRE_GRASP_HEIGHT 等同于环境中可视化目标高度marker平面的高度
    # 确保在 approach 阶段末端在方块上方且不太远
    PRE_GRASP_HEIGHT: float = 0.15
    GRASP_HEIGHT: float = 0.03
    LIFT_HEIGHT: float = 0.20
    APPROACH_SPEED: float = 0.03
    DESCEND_SPEED: float = 0.015
    LIFT_SPEED: float = 0.04
    MIN_PHASE_STEPS: int = 5

    # 判定阈值：方块与 mid_point 的最大允许距离（单位：米）
    CHECK_MAX_DISTANCE: float = 0.05

    _HAND_MAX: float = 0.0095
    _HAND_MIN: float = 0.0
    GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    HAND_OPEN = np.full(6, _HAND_MIN, dtype=np.float64)
    HAND_CLOSE = np.full(6, _HAND_MAX, dtype=np.float64)
    HAND_GRIPPER = np.array([_HAND_MAX, _HAND_MAX, _HAND_MIN, _HAND_MIN, _HAND_MIN, _HAND_MAX])

    @property
    def phases(self) -> list:
        return ["make_gripper_hand_form", "approach", "descend", "adjust", "grasp", "lift", "check"]

    def _clear_computed_cache(self):
        """清除所有阶段间缓存的计算量，用于重新计算策略时调用。"""
        attrs_to_clear = [
            '_d_hand',
            '_finger_in_local',
            '_approach_R_target',
            '_approach_quat_target',
            '_approach_target_mid',
            '_descend_target_pos',
            '_grasp_target_pos',
            '_grasp_start_time',
            '_cube_pos_at_descend',
            '_mid_point_at_descend',
            '_lift_target_pos',
            '_hold_target_pos',
            '_lift_target_mid',
        ]
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)

    def execute_phase(self, phase_idx: int, ctx: PhaseContext) -> Tuple[PhaseResult, ActionContext]:
        phase = self.phases[phase_idx]
        env = ctx.env
        act = ActionContext()

        obj_pos = env.get_block_position()
        table_z = env._table_height
        ee_pos, ee_quat = env.get_ee_pose()
        hand_qpos = env.get_hand_qpos()

        if phase == "make_gripper_hand_form":
            act.hand_target = self.HAND_GRIPPER.copy()
            act.ee_delta_pos = np.zeros(3)
            act.ee_delta_rot = np.zeros(3)
            max_error = np.max(np.abs(hand_qpos - act.hand_target))
            if ctx.phase_step > self.MIN_PHASE_STEPS and max_error < 0.001:
                if not hasattr(self, '_d_hand'):
                    # ===== 标定：计算 _d_hand 和 _finger_in_local =====
                    ee_pos_cal, ee_quat_cal = env.get_ee_pose()
                    mid_cal = env.get_mid_point_position()

                    R_ee_flat = np.zeros(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(R_ee_flat, ee_quat_cal)
                    R_ee_cal = R_ee_flat.reshape(3, 3)

                    self._d_hand = R_ee_cal.T @ (mid_cal - ee_pos_cal)

                    thumb_cal   = env.get_site_pos("inspirehand_fingertip_thumb")
                    finger3_cal = env.get_site_pos("inspirehand_fingertip_3")
                    finger_vec_world = finger3_cal - thumb_cal
                    finger_vec_local = R_ee_cal.T @ finger_vec_world
                    self._finger_in_local = finger_vec_local / np.linalg.norm(finger_vec_local)

                    # ===== 优化目标旋转矩阵：考虑方块旋转，对齐手指与方块边缘 =====
                    from scipy.optimize import minimize_scalar

                    obj_pos_cal = env.get_block_position()
                    obj_quat_cal = env.get_block_quaternion()  # [w, x, y, z]
                    
                    # 获取方块的世界旋转矩阵（仅Z轴旋转）
                    R_obj_flat = np.zeros(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(R_obj_flat, obj_quat_cal)
                    R_obj = R_obj_flat.reshape(3, 3)
                    
                    # 方块的X轴和Y轴方向（在XY平面内）
                    obj_x_axis = R_obj[:, 0]  # 方块局部X轴在世界坐标系中的方向
                    obj_y_axis = R_obj[:, 1]  # 方块局部Y轴在世界坐标系中的方向
                    
                    # 投影到XY平面并归一化（忽略Z分量，因为方块只绕Z轴旋转）
                    obj_x_axis[2] = 0
                    obj_y_axis[2] = 0
                    obj_x_axis /= np.linalg.norm(obj_x_axis)
                    obj_y_axis /= np.linalg.norm(obj_y_axis)

                    approach_mid = np.array([
                        obj_pos_cal[0],
                        obj_pos_cal[1],
                        env._table_height + self.PRE_GRASP_HEIGHT,
                    ])

                    f_local = self._finger_in_local
                    p_local = np.array([0.0, 0.0, -1.0])
                    p_perp  = p_local - np.dot(p_local, f_local) * f_local
                    p_perp /= np.linalg.norm(p_perp)
                    h_local = np.cross(f_local, p_perp)
                    h_local /= np.linalg.norm(h_local)
                    src_basis = np.column_stack([f_local, p_perp, h_local])
                    target_p = np.array([0.0, 0.0, -1.0])

                    def _R_from_yaw(yaw: float) -> np.ndarray:
                        tf = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                        th = np.cross(tf, target_p)
                        th /= np.linalg.norm(th)
                        tf = np.cross(target_p, th)
                        tf /= np.linalg.norm(tf)
                        return np.column_stack([tf, target_p, th]) @ src_basis.T

                    def _ik_residual(yaw: float) -> float:
                        """在虚拟副本上跑 DLS IK，返回收敛残差（越小越可达）."""
                        R   = _R_from_yaw(yaw)
                        pos = approach_mid - R @ self._d_hand

                        quat = np.zeros(4, dtype=np.float64)
                        mujoco.mju_mat2Quat(quat, R.flatten('C'))

                        ctrl = env.controller
                        tmp  = mujoco.MjData(env.model)
                        mujoco.mj_copyData(tmp, env.model, env.data)
                        tmp.qvel[:] = 0.0
                        q_try = env.get_arm_qpos().copy()
                        tmp.qpos[ctrl.arm_qpos_ids] = q_try
                        mujoco.mj_fwdPosition(env.model, tmp)

                        damping = 0.05
                        jac_p = np.zeros((3, env.model.nv))
                        jac_r = np.zeros((3, env.model.nv))

                        for _ in range(50):
                            err_p = pos - tmp.site_xpos[ctrl.ee_id]
                            quat_cur = np.zeros(4)
                            mujoco.mju_mat2Quat(quat_cur, tmp.site_xmat[ctrl.ee_id])
                            qt = quat.copy()
                            if np.dot(qt, quat_cur) < 0:
                                qt = -qt
                            neg_cur = np.zeros(4)
                            mujoco.mju_negQuat(neg_cur, quat_cur)
                            dq4 = np.zeros(4)
                            mujoco.mju_mulQuat(dq4, qt, neg_cur)
                            err_r = np.zeros(3)
                            mujoco.mju_quat2Vel(err_r, dq4, 1.0)

                            err_norm = np.linalg.norm(err_p) + np.linalg.norm(err_r)
                            if err_norm < 1e-6:
                                break

                            mujoco.mj_jacSite(env.model, tmp, jac_p, jac_r, ctrl.ee_id)
                            Jp = jac_p[:, ctrl.arm_qvel_ids]
                            Jr = jac_r[:, ctrl.arm_qvel_ids]
                            J  = np.vstack([Jp, Jr])
                            err = np.concatenate([err_p, err_r])

                            A = J @ J.T + damping**2 * np.eye(6)
                            try:
                                dq_j = J.T @ np.linalg.solve(A, err)
                            except np.linalg.LinAlgError:
                                dq_j = J.T @ np.linalg.pinv(A) @ err

                            q_try = np.clip(
                                q_try + dq_j,
                                ctrl.arm_range[:, 0],
                                ctrl.arm_range[:, 1],
                            )
                            tmp.qpos[ctrl.arm_qpos_ids] = q_try
                            mujoco.mj_fwdPosition(env.model, tmp)

                        # 最终残差
                        err_p = np.linalg.norm(pos - tmp.site_xpos[ctrl.ee_id])
                        quat_cur = np.zeros(4)
                        mujoco.mju_mat2Quat(quat_cur, tmp.site_xmat[ctrl.ee_id])
                        qt = quat.copy()
                        if np.dot(qt, quat_cur) < 0:
                            qt = -qt
                        neg_cur = np.zeros(4)
                        mujoco.mju_negQuat(neg_cur, quat_cur)
                        dq4 = np.zeros(4)
                        mujoco.mju_mulQuat(dq4, qt, neg_cur)
                        vel = np.zeros(3)
                        mujoco.mju_quat2Vel(vel, dq4, 1.0)
                        return float(err_p + np.linalg.norm(vel))

                    def _orientation_alignment_error(yaw: float) -> float:
                        """计算手指方向与方块边缘的对齐误差（越小越对齐）."""
                        R = _R_from_yaw(yaw)
                        # 手指方向在世界坐标系中
                        finger_world = R @ f_local
                        finger_world[2] = 0  # 投影到XY平面
                        norm = np.linalg.norm(finger_world)
                        if norm > 1e-6:
                            finger_world /= norm
                        
                        # 计算与方块X轴和Y轴的夹角（取最小值，因为可以对齐X或Y）
                        dot_x = abs(np.dot(finger_world, obj_x_axis))
                        dot_y = abs(np.dot(finger_world, obj_y_axis))
                        # 理想情况下 dot_x 或 dot_y 应该接近 1（平行）
                        # 误差 = 1 - max(|dot_x|, |dot_y|)
                        alignment_error = 1.0 - max(dot_x, dot_y)
                        return float(alignment_error)

                    # ===== 多目标优化：IK可达性 + 方向对齐 =====
                    # 先找出4个候选方向（与方块边缘对齐的yaw值）
                    # 手指方向 f_local 在yaw=0时的世界方向
                    R0 = _R_from_yaw(0.0)
                    finger0_world = R0 @ f_local
                    finger0_world[2] = 0
                    finger0_world /= np.linalg.norm(finger0_world)
                    
                    # 计算 finger0 与方块X轴的夹角
                    angle_to_x = np.arctan2(
                        np.cross(obj_x_axis[:2], finger0_world[:2]),
                        np.dot(obj_x_axis[:2], finger0_world[:2])
                    )
                    # 4个对齐候选：使手指与X轴或Y轴对齐
                    # 需要旋转的角度偏移
                    candidates = []
                    for base_angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                        # 对齐到X轴
                        candidates.append(base_angle - angle_to_x)
                        # 对齐到Y轴（再加90度）
                        candidates.append(base_angle - angle_to_x + np.pi/2)
                    
                    # 归一化到 [-pi, pi]
                    candidates = [(a + np.pi) % (2*np.pi) - np.pi for a in candidates]
                    
                    # 评估每个候选：组合代价 = IK残差 + 权重 * 对齐误差
                    best_cost = float('inf')
                    best_yaw = 0.0
                    
                    for cand_yaw in candidates:
                        # 限制在合理范围内
                        if abs(cand_yaw) > 0.5 * np.pi:
                            continue
                        
                        ik_err = _ik_residual(cand_yaw)
                        align_err = _orientation_alignment_error(cand_yaw)
                        # 组合代价：IK优先，方向对齐次之
                        cost = ik_err + 0.5 * align_err
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_yaw = cand_yaw

                    # 如果候选都不理想，回退到原始优化
                    if best_cost > 0.1:  # 阈值可调
                        result = minimize_scalar(
                            lambda y: _ik_residual(y) + 0.3 * _orientation_alignment_error(y),
                            bounds=(-0.5 * np.pi, 0.5 * np.pi),
                            method='bounded'
                        )
                        best_yaw = result.x

                    self._approach_R_target = _R_from_yaw(best_yaw)

                    q = np.zeros(4, dtype=np.float64)
                    mujoco.mju_mat2Quat(q, self._approach_R_target.flatten('C'))
                    self._approach_quat_target = q

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
            # print(f"ee pos: {ee_pos}, target pos: {target_ee_pos}")
            # print(f"ee quat: {ee_quat}, target quat: {target_quat}")
            if ctx.phase_step > self.MIN_PHASE_STEPS and pos_err < 0.05 and quat_err < 0.05:
                # ===== 修复：在 approach 收敛时直接用几何关系算出 descend 的完整目标 =====
                # 不依赖当前 ee_pos，完全由方块位置和固定的 R_target、_d_hand 推算
                # 目标：mid_point 对准方块中心（obj_pos）
                # p_mid = p_ee + R_target @ _d_hand  =>  p_ee = obj_pos - R_target @ _d_hand
                d_world = R_target @ self._d_hand
                self._descend_target_pos = obj_pos.copy() - d_world
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "descend":
            # _descend_target_pos 已在 approach 收敛时完整计算，直接使用
            target_pos = self._descend_target_pos
            delta = target_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist > self.DESCEND_SPEED:
                delta = delta / dist * self.DESCEND_SPEED
            act.ee_delta_pos = delta
            act.ee_delta_rot = self._rot_correction(ee_quat, self._approach_quat_target)
            act.hand_target = self.HAND_GRIPPER.copy()
            # 可视化
            act.ee_target_pos = target_pos.copy()
            act.ee_target_quat = self._approach_quat_target
            if ctx.phase_step > self.MIN_PHASE_STEPS and ee_pos[2] <= target_pos[2] + 0.01:
                self._grasp_target_pos = ee_pos.copy()
                self._grasp_start_time = time.time()
                self._cube_pos_at_descend = obj_pos.copy()
                self._mid_point_at_descend = env.get_mid_point_position()
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "adjust":
            # 初始化：只在进入 adjust 第一步时锁定目标
            if not hasattr(self, '_adjust_step'):
                self._adjust_step = 0
                # 锁定目标：当前方块位置 + 当前指尖中点高度（只调水平面，不调高度）
                cube_now = env.get_block_position()
                self._adjust_target_mid = np.array([
                    cube_now[0],
                    cube_now[1],
                    self._mid_point_at_descend[2]
                ])

            self._adjust_step += 1

            # 由目标 mid-point 反推目标 ee 位置
            # 当前姿态下，mid-point = ee_pos + R_ee @ _d_hand
            # 所以 ee_pos = target_mid - R_ee @ _d_hand
            R_ee_flat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(R_ee_flat, ee_quat)
            R_ee = R_ee_flat.reshape(3, 3)

            target_ee_pos = self._adjust_target_mid - R_ee @ self._d_hand

            delta = target_ee_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist > self.APPROACH_SPEED:
                delta = delta / dist * self.APPROACH_SPEED
            act.ee_delta_pos = delta

            act.ee_delta_rot = self._rot_correction(ee_quat, self._approach_quat_target)
            act.hand_target = self.HAND_GRIPPER.copy()

            # 收敛判断：检查指尖中点是否对准
            current_mid = env.get_mid_point_position()
            mid_err = np.linalg.norm(current_mid[:2] - self._adjust_target_mid[:2])

            if self._adjust_step >= 50 or mid_err < 0.005:
                delattr(self, '_adjust_step')
                delattr(self, '_adjust_target_mid')
                self._grasp_target_pos = ee_pos.copy()
                return PhaseResult.NEXT, act

            return PhaseResult.CONTINUE, act

        elif phase == "grasp":
            ee_delta_pos = self._grasp_target_pos - ee_pos
            ee_delta_rot = self._rot_correction(ee_quat, self._approach_quat_target)
            act.ee_delta_pos = ee_delta_pos
            act.ee_delta_rot = ee_delta_rot

            finger_close_value = 0.7
            thumb_close_value = 0.3
            finger_close_value = finger_close_value*self._HAND_MAX+(1-finger_close_value)*self._HAND_MIN
            thumb_close_value = thumb_close_value*self._HAND_MAX+(1-thumb_close_value)*self._HAND_MIN

            HAND_GRIPPER_CLOSE = np.array([self._HAND_MAX, self._HAND_MAX, finger_close_value, finger_close_value, thumb_close_value, self._HAND_MAX])
            act.hand_target = HAND_GRIPPER_CLOSE.copy()
            if ctx.phase_step > self.MIN_PHASE_STEPS and ctx.phase_step >= 100:
                self._lift_target_pos = ee_pos.copy()
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "lift":
            # ========== 保持 grasp 姿态不变 ==========
            target_quat = self._approach_quat_target.copy()

            # ========== 目标 mid-point：水平位置保持 grasp 时的方块位置，高度提升到目标 ==========
            if not hasattr(self, '_lift_target_mid'):
                self._lift_target_mid = np.array([
                    self._cube_pos_at_descend[0],
                    self._cube_pos_at_descend[1],
                    env._table_height + self.LIFT_HEIGHT + self.PRE_GRASP_HEIGHT,
                ])
            target_mid = self._lift_target_mid

            # ========== 由目标 mid-point 和固定姿态反推目标 ee 位置 ==========
            R_target_flat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(R_target_flat, target_quat)
            R_target = R_target_flat.reshape(3, 3)
            target_ee_pos = target_mid - R_target @ self._d_hand

            # ========== IK 可达性检查与修正 ==========
            ctrl = env.controller

            tmp = mujoco.MjData(env.model)
            mujoco.mj_copyData(tmp, env.model, env.data)
            tmp.qvel[:] = 0.0
            q_try = env.get_arm_qpos().copy()
            tmp.qpos[ctrl.arm_qpos_ids] = q_try
            mujoco.mj_fwdPosition(env.model, tmp)

            damping = 0.05
            jac_p = np.zeros((3, env.model.nv))
            jac_r = np.zeros((3, env.model.nv))

            for _ in range(100):
                err_p = target_ee_pos - tmp.site_xpos[ctrl.ee_id]

                quat_cur = np.zeros(4)
                mujoco.mju_mat2Quat(quat_cur, tmp.site_xmat[ctrl.ee_id])
                qt = target_quat.copy()
                if np.dot(qt, quat_cur) < 0:
                    qt = -qt
                neg_cur = np.zeros(4)
                mujoco.mju_negQuat(neg_cur, quat_cur)
                dq4 = np.zeros(4)
                mujoco.mju_mulQuat(dq4, qt, neg_cur)
                err_r = np.zeros(3)
                mujoco.mju_quat2Vel(err_r, dq4, 1.0)

                err_norm = np.linalg.norm(err_p) + np.linalg.norm(err_r)
                if err_norm < 1e-6:
                    break

                mujoco.mj_jacSite(env.model, tmp, jac_p, jac_r, ctrl.ee_id)
                Jp = jac_p[:, ctrl.arm_qvel_ids]
                Jr = jac_r[:, ctrl.arm_qvel_ids]
                J = np.vstack([Jp, Jr])
                err = np.concatenate([err_p, err_r])

                A = J @ J.T + damping**2 * np.eye(6)
                try:
                    dq_j = J.T @ np.linalg.solve(A, err)
                except np.linalg.LinAlgError:
                    dq_j = J.T @ np.linalg.pinv(A) @ err

                q_try = np.clip(
                    q_try + dq_j,
                    ctrl.arm_range[:, 0],
                    ctrl.arm_range[:, 1],
                )
                tmp.qpos[ctrl.arm_qpos_ids] = q_try
                mujoco.mj_fwdPosition(env.model, tmp)
            else:
                target_ee_pos = tmp.site_xpos[ctrl.ee_id].copy()
                target_mid = target_ee_pos + R_target @ self._d_hand

            del tmp

            # ========== 位置控制 ==========
            delta_pos = target_ee_pos - ee_pos
            dist_pos = np.linalg.norm(delta_pos)
            if dist_pos > self.LIFT_SPEED:
                delta_pos = delta_pos / dist_pos * self.LIFT_SPEED
            act.ee_delta_pos = delta_pos

            act.ee_delta_rot = self._rot_correction(ee_quat, target_quat)

            act.ee_target_pos = target_ee_pos.copy()
            act.ee_target_quat = target_quat.copy()

            act.hand_target = self.HAND_CLOSE.copy()

            # ========== 收敛判断 ==========
            pos_err = np.linalg.norm(ee_pos - target_ee_pos)
            quat_err = quat_distance(ee_quat, target_quat)
            if (ctx.phase_step > self.MIN_PHASE_STEPS and pos_err < 0.02 and quat_err < 0.05) or ctx.phase_step > 300:
                self._hold_target_pos = ee_pos.copy()
                if hasattr(self, '_lift_target_mid'):
                    delattr(self, '_lift_target_mid')
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "check":
            # ========== 判定阶段：检查方块与 mid_point 的位置误差 ==========
            cube_pos = env.get_block_position()
            mid_pos = env.get_mid_point_position()
            error = np.linalg.norm(cube_pos - mid_pos)

            act.ee_delta_pos = np.zeros(3)
            act.ee_delta_rot = np.zeros(3)
            act.hand_target = self.HAND_CLOSE.copy()

            if ctx.phase_step > self.MIN_PHASE_STEPS:
                if error > self.CHECK_MAX_DISTANCE:
                    self._clear_computed_cache()
                    return PhaseResult.RESTART, act
                else:
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