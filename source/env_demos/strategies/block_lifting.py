"""
BlockLifting 任务策略实现.

阶段序列: make_gripper_hand_form → approach → descend → adjust → grasp → lift → check
"""

import time
from typing import Tuple, Optional

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
      2. descend   : 下降接触方块，同时精细调整姿态
      3. adjust    : 如果下降后末端位置在水平面上偏离了预定的抓取位置，进行微调回预定位置
      4. grasp     : 闭合手指抓取
      5. lift      : 垂直提升
      6. check     : 判定方块与 mid_point 位置误差，误差过大则判定抓取失败，从阶段0重新开始
    """

    # ========== 方案二：多高度候选，IK失败时自动搜索 ==========
    # 不再固定 PRE_GRASP_HEIGHT，而是提供候选高度列表
    PRE_GRASP_HEIGHT_CANDIDATES: list = [0.08, 0.10, 0.12, 0.15]
    GRASP_HEIGHT: float = 0.03
    LIFT_HEIGHT: float = 0.20
    APPROACH_SPEED: float = 0.03
    DESCEND_SPEED: float = 0.015
    LIFT_SPEED: float = 0.04
    MIN_PHASE_STEPS: int = 5

    # 判定阈值：方块与 mid_point 的最大允许距离（单位：米）
    CHECK_MAX_DISTANCE: float = 0.05

    # ========== 新增：环境变化检测阈值（宁可多清也不少清）==========
    CACHE_INVALIDATE_POS_THRESH: float = 0.01   # 1cm 位置跳变即清缓存
    CACHE_INVALIDATE_ROT_THRESH: float = np.deg2rad(5.0)  # 5° 旋转跳变即清缓存
    # =============================================================

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
            # ========== 新增：环境指纹也要清 ==========
            '_cached_obj_pos',
            '_cached_obj_quat',
            # =========================================
        ]
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)

    # ========== 新增：环境变化检测与缓存自洁 ==========
    def _maybe_invalidate_cache(self, env, current_obj_pos: np.ndarray):
        """
        检测环境是否发生显著变化（新回合、方块被移动/重置）。
        如果变化显著，清除所有与方块状态绑定的缓存，避免旧数据污染新决策。
        阈值设置较敏感：宁可多清也不少清。
        """
        # 首次运行：建立指纹
        if not hasattr(self, '_cached_obj_pos'):
            self._cached_obj_pos = current_obj_pos.copy()
            self._cached_obj_quat = env.get_block_quaternion().copy()
            return

        # 位置跳变检测
        pos_shift = np.linalg.norm(current_obj_pos - self._cached_obj_pos)
        if pos_shift > self.CACHE_INVALIDATE_POS_THRESH:
            # print(f"[CacheInvalidate] 方块位置突变 {pos_shift:.4f}m > {self.CACHE_INVALIDATE_POS_THRESH:.4f}m，清除旋转相关缓存")
            self._clear_computed_cache()
            self._cached_obj_pos = current_obj_pos.copy()
            self._cached_obj_quat = env.get_block_quaternion().copy()
            return

        # 旋转跳变检测
        current_quat = env.get_block_quaternion()
        dot = np.clip(np.dot(current_quat, self._cached_obj_quat), -1.0, 1.0)
        angle_diff = 2.0 * np.arccos(abs(dot))
        if angle_diff > self.CACHE_INVALIDATE_ROT_THRESH:
            # print(f"[CacheInvalidate] 方块旋转突变 {np.degrees(angle_diff):.1f}° > {np.degrees(self.CACHE_INVALIDATE_ROT_THRESH):.1f}°，清除旋转相关缓存")
            self._clear_computed_cache()
            self._cached_obj_pos = current_obj_pos.copy()
            self._cached_obj_quat = current_quat.copy()
            return

        # 正常更新指纹
        self._cached_obj_pos = current_obj_pos.copy()
        self._cached_obj_quat = current_quat.copy()
    # ==================================================

    def execute_phase(self, phase_idx: int, ctx: PhaseContext) -> Tuple[PhaseResult, ActionContext]:
        phase = self.phases[phase_idx]
        env = ctx.env
        act = ActionContext()

        obj_pos = env.get_block_position()
        table_z = env._table_height
        ee_pos, ee_quat = env.get_ee_pose()
        hand_qpos = env.get_hand_qpos()

        # ========== 仅在抓取前阶段检测环境变化 ==========
        # grasp/lift/check 阶段方块被夹持移动属于预期行为，不应触发缓存清除；
        # 只在 make_gripper_hand_form / approach / descend / adjust 阶段做检测，
        # 防止新回合方块重置或外力干扰导致旧缓存污染决策。
        _PRE_GRASP_PHASES = {"make_gripper_hand_form", "approach", "descend", "adjust"}
        if phase in _PRE_GRASP_PHASES:
            self._maybe_invalidate_cache(env, obj_pos)
        # ==================================================

        if phase == "make_gripper_hand_form":
            act.hand_target = self.HAND_GRIPPER.copy()
            act.ee_delta_pos = np.zeros(3)
            act.ee_delta_rot = np.zeros(3)
            max_error = np.max(np.abs(hand_qpos - act.hand_target))
            if ctx.phase_step > self.MIN_PHASE_STEPS and max_error < 0.001:
                if not hasattr(self, '_d_hand'):
                    # ===== 标定：计算 _d_hand 和 _finger_in_local（只需标定一次）=====
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

                # ===== 每回合都重新优化目标旋转矩阵（方块旋转每回合不同）=====
                if not hasattr(self, '_approach_R_target'):
                    from scipy.optimize import minimize_scalar

                    ee_pos_cal, ee_quat_cal = env.get_ee_pose()
                    obj_pos_cal = env.get_block_position()
                    obj_quat_cal = env.get_block_quaternion()  # [w, x, y, z]

                    # 获取方块的世界旋转矩阵（仅Z轴旋转）
                    R_obj_flat = np.zeros(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(R_obj_flat, obj_quat_cal)
                    R_obj = R_obj_flat.reshape(3, 3)

                    # 方块的X轴和Y轴方向（在XY平面内）
                    obj_x_axis = R_obj[:, 0].copy()
                    obj_y_axis = R_obj[:, 1].copy()

                    # 投影到XY平面并归一化
                    obj_x_axis[2] = 0
                    obj_y_axis[2] = 0
                    obj_x_axis /= np.linalg.norm(obj_x_axis)
                    obj_y_axis /= np.linalg.norm(obj_y_axis)

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

                    # ===== 收拢轴确认：actual closing vec ≈ -f_world =====
                    # f_local 定义为 finger3 - thumb（食指→拇指方向）
                    # 实测收拢方向 = -f_local 在世界的投影
                    # 因此对齐目标：让 -f_world_xy 对齐方块面法线（obj_x 或 obj_y）
                    R0 = _R_from_yaw(0.0)
                    # yaw=0 时 -f_local 在世界 XY 的绝对角
                    neg_f0_world = -(R0 @ f_local)
                    neg_f0_world[2] = 0.0
                    neg_f0_world /= np.linalg.norm(neg_f0_world)
                    neg_f0_angle = np.arctan2(neg_f0_world[1], neg_f0_world[0])

                    # ========== 方案二：IK 残差函数支持可变高度 ==========
                    def _ik_residual(yaw: float, pre_grasp_height: float) -> float:
                        """在虚拟副本上跑 DLS IK，返回收敛残差（越小越可达）."""
                        R   = _R_from_yaw(yaw)
                        # 用方块局部坐标系推算 approach ee 位置
                        d_world = R @ self._d_hand
                        target_ee_pos = np.array([
                            obj_pos_cal[0] - d_world[0],
                            obj_pos_cal[1] - d_world[1],
                            env._table_height + pre_grasp_height - d_world[2],
                        ])

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
                            err_p = target_ee_pos - tmp.site_xpos[ctrl.ee_id]
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
                        err_p = np.linalg.norm(target_ee_pos - tmp.site_xpos[ctrl.ee_id])
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
                        """计算收拢轴（-f_local）与方块面法线的对齐误差（越小越对齐）."""
                        R = _R_from_yaw(yaw)
                        closing_world = -(R @ f_local)
                        closing_world[2] = 0.0
                        norm = np.linalg.norm(closing_world)
                        if norm > 1e-6:
                            closing_world /= norm
                        dot_x = abs(np.dot(closing_world, obj_x_axis))
                        dot_y = abs(np.dot(closing_world, obj_y_axis))
                        return float(1.0 - max(dot_x, dot_y))

                    # ===== 候选 yaw：让收拢轴（th）对齐方块面法线 =====
                    R0 = _R_from_yaw(0.0)
                    # th 在局部坐标系是 h_local，世界方向 = R0 @ h_local
                    th0_world = R0 @ h_local
                    th0_world[2] = 0.0
                    th0_world /= np.linalg.norm(th0_world)
                    th0_angle = np.arctan2(th0_world[1], th0_world[0])

                    # obj_x_axis / obj_y_axis 的世界角
                    ax_angle = np.arctan2(obj_x_axis[1], obj_x_axis[0])
                    ay_angle = np.arctan2(obj_y_axis[1], obj_y_axis[0])

                    # 让 -f_world 对齐 obj_x_axis / obj_y_axis，±180° 各一个
                    base_x = ax_angle - neg_f0_angle
                    base_y = ay_angle - neg_f0_angle
                    candidates = [
                        (base_x         + np.pi) % (2 * np.pi) - np.pi,
                        (base_x + np.pi + np.pi) % (2 * np.pi) - np.pi,
                        (base_y         + np.pi) % (2 * np.pi) - np.pi,
                        (base_y + np.pi + np.pi) % (2 * np.pi) - np.pi,
                    ]

                    # ========== 方案二核心：多高度搜索，选 IK 最优且高度最低的组合 ==========
                    best_overall_cost = float('inf')
                    best_yaw = candidates[0]
                    best_height = self.PRE_GRASP_HEIGHT_CANDIDATES[0]
                    best_R = None

                    for height in self.PRE_GRASP_HEIGHT_CANDIDATES:
                        for cand_yaw in candidates:
                            ik_err = _ik_residual(cand_yaw, height)
                            # 成本 = IK残差 + 高度惩罚（越低越好，系数小避免过度惩罚）
                            # 高度惩罚：每高 1cm 增加 0.005 成本，鼓励选更低的高度
                            height_penalty = (height - self.PRE_GRASP_HEIGHT_CANDIDATES[0]) * 0.5
                            total_cost = ik_err + height_penalty

                            if total_cost < best_overall_cost:
                                best_overall_cost = total_cost
                                best_yaw = cand_yaw
                                best_height = height
                                best_R = _R_from_yaw(cand_yaw)

                    # 在最优候选附近 ±5° 内做连续细化（只优化 IK，不破坏对齐）
                    lo = best_yaw - np.pi / 36   # -5°
                    hi = best_yaw + np.pi / 36   # +5°
                    result = minimize_scalar(
                        lambda yaw: _ik_residual(yaw, best_height),
                        bounds=(lo, hi),
                        method='bounded',
                    )
                    best_yaw = result.x
                    best_R = _R_from_yaw(best_yaw)

                    self._approach_R_target = best_R
                    self._selected_pre_grasp_height = best_height  # 缓存选中的高度

                    # DEBUG: 验证最终选出的收拢轴方向
                    _closing_check = -(self._approach_R_target @ f_local)
                    _closing_check[2] = 0
                    _closing_check /= np.linalg.norm(_closing_check)
                    # print(f"[YAW] best_yaw={np.degrees(best_yaw):.1f}°, best_height={best_height:.3f}m")
                    # print(f"[YAW] closing_world_xy={_closing_check[:2]}")
                    # print(f"[YAW] obj_x={obj_x_axis[:2]}, obj_y={obj_y_axis[:2]}")
                    # print(f"[YAW] dot_x={abs(np.dot(_closing_check[:2], obj_x_axis[:2])):.3f}, dot_y={abs(np.dot(_closing_check[:2], obj_y_axis[:2])):.3f}")

                    q = np.zeros(4, dtype=np.float64)
                    mujoco.mju_mat2Quat(q, self._approach_R_target.flatten('C'))
                    self._approach_quat_target = q

                # ===== FIX: approach 目标 mid-point 需补偿 d_hand 的 XY 分量 =====
                obj_pos_now = env.get_block_position()
                d_world = self._approach_R_target @ self._d_hand

                # 使用搜索到的最佳高度
                target_ee_approach = np.array([
                    obj_pos_now[0] - d_world[0],
                    obj_pos_now[1] - d_world[1],
                    env._table_height + self._selected_pre_grasp_height - d_world[2],
                ])
                # approach 阶段用 mid-point 反推控制，缓存 mid-point 目标
                self._approach_target_mid = target_ee_approach + d_world

                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "approach":
            # ========== 方案四：approach 放宽姿态要求，只保证 XY 位置 ==========
            R_target    = self._approach_R_target
            target_quat = self._approach_quat_target

            # 目标 ee 位置：由目标 mid-point 反推
            target_ee_pos = self._approach_target_mid - R_target @ self._d_hand

            act.ee_target_pos  = target_ee_pos.copy()
            act.ee_target_quat = target_quat.copy()

            # ========== 位置控制（优先保证 XY 收敛）==========
            pos_delta = target_ee_pos - ee_pos
            dist = np.linalg.norm(pos_delta)
            if dist > self.APPROACH_SPEED:
                pos_delta = pos_delta / dist * self.APPROACH_SPEED
            act.ee_delta_pos = pos_delta

            # ========== 姿态控制（放宽：只纠正大角度误差）==========
            # 方案四：approach 阶段姿态误差阈值放宽，允许粗略对准
            inv_quat = np.zeros(4, dtype=np.float64)
            rel_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_negQuat(inv_quat, ee_quat)
            mujoco.mju_mulQuat(rel_quat, target_quat, inv_quat)

            axis_angle = np.zeros(3, dtype=np.float64)
            mujoco.mju_quat2Vel(axis_angle, rel_quat, 1.0)
            angle = np.linalg.norm(axis_angle)
            
            # 放宽姿态修正：最大步长从 0.1 改为 0.2，允许更快收敛
            max_rot_step = 0.2
            if angle > max_rot_step:
                axis_angle = axis_angle / angle * max_rot_step
            act.ee_delta_rot = axis_angle

            act.hand_target = self.HAND_GRIPPER.copy()

            # ========== 收敛判断（方案四：姿态要求降低）==========
            pos_err  = distance(ee_pos, target_ee_pos)
            quat_err = quat_distance(ee_quat, target_quat)
            # 放宽姿态收敛阈值：从 0.05 改为 0.15（约 8.6°），XY 位置保持严格
            xy_err = np.linalg.norm(ee_pos[:2] - target_ee_pos[:2])
            
            if ctx.phase_step > self.MIN_PHASE_STEPS and xy_err < 0.03 and quat_err < 0.15:
                # ===== FIX: descend 目标由方块位置 + R_target + GRASP_HEIGHT 完整推算 =====
                d_world = self._approach_R_target @ self._d_hand
                # 计算 0.25 倍方块高度位置（相对于桌面）
                block_height = 2.0 * (obj_pos[2] - env._table_height)  # 如果 obj_pos[2] 是中心，则高度 = 2*(中心-桌面)
                grasp_z = env._table_height + block_height * 0.25

                self._descend_target_pos = np.array([
                    obj_pos[0] - d_world[0],
                    obj_pos[1] - d_world[1],
                    grasp_z - d_world[2],
                ])
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "descend":
            # ===== 兜底守卫 =====
            if not hasattr(self, '_approach_quat_target') or not hasattr(self, '_descend_target_pos'):
                self._clear_computed_cache()
                return PhaseResult.RESTART, act
            # ====================
            # ========== 方案四：descend 阶段合并精细姿态对准 ==========
            target_pos = self._descend_target_pos
            
            # 位置控制：下降
            delta = target_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist > self.DESCEND_SPEED:
                delta = delta / dist * self.DESCEND_SPEED
            act.ee_delta_pos = delta
            
            # 姿态控制：精细修正（使用更严格的步长限制）
            act.ee_delta_rot = self._rot_correction(ee_quat, self._approach_quat_target, max_step=0.05)
            
            act.hand_target = self.HAND_GRIPPER.copy()
            act.ee_target_pos = target_pos.copy()
            act.ee_target_quat = self._approach_quat_target
            
            # 收敛判断：位置到达即可，姿态在下降过程中逐步修正
            if ctx.phase_step > self.MIN_PHASE_STEPS and ee_pos[2] <= target_pos[2] + 0.01:
                self._grasp_target_pos = ee_pos.copy()
                self._grasp_start_time = time.time()
                self._cube_pos_at_descend = obj_pos.copy()
                self._mid_point_at_descend = env.get_mid_point_position()
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "adjust":
            # ===== 兜底守卫 =====
            if not hasattr(self, '_approach_quat_target') or not hasattr(self, '_mid_point_at_descend'):
                self._clear_computed_cache()
                return PhaseResult.RESTART, act
            # ====================
            if not hasattr(self, '_adjust_step'):
                self._adjust_step = 0
                cube_now = env.get_block_position()
                self._adjust_target_mid = np.array([
                    cube_now[0],
                    cube_now[1],
                    self._mid_point_at_descend[2]
                ])

            self._adjust_step += 1

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

            current_mid = env.get_mid_point_position()
            mid_err = np.linalg.norm(current_mid[:2] - self._adjust_target_mid[:2])

            if self._adjust_step >= 50 or mid_err < 0.005:
                delattr(self, '_adjust_step')
                delattr(self, '_adjust_target_mid')
                self._grasp_target_pos = ee_pos.copy()
                return PhaseResult.NEXT, act

            return PhaseResult.CONTINUE, act

        elif phase == "grasp":
            # ===== 兜底守卫 =====
            if not hasattr(self, '_approach_quat_target') or not hasattr(self, '_grasp_target_pos'):
                self._clear_computed_cache()
                return PhaseResult.RESTART, act
            # ====================
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
            # ===== 兜底守卫：_approach_quat_target 不应在此阶段缺失，若缺失则重置 =====
            if not hasattr(self, '_approach_quat_target'):
                self._clear_computed_cache()
                return PhaseResult.RESTART, act
            # =========================================================================
            target_quat = self._approach_quat_target.copy()

            if not hasattr(self, '_lift_target_mid'):
                self._lift_target_mid = np.array([
                    self._cube_pos_at_descend[0],
                    self._cube_pos_at_descend[1],
                    env._table_height + self.LIFT_HEIGHT + self._selected_pre_grasp_height,
                ])
            target_mid = self._lift_target_mid

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

            delta_pos = target_ee_pos - ee_pos
            dist_pos = np.linalg.norm(delta_pos)
            if dist_pos > self.LIFT_SPEED:
                delta_pos = delta_pos / dist_pos * self.LIFT_SPEED
            act.ee_delta_pos = delta_pos

            act.ee_delta_rot = self._rot_correction(ee_quat, target_quat)

            act.ee_target_pos = target_ee_pos.copy()
            act.ee_target_quat = target_quat.copy()

            act.hand_target = self.HAND_CLOSE.copy()

            pos_err = np.linalg.norm(ee_pos - target_ee_pos)
            quat_err = quat_distance(ee_quat, target_quat)
            if (ctx.phase_step > self.MIN_PHASE_STEPS and pos_err < 0.02 and quat_err < 0.05) or ctx.phase_step > 300:
                self._hold_target_pos = ee_pos.copy()
                if hasattr(self, '_lift_target_mid'):
                    delattr(self, '_lift_target_mid')
                return PhaseResult.NEXT, act
            return PhaseResult.CONTINUE, act

        elif phase == "check":
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
                    # 清除与方块旋转相关的缓存，下一回合重新优化
                    for attr in ('_approach_R_target', '_approach_quat_target',
                                 '_approach_target_mid', '_descend_target_pos',
                                 '_grasp_target_pos', '_grasp_start_time',
                                 '_cube_pos_at_descend', '_mid_point_at_descend',
                                 '_lift_target_pos', '_hold_target_pos',
                                 '_selected_pre_grasp_height'):  # 新增：清除选中高度
                        if hasattr(self, attr):
                            delattr(self, attr)
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