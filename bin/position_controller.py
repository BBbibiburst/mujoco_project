"""
操作空间控制器（OSC）与兼容旧版 IK/PD 控制实现
=====================================================

本模块实现两个控制器：
- OSC_PositionController：任务空间 OSC + 独立手部关节 PD 控制
- IK_PositionController：基于雅可比的末端位置/姿态 IK + 关节空间 PD 控制

设计要点：
- OSC 直接在笛卡尔空间定义期望加速度，通过 Λ 映射到关节力矩
- 采用截断 SVD 伪逆处理奇异性，避免在不可控方向施加力
- 手部自由度独立处理，避免将手部纳入任务空间控制
- 支持零空间恢复、速度前馈、力矩变化率限制和关节绝对限幅
"""

import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings
from enum import Enum, auto

class _CtrlMode(Enum):
    NONE      = auto()
    JOINT_PD  = auto()
    OSC       = auto()



# ====================== 控制参数配置 ======================

@dataclass
class OSCGains:
    """
    操作空间控制器增益配置。

    所有增益均定义在笛卡尔任务空间，物理意义直观：
    - Kp 对应弹簧刚度 [N/m 或 N·m/rad]
    - Kd 对应阻尼系数 [N·s/m 或 N·m·s/rad]
    临界阻尼条件：Kd ≈ 2 * sqrt(Kp)（对单位质量系统）

    Attributes:
        kp_pos:       位置比例增益（笛卡尔刚度）[N/m]。
                      建议范围：[100, 800]。
        kd_pos:       位置微分增益（笛卡尔阻尼）[N·s/m]。
                      临界阻尼参考：2*sqrt(kp_pos)，例如 kp=400 → kd≈40。
        kp_rot:       姿态比例增益 [N·m/rad]。
        kd_rot:       姿态微分增益 [N·m·s/rad]。
        
        # 关节空间增益（独立配置，不再从笛卡尔增益换算）
        kp_joint:     机械臂关节比例增益（关节空间 PD）[N·m/rad]。
                      仅用于 set_target 关节空间模式。
        kd_joint:     机械臂关节微分增益 [N·m·s/rad]。
        kp_hand:      机械手关节比例增益 [N·m/rad]。
        kd_hand:      机械手关节微分增益 [N·m·s/rad]。
        
        # 前馈与滤波
        ff_scale:     速度前馈缩放系数 [0~1]。
                      1.0 = 完全前馈目标速度（低延迟轨迹跟踪）；
                      0.0 = 纯 PD（退化为经典 OSC）。
                      建议：平滑目标用 0.8~1.0，有噪声目标用 0.3~0.6。
        vel_filter_alpha: 目标速度低通滤波系数 [0~1]。
                          越小越平滑，越大响应越快。建议 0.4~0.7。
        
        # 数值稳定性
        singular_thresh:  SVD 截断阈值，低于此奇异值的方向被忽略。
                          建议 [0.01, 0.1]，值越大奇异处越保守。
        
        # 零空间控制
        null_kp:      零空间姿态恢复增益（将冗余自由度拉向参考构型）。
                      0 = 不使用零空间控制。
        null_kd:      零空间阻尼增益，建议 2*sqrt(null_kp) 实现临界阻尼。
        
        # 安全限制
        torque_rate_limit: 关节力矩变化率限制 [N·m/step]。
                           防止力矩突变导致振动或硬件损坏。
                           建议根据电机响应特性设置，通常 20-100 N·m/step。
    """
    # 笛卡尔任务空间增益
    kp_pos: float = 400.0
    kd_pos: float = 40.0
    kp_rot: float = 100.0
    kd_rot: float = 20.0
    
    # 关节空间增益（独立配置，避免魔法换算）
    kp_joint: float = 40000.0  # 关节空间刚度通常比笛卡尔高 100 倍量级
    kd_joint: float = 400.0    # 对应临界阻尼
    
    # 手部增益
    kp_hand: np.ndarray = field(default_factory=lambda: np.full(6, 400000.))
    kd_hand: np.ndarray = field(default_factory=lambda: np.full(6, 400.))
    
    # 前馈与滤波
    ff_scale: float = 0.9
    vel_filter_alpha: float = 0.5
    
    # 数值稳定性
    singular_thresh: float = 0.01
    
    # 零空间控制
    null_kp: float = 10.0
    null_kd: Optional[float] = None  # None 时自动计算为 2*sqrt(null_kp)
    
    # 安全限制
    torque_rate_limit: float = 50.0

    def __post_init__(self):
        """自动计算默认零空间阻尼"""
        if self.null_kd is None:
            self.null_kd = 2.0 * np.sqrt(self.null_kp)


# ====================== 预设增益配置 ======================

Stable_OSCGains = OSCGains(
    kp_pos=500,
    kd_pos=35,            # 轻微欠阻尼
    kp_rot=120,
    kd_rot=16,
    kp_joint=50000.0,     # 高刚度用于初始化
    kd_joint=450.0,
    ff_scale=0.95,
    vel_filter_alpha=0.4,
    singular_thresh=0.02,
    null_kp=3,
    null_kd=3.5,          # 略高于临界阻尼，保守
    torque_rate_limit=30.0,  # 保守的力矩变化率
)

FastTracking_OSCGains = OSCGains(
    kp_pos=600,           # 高刚度
    kd_pos=30,            # 欠阻尼（1.2*sqrt(600)≈29）
    kp_rot=150,
    kd_rot=15,
    kp_joint=60000.0,
    kd_joint=400.0,
    ff_scale=1.0,         # 完全前馈
    vel_filter_alpha=0.3, # 快速响应，依赖平滑输入
    singular_thresh=0.01, # 较激进
    null_kp=5,
    null_kd=4.5,          # 弱零空间约束
    torque_rate_limit=80.0,  # 允许较快的力矩变化
)

FastPointToPoint_OSCGains = OSCGains(
    kp_pos=800,
    kd_pos=28,            # 明显欠阻尼 ~sqrt(800)
    kp_rot=200,
    kd_rot=20,
    kp_joint=80000.0,
    kd_joint=350.0,
    ff_scale=0.0,         # 纯 PD，无前馈（目标跳变时更稳定）
    vel_filter_alpha=0.5,
    singular_thresh=0.05,
    null_kp=0,            # 禁用零空间，全自由度用于速度
    null_kd=0.0,
    torque_rate_limit=100.0,  # 点到点运动允许快速调整
)


class OSC_PositionController:
    """
    操作空间控制器（OSC）+ 手部关节 PD 控制器。

    OSC 核心：直接在笛卡尔任务空间定义期望力/加速度，通过惯量矩阵映射
    得到关节力矩，避免了旧版 IK+PD 的两层解耦结构性延迟。

    接口与旧版完全兼容：
        set_ee_target(data, ee_pos_target, ee_quat_target, hand_target)
        set_target(data, arm_target, hand_target)  # 保留，退化为关节 PD
        reset_targets(data)

    Attributes:
        base:              硬件/模型抽象接口。
        model:             MuJoCo 模型对象。
        gains:             OSC 增益配置。
        arm_qpos_ids:      机械臂 qpos 索引。
        arm_qvel_ids:      机械臂 qvel 索引。
        arm_joint_ids:     机械臂 joint 索引。
        arm_range:         机械臂关节限位 (n, 2)。
        hand_*:            同上，针对机械手。
        _null_qpos_ref:    零空间恢复目标构型（默认为初始化时的 qpos）。
        _prev_pos_target:  上帧位置目标（速度估计用）。
        _prev_quat_target: 上帧姿态目标（角速度估计用）。
        _vel_ff_pos:       滤波后的目标位置速度前馈。
        _vel_ff_rot:       滤波后的目标角速度前馈。
        ee_id:             末端 Site 的 MuJoCo ID。
        jac_p / jac_r:     雅可比缓冲区（预分配）。
        _M_full:           完整惯量矩阵缓冲区（预分配，nv×nv）。
        _hand_target:      手部目标缓存。
        _arm_target:       关节空间目标缓存（用于 set_target）。
        _prev_tau:         上帧力矩（用于变化率限制）。
    """

    def __init__(
        self,
        base,
        model: mujoco.MjModel,
        gains: Optional[OSCGains] = None,
        ee_site_name: str = "right_hand_site",
        null_qpos_ref: Optional[np.ndarray] = None,
    ):
        """
        Args:
            base:          硬件抽象接口，需提供 arm_names / hand_names /
                           hand_key_order / actuator_map / torque_min/max /
                           ARM_DOF / HAND_DOF / apply_control。
            model:         MuJoCo 模型。
            gains:         OSC 增益，None 时使用默认值。
            ee_site_name:  末端执行器 Site 名称。
            null_qpos_ref: 零空间恢复目标构型，None 时在首帧从 data.qpos 初始化。
        """
        self.base = base
        self.model = model
        self.gains = gains if gains is not None else OSCGains()

        # ----- 1. 关节索引解析 -----
        self.arm_qpos_ids, self.arm_qvel_ids, self.arm_joint_ids = \
            self._resolve_joint_ids(base.arm_names)
        self.hand_qpos_ids, self.hand_qvel_ids, self.hand_joint_ids = \
            self._resolve_joint_ids(
                [base.hand_names[k] for k in base.hand_key_order if k in base.hand_names]
            )

        # ----- 2. 关节范围 -----
        self.arm_range = model.jnt_range[self.arm_joint_ids]
        self.hand_range = model.jnt_range[self.hand_joint_ids]

        # ----- 3. 力矩限制 -----
        self._torque_min = base.torque_min
        self._torque_max = base.torque_max

        # ----- 4. 预分配缓冲区 -----
        nv = model.nv
        self.jac_p = np.zeros((3, nv))
        self.jac_r = np.zeros((3, nv))
        self._M_full = np.zeros((nv, nv))
        self._arm_torques = np.zeros(base.ARM_DOF)
        self._hand_torques = np.zeros(base.HAND_DOF)

        # ----- 5. 末端 Site ID -----
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_id == -1:
            raise ValueError(f"Site '{ee_site_name}' not found in MuJoCo model.")
        self.ee_site_name = ee_site_name

        # ----- 6. 零空间参考构型 -----
        self._null_qpos_ref: Optional[np.ndarray] = (
            null_qpos_ref.copy() if null_qpos_ref is not None else None
        )

        # ----- 7. 速度前馈状态 -----
        self._prev_pos_target: Optional[np.ndarray] = None
        self._prev_quat_target: Optional[np.ndarray] = None
        self._vel_ff_pos = np.zeros(3)
        self._vel_ff_rot = np.zeros(3)

        # ----- 8. 目标缓存 -----
        self._hand_target: Optional[np.ndarray] = None
        self._arm_target: Optional[np.ndarray] = None  # 用于 set_target 兼容

        # ----- 9. 力矩变化率限制状态 -----
        self._prev_tau: Optional[np.ndarray] = None
        self._ctrl_mode = _CtrlMode.NONE        # ✅ 新增模式追踪

    # =========================================================
    # 公共接口 1：末端位置/姿态控制（OSC 模式）
    # =========================================================

    def set_ee_target(
        self,
        data: mujoco.MjData,
        ee_pos_target: np.ndarray,
        ee_quat_target: Optional[np.ndarray] = None,
        hand_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        操作空间控制（OSC）主接口。

        计算并应用末端执行器的笛卡尔空间控制力矩，同时独立控制手部。

        Args:
            data:           MuJoCo 数据对象。
            ee_pos_target:  目标位置 [3,]。
            ee_quat_target: 目标姿态（四元数 [w,x,y,z]）[4,]，可选。
            hand_target:    手部目标关节角 [HAND_DOF,]，可选。

        Returns:
            Tuple[np.ndarray, np.ndarray]: (arm_torques, hand_torques) 实际应用的力矩。
        """
        # ✅ 检测模式切换，切换时清除历史力矩
        if self._ctrl_mode != _CtrlMode.OSC:
            self._prev_tau = None
            self._ctrl_mode = _CtrlMode.OSC
        g = self.gains
        dt = self.model.opt.timestep

        # 初始化零空间参考（首次调用）
        if self._null_qpos_ref is None:
            self._null_qpos_ref = data.qpos[self.arm_qpos_ids].copy()

        # ==================== 1. 速度前馈计算 ====================
        self._update_velocity_feedforward(ee_pos_target, ee_quat_target, dt, g)

        # ==================== 2. 当前状态获取 ====================
        ee_pos = data.site_xpos[self.ee_id].copy()

        mujoco.mj_jacSite(self.model, data, self.jac_p, self.jac_r, self.ee_id)
        J_p = self.jac_p[:, self.arm_qvel_ids].copy()
        J_r = self.jac_r[:, self.arm_qvel_ids].copy()

        qd = data.qvel[self.arm_qvel_ids]
        ee_vel = J_p @ qd

        # ==================== 3. 笛卡尔误差计算 ====================
        xacc_des, xacc_rot = self._compute_task_acceleration(
            ee_pos, ee_pos_target, J_p, J_r, qd, g, ee_quat_target, data
        )

        # ==================== 4. 操作空间动力学 ====================
        tau_arm = self._compute_osc_torque(data, J_p, J_r, xacc_des, xacc_rot, g)

        # ==================== 5. 力矩限制与平滑 ====================
        tau_arm = self._apply_torque_limits(tau_arm, g)

        self._arm_torques[:] = tau_arm

        # ==================== 6. 手部控制 ====================
        self._update_hand(data, hand_target)

        # ==================== 7. 应用控制 ====================
        self.base.apply_control(data, self._arm_torques, self._hand_torques)

        return self._arm_torques.copy(), self._hand_torques.copy()

    def _update_velocity_feedforward(
        self,
        ee_pos_target: np.ndarray,
        ee_quat_target: Optional[np.ndarray],
        dt: float,
        g: OSCGains,
    ) -> None:
        """更新目标位置/姿态的速度前馈，使用一阶低通滤波平滑目标速度。"""
        alpha = g.vel_filter_alpha

        # 位置速度前馈：目标位移 / dt
        if self._prev_pos_target is not None:
            raw_vel = (ee_pos_target - self._prev_pos_target) / dt
            self._vel_ff_pos = alpha * raw_vel + (1 - alpha) * self._vel_ff_pos
        else:
            self._vel_ff_pos[:] = 0.0

        # 姿态角速度前馈
        if ee_quat_target is not None and self._prev_quat_target is not None:
            # 计算相对旋转的四元数
            neg_prev = np.zeros(4)
            mujoco.mju_negQuat(neg_prev, self._prev_quat_target)
            dq = np.zeros(4)
            mujoco.mju_mulQuat(dq, ee_quat_target, neg_prev)

            # 转换为角速度
            raw_angvel = np.zeros(3)
            mujoco.mju_quat2Vel(raw_angvel, dq, 1.0)
            raw_angvel /= dt

            self._vel_ff_rot = alpha * raw_angvel + (1 - alpha) * self._vel_ff_rot
        else:
            self._vel_ff_rot[:] = 0.0

        # 更新历史
        self._prev_pos_target = ee_pos_target.copy()
        self._prev_quat_target = ee_quat_target.copy() if ee_quat_target is not None else None

    def _compute_task_acceleration(
        self,
        ee_pos: np.ndarray,
        ee_pos_target: np.ndarray,
        J_p: np.ndarray,
        J_r: np.ndarray,
        qd: np.ndarray,
        g: OSCGains,
        ee_quat_target: Optional[np.ndarray],
        data: mujoco.MjData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算任务空间期望加速度"""
        # 位置误差
        e_pos = ee_pos_target - ee_pos
        e_vel = g.ff_scale * self._vel_ff_pos - (J_p @ qd)
        xacc_des = g.kp_pos * e_pos + g.kd_pos * e_vel

        # 姿态误差（可选）
        xacc_rot = np.zeros(3)
        if ee_quat_target is not None:
            quat_cur = np.zeros(4)
            mujoco.mju_mat2Quat(quat_cur, data.site_xmat[self.ee_id])

            # 处理四元数符号二义性：对齐目标四元数和当前四元数，避免绕反方向旋转
            quat_target_aligned = ee_quat_target.copy()
            if np.dot(quat_target_aligned, quat_cur) < 0:
                quat_target_aligned = -quat_target_aligned

            # 计算姿态误差（旋转矢量）
            neg_cur = np.zeros(4)
            mujoco.mju_negQuat(neg_cur, quat_cur)
            dq = np.zeros(4)
            mujoco.mju_mulQuat(dq, quat_target_aligned, neg_cur)

            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, dq, 1.0)

            # 姿态速度误差
            ee_angvel = J_r @ qd
            e_vel_rot = g.ff_scale * self._vel_ff_rot - ee_angvel

            xacc_rot = g.kp_rot * e_rot + g.kd_rot * e_vel_rot

        return xacc_des, xacc_rot

    # ✅ 修改后：选择标准 OSC 公式，只补偿一次
    def _compute_osc_torque(self, data, J_p, J_r, xacc_des, xacc_rot, g):
        arm_idx = self.arm_qvel_ids

        mujoco.mj_fullM(self.model, self._M_full, data.qM)
        M = self._M_full[np.ix_(arm_idx, arm_idx)]

        if np.any(xacc_rot != 0):
            J = np.vstack([J_p, J_r])
            xacc = np.concatenate([xacc_des, xacc_rot])
        else:
            J = J_p
            xacc = xacc_des

        X = np.linalg.solve(M, J.T)
        JMinvJT = J @ X
        Lambda = self._svd_pinv(JMinvJT, g.singular_thresh)
        Lambda = 0.5 * (Lambda + Lambda.T)

        bias = data.qfrc_bias[arm_idx]

        # ✅ 标准 OSC：F = Λ(ẍ_des + J M⁻¹ bias) 使得 tau = Jᵀ F 自然包含动力学补偿
        # 等价于：tau = Jᵀ Λ ẍ_des + Jᵀ Λ J M⁻¹ bias
        # 其中 Jᵀ Λ J M⁻¹ 是任务空间投影的动力学补偿，不是全量 bias
        F_task = Lambda @ (xacc + J @ np.linalg.solve(M, bias))
        tau_task = J.T @ F_task

        # ✅ 零空间补偿保留完整 bias（零空间不受任务空间控制）
        tau_null = self._compute_null_space_torque(data, J, M, Lambda, arm_idx, g)

        # ✅ 残余补偿：bias 中未被任务空间投影覆盖的部分
        # N_null = I - Jᵀ(J M⁻¹ Jᵀ)⁻¹ J M⁻¹ 是零空间投影
        J_bar = np.linalg.solve(M, J.T) @ Lambda   # (n, 6)
        N = np.eye(len(arm_idx)) - J_bar @ J        # 零空间投影矩阵
        tau_bias_null = N.T @ bias                  # 仅补偿零空间方向的 bias

        return tau_task + tau_bias_null + tau_null
    
    def _compute_null_space_torque(
        self,
        data: mujoco.MjData,
        J: np.ndarray,
        M: np.ndarray,
        Lambda: np.ndarray,
        arm_idx: np.ndarray,
        g: OSCGains,
    ) -> np.ndarray:
        """计算零空间恢复力矩"""
        if g.null_kp <= 0 or self._null_qpos_ref is None:
            return np.zeros(len(arm_idx))

        # 零空间投影矩阵 N = I - J^T * J_bar
        # 其中 J_bar = M^-1 J^T Λ 是动力学一致性伪逆
        J_bar = np.linalg.solve(M, J.T) @ Lambda
        N = np.eye(len(arm_idx)) - J_bar @ J

        # 零空间 PD 控制
        q = data.qpos[self.arm_qpos_ids]
        qd = data.qvel[arm_idx]
        q_err = self._null_qpos_ref - q

        tau_null_raw = g.null_kp * q_err - g.null_kd * qd

        return N.T @ tau_null_raw

    def _apply_torque_limits(self, tau_raw: np.ndarray, g: OSCGains) -> np.ndarray:
        """应用力矩变化率限制和绝对限制"""
        # 变化率限制（平滑）
        if self._prev_tau is not None:
            max_delta = g.torque_rate_limit
            delta = tau_raw - self._prev_tau
            delta = np.clip(delta, -max_delta, max_delta)
            tau_smooth = self._prev_tau + delta
        else:
            tau_smooth = tau_raw

        self._prev_tau = tau_smooth.copy()

        # 绝对限制（硬件保护）
        return np.clip(
            tau_smooth,
            self._torque_min[:self.base.ARM_DOF],
            self._torque_max[:self.base.ARM_DOF],
        )

    # =========================================================
    # 公共接口 2：关节空间控制（兼容旧版接口）
    # =========================================================

    def set_target(
        self,
        data: mujoco.MjData,
        arm_target: np.ndarray,
        hand_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        关节空间 PD 控制（兼容旧版接口）。

        注意：此方法绕过 OSC，直接在关节空间做 PD。
        适合初始化阶段将机械臂移动到起始位置，或纯关节空间任务。
        正常跟踪任务请使用 set_ee_target。

        Args:
            data:        MuJoCo 数据对象。
            arm_target:  机械臂目标关节角度 [ARM_DOF,]。
            hand_target: 手部目标关节角 [HAND_DOF,]，可选。

        Returns:
            Tuple[np.ndarray, np.ndarray]: (arm_torques, hand_torques) 实际应用的力矩。
        """
        if self._ctrl_mode != _CtrlMode.JOINT_PD:
            self._prev_tau = None
            self._ctrl_mode = _CtrlMode.JOINT_PD
        g = self.gains

        # 缓存目标（保持行为一致性）
        if self._arm_target is None:
            self._arm_target = data.qpos[self.arm_qpos_ids].copy()

        # 限幅到关节范围
        self._arm_target = np.clip(
            arm_target, self.arm_range[:, 0], self.arm_range[:, 1]
        )

        # 关节空间 PD（使用独立配置的增益）
        e_q = self._arm_target - data.qpos[self.arm_qpos_ids]
        e_qd = -data.qvel[self.arm_qvel_ids]
        
        tau_arm = g.kp_joint * e_q + g.kd_joint * e_qd + data.qfrc_bias[self.arm_qvel_ids]

        # 应用限制
        tau_arm = self._apply_torque_limits(tau_arm, g)
        self._arm_torques[:] = tau_arm

        # 手部控制
        self._update_hand(data, hand_target)

        # 应用控制
        self.base.apply_control(data, self._arm_torques, self._hand_torques)

        return self._arm_torques.copy(), self._hand_torques.copy()

    # =========================================================
    # 公共接口 3：重置状态
    # =========================================================

    def reset_targets(self, data: mujoco.MjData) -> None:
        """
        重置所有控制器状态。

        在轨迹切换、急停或重新初始化时调用。
        - 清空速度前馈历史（防止切换时速度跳变）
        - 将零空间参考构型更新为当前实际构型
        - 重置力矩历史（避免变化率限制误触发）

        Args:
            data: MuJoCo 数据对象。
        """
        self._prev_pos_target = None
        self._prev_quat_target = None
        self._vel_ff_pos[:] = 0.0
        self._vel_ff_rot[:] = 0.0
        self._hand_target = data.qpos[self.hand_qpos_ids].copy()
        self._arm_target = data.qpos[self.arm_qpos_ids].copy()
        self._null_qpos_ref = data.qpos[self.arm_qpos_ids].copy()
        self._prev_tau = None  # 重置力矩历史，避免首步限制

    # =========================================================
    # 私有方法
    # =========================================================

    def _update_hand(self, data: mujoco.MjData, hand_target: Optional[np.ndarray]) -> None:
        """
        手部关节 PD 控制（内部调用）。

        手部自由度不纳入 OSC 任务空间，使用独立的关节 PD 控制。
        hand_target=None 时保持上次缓存目标（不跟随实际值，避免跟随滞后）。
        """
        g = self.gains

        if self._hand_target is None:
            self._hand_target = data.qpos[self.hand_qpos_ids].copy()
        
        if hand_target is not None:
            self._hand_target = np.clip(
                hand_target, self.hand_range[:, 0], self.hand_range[:, 1]
            )

        e_q = self._hand_target - data.qpos[self.hand_qpos_ids]
        e_qd = data.qvel[self.hand_qvel_ids]
        tau = g.kp_hand * e_q - g.kd_hand * e_qd

        self._hand_torques[:] = np.clip(
            tau,
            self._torque_min[self.base.ARM_DOF:],
            self._torque_max[self.base.ARM_DOF:],
        )

    @staticmethod
    def _svd_pinv(A: np.ndarray, thresh: float) -> np.ndarray:
        """
        SVD 截断伪逆（Truncated SVD Pseudo-inverse）。

        相较于 DLS（阻尼最小二乘），TSVD 的优势：
        - 奇异方向直接清零（物理意义明确：不在不可控方向施力）
        - 非奇异方向不受阻尼影响（不牺牲正常方向的精度）
        - 阈值 thresh 含义直观：低于此奇异值的方向被视为奇异

        Args:
            A:      待求伪逆的方阵（通常为 J M⁻¹ J^T）。
            thresh: 奇异值截断阈值。

        Returns:
            A 的截断伪逆。
        """
        U, s, Vt = np.linalg.svd(A)
        s_inv = np.where(s > thresh, 1.0 / s, 0.0)
        return (Vt.T * s_inv) @ U.T

    def _resolve_joint_ids(
        self, actuator_names
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将执行器名称解析为 MuJoCo 内部的 qpos、qvel、joint 三组索引。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (qpos_ids, qvel_ids, joint_ids)
        """
        qpos_ids, qvel_ids, joint_ids = [], [], []
        
        for name in actuator_names:
            act_id = self.base.actuator_map[name]
            joint_id = self.model.actuator_trnid[act_id, 0]
            joint_ids.append(joint_id)
            qpos_ids.append(self.model.jnt_qposadr[joint_id])
            qvel_ids.append(self.model.jnt_dofadr[joint_id])
            
        return (
            np.array(qpos_ids, dtype=np.int32),
            np.array(qvel_ids, dtype=np.int32),
            np.array(joint_ids, dtype=np.int32),
        )

    # =========================================================
    # 工具方法（可选）
    # =========================================================

    def get_ee_pose(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前末端位姿（位置和四元数）"""
        pos = data.site_xpos[self.ee_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self.ee_id])
        return pos, quat

    def check_singularity(self, data: mujoco.MjData, threshold: Optional[float] = None) -> bool:
        """
        检查当前是否接近奇异构型。

        Returns:
            bool: 如果最小奇异值低于阈值则返回 True。
        """
        if threshold is None:
            threshold = self.gains.singular_thresh

        mujoco.mj_jacSite(self.model, data, self.jac_p, self.jac_r, self.ee_id)
        J = np.vstack([
            self.jac_p[:, self.arm_qvel_ids],
            self.jac_r[:, self.arm_qvel_ids]
        ])
        
        _, s, _ = np.linalg.svd(J)
        return np.min(s) < threshold
    
    
"""
Legacy IK/PD 混合控制器模块

本段实现一个兼容旧版接口的混合控制器：
- 上层 IK：将末端位姿误差转换为关节增量
- 下层 PD：对关节目标施加 PD 控制并做力矩饱和保护

适用场景：初始化、粗略轨迹预定位或传统关节空间控制。
"""


# ====================== 控制参数配置 ======================
@dataclass
class PDGains:
    """
    PD 控制器增益配置容器.
    
    采用 dataclass 实现不可变配置对象。增益值直接影响系统的刚度和阻尼特性。
    默认值设定依据：
    - 机械臂 (Arm): 通常需要较高的刚度以抵抗重力和外部扰动
    - 机械手 (Hand): 需要相对灵活，避免过大的刚度导致碰撞冲击
    
    Attributes:
        kp_arm: 机械臂比例增益 (刚度) [N·m/rad]。值越大定位越硬。
        kd_arm: 机械臂微分增益 (阻尼) [N·m·s/rad]。用于抑制运动过程中的振荡。
        kp_hand: 机械手比例增益。通常设置为与臂同量级或略低。
        kd_hand: 机械手微分增益。
    """
    kp_arm: np.ndarray = field(default_factory=lambda: np.full(7, 40000.))
    kd_arm: np.ndarray = field(default_factory=lambda: np.full(7, 400.))
    kp_hand: np.ndarray = field(default_factory=lambda: np.full(6, 40000.))
    kd_hand: np.ndarray = field(default_factory=lambda: np.full(6, 400.))


class IK_PositionController:
    """
    混合位置控制器 (IK + PD).
    
    该控制器采用两层架构：
    1. 上层 (IK Solver): 将末端执行器 (End-Effector) 的空间目标转换为关节空间的目标增量。
    2. 下层 (PD Controller): 接收关节目标，计算所需的关节力矩并应用饱和限制。
    
    Attributes:
        base: 硬件/模型抽象接口，提供 DOF 数量、名称映射和力矩限制。
        model: MuJoCo 模型对象 (mjModel)，用于雅可比计算。
        gains: PD 增益配置。
        arm_qpos_ids: 机械臂位置自由度 (qpos) 在全局数组中的索引。
        arm_qvel_ids: 机械臂速度自由度 (qvel) 在全局数组中的索引。
        arm_range: 机械臂关节的物理限位范围。
        _torque_min/_torque_max: 读取自 base 的执行器力矩限制。
        ee_id: 末端执行器 Site 在 MuJoCo 模型中的 ID。
        jac_p/jac_r: 用于存储位置和旋转雅可比矩阵的缓冲区。
    """

    def __init__(self, base, model, gains: PDGains | None = None):
        """
        初始化控制器。
        
        Args:
            base: 包含机器人硬件参数的基础对象。
            model: MuJoCo 模型实例。
            gains: 可选的自定义 PD 增益。若为 None，则使用 PDGains 默认值。
        """
        self.base = base
        self.model = model
        self.gains = gains if gains is not None else PDGains()

        # ----- 1. 关节索引解析 -----
        # 通过 Base 接口获取关节名称，并映射为 MuJoCo 内部索引
        # 这种映射避免了硬编码索引，提高了代码对不同 URDF/XML 的适应性
        # ✅ 改为传 actuator_names，在 _resolve_joint_ids 里同时返回 joint_ids
        self.arm_qpos_ids, self.arm_qvel_ids, self.arm_joint_ids = \
            self._resolve_joint_ids(base.arm_names)
        self.hand_qpos_ids, self.hand_qvel_ids, self.hand_joint_ids = \
            self._resolve_joint_ids(
                [base.hand_names[k] for k in base.hand_key_order if k in base.hand_names]
            )

        # ✅ 直接用 joint_ids 取范围，安全可靠
        self.arm_range  = model.jnt_range[self.arm_joint_ids]
        self.hand_range = model.jnt_range[self.hand_joint_ids]
        
        # 读取力矩限制，用于底层饱和处理
        self._torque_min = base.torque_min
        self._torque_max = base.torque_max

        # ----- 3. 初始化扭矩缓冲区 -----
        # 预分配数组以避免在控制循环中频繁分配内存（实时性优化）
        self._arm_torques = np.zeros(base.ARM_DOF)
        self._hand_torques = np.zeros(base.HAND_DOF)

        # --- IK 相关初始化 ---
        # 末端执行器的 site 名称，如果你的 model 里叫别的名字请修改
        self.ee_site_name = "right_hand_site" 
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        
        # 用于 IK 计算的临时变量 (预分配内存)
        self.jac_p = np.zeros((3, model.nv)) # 位置雅可比
        self.jac_r = np.zeros((3, model.nv)) # 旋转雅可比

    # -----------------------------
    # 新增：末端位置控制接口 (IK)
    # -----------------------------
    def set_ee_target(self, data, ee_pos_target, ee_quat_target=None, hand_target=None):
        """
        通过逆运动学 (IK) 控制机械臂末端位置，并协同控制手部姿态。
        
        该方法采用基于雅可比转置的数值方法求解逆运动学。
        流程：计算误差 -> 获取雅可比 -> 求解关节速度/增量 -> 更新目标 -> 调用 PD 下发
        
        Args:
            data: MuJoCo 数据对象 (mjData)。
            ee_pos_target: 3D 目标位置 [x, y, z]。
            ee_quat_target: 目标姿态四元数 [w, x, y, z] (可选)。None 表示仅位置控制。
            hand_target: 手部关节目标角度 (直接传递给 PD 控制器)。
        """
        # 1. 获取当前末端状态
        ee_pos_current = data.site_xpos[self.ee_id]
        ee_rot_current = data.site_xmat[self.ee_id].reshape(3, 3)

        # 2. 计算位置误差
        error_pos = ee_pos_target - ee_pos_current
        
        # 3. 计算旋转误差 (如果提供了四元数)
        error_rot = np.zeros(3)
        if ee_quat_target is not None:
            # 获取当前末端 site 的四元数 (MuJoCo 存储的是旋转矩阵，先转四元数)
            ee_quat_current = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat_current, data.site_xmat[self.ee_id])
            
            # 计算四元数误差 (Target * Current_inv)
            # MuJoCo 提供 mju_subQuat 得到旋转轴/速度向量
            neg_ee_quat_current = np.zeros(4)
            mujoco.mju_negQuat(neg_ee_quat_current, ee_quat_current)
            
            error_quat = np.zeros(4)
            mujoco.mju_mulQuat(error_quat, ee_quat_target, neg_ee_quat_current)
            
            # 将误差四元数转换为旋转向量 (3维)
            # 这是误差的方向和大小
            mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        # 4. 获取雅可比矩阵
        # 计算末端 Site 相对于全局坐标系的几何雅可比
        mujoco.mj_jacSite(self.model, data, self.jac_p, self.jac_r, self.ee_id)
        
        # 只提取机械臂对应的 Dof 雅可比列 (假设机械臂对应前几个 dof)
        arm_jac_p = self.jac_p[:, self.arm_qvel_ids]
        arm_jac_r = self.jac_r[:, self.arm_qvel_ids]
        
        # 拼接雅可比 (如果只要位置就只用 jac_p)
        if ee_quat_target is not None:
            full_jac = np.vstack([arm_jac_p, arm_jac_r])
            full_error = np.concatenate([error_pos, error_rot])
        else:
            full_jac = arm_jac_p
            full_error = error_pos

        # 5. 计算关节增量 (自适应 DLS)
    
        # ⭐ 自适应阻尼：误差大时阻尼大（防抖动），误差小时阻尼小（快收敛）
        error_norm = np.linalg.norm(full_error)
        lambda_sq = np.clip(0.001 * error_norm, 1e-6, 0.01)  # 随误差线性缩放
        
        dq = full_jac.T @ np.linalg.solve(
            full_jac @ full_jac.T + lambda_sq * np.eye(full_jac.shape[0]), 
            full_error
        )

        # ⭐ 自适应步长：误差小时步长也按比例缩小，不做硬截断
        max_dq_base = 0.1745
        # 误差大于阈值时正常截断，小于阈值时线性缩放（跟踪误差）
        error_threshold = 0.05  # 5cm / ~3deg 以内开始线性缩放
        
        magnitude = np.linalg.norm(dq)
        if magnitude > 0:
            # 步长上限随误差线性降低，避免小误差时过冲
            adaptive_max_dq = max_dq_base * min(1.0, error_norm / error_threshold)
            adaptive_max_dq = max(adaptive_max_dq, 1e-4)  # 保底，避免完全停住
            
            if magnitude > adaptive_max_dq:
                dq = dq * (adaptive_max_dq / magnitude)

        # 6. 计算新的目标关节角
        # 注意：这里我们基于当前实际位置 data.qpos 计算下一个目标点
        arm_target = data.qpos[self.arm_qpos_ids] + dq
        
        # 7. 调用原有的 set_target 进行 PD 控制和下发
        if hand_target is None:
            hand_target = data.qpos[self.hand_qpos_ids] # 保持当前手部姿态
            
        self.set_target(data, arm_target, hand_target)

    # -----------------------------
    # 原有：关节空间主控制
    # -----------------------------
    def set_target(self, data, arm_target, hand_target):
        """
        关节空间 PD 控制器。
        
        执行标准的比例-微分控制逻辑，并应用力矩饱和限制。
        
        Args:
            data: MuJoCo 数据对象。
            arm_target: 机械臂目标关节角度。
            hand_target: 机械手目标关节角度。
        """
        # --- 1. 范围限制 ---
        # 确保目标值在物理关节限位之内，防止非法指令导致仿真崩溃
        arm_target = np.clip(arm_target, self.arm_range[:, 0], self.arm_range[:, 1])
        hand_target = np.clip(hand_target, self.hand_range[:, 0], self.hand_range[:, 1])

        # --- 2. 机械臂 PD 计算 ---
        # torque = Kp * (q_target - q_current) - Kd * qvel
        np.subtract(arm_target, data.qpos[self.arm_qpos_ids], out=self._arm_torques)
        self._arm_torques *= self.gains.kp_arm
        self._arm_torques -= (self.gains.kd_arm * data.qvel[self.arm_qvel_ids])

        # --- 3. 机械手 PD 计算 ---
        np.subtract(hand_target, data.qpos[self.hand_qpos_ids], out=self._hand_torques)
        self._hand_torques *= self.gains.kp_hand
        self._hand_torques -= (self.gains.kd_hand * data.qvel[self.hand_qvel_ids])

        # --- 4. 应用限制并下发 ---
        self._apply_saturation()
        self.base.apply_control(data, self._arm_torques, self._hand_torques)

    def _apply_saturation(self):
        """
        力矩饱和限制 (In-place 修改)。
        
        将计算出的力矩限制在执行器的物理能力范围内。
        """
        np.clip(self._arm_torques, self._torque_min[:self.base.ARM_DOF], 
                self._torque_max[:self.base.ARM_DOF], out=self._arm_torques)
        np.clip(self._hand_torques, self._torque_min[self.base.ARM_DOF:], 
                self._torque_max[self.base.ARM_DOF:], out=self._hand_torques)

    def _resolve_joint_ids(self, actuator_names):
        qpos_ids, qvel_ids, joint_ids = [], [], []
        for name in actuator_names:
            act_id   = self.base.actuator_map[name]
            joint_id = self.model.actuator_trnid[act_id, 0]
            joint_ids.append(joint_id)
            qpos_ids.append(self.model.jnt_qposadr[joint_id])
            qvel_ids.append(self.model.jnt_dofadr[joint_id])
        return (
            np.array(qpos_ids,  dtype=np.int32),
            np.array(qvel_ids,  dtype=np.int32),
            np.array(joint_ids, dtype=np.int32),
        )