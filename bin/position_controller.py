"""
操作空间控制器（Operational Space Control, OSC）
=================================================

架构对比
---------
旧版（IK + PD，两层解耦）：
    error ──IK(DLS)──> Δq ──> q_target ──> PD ──> τ
    问题：IK 不知道机器人动力学，PD 不知道任务空间目标，两层之间存在结构性延迟。

新版（OSC，单层闭合）：
    error ──> F_task（笛卡尔力）──> τ = J^T Λ F_task + τ_comp
    优势：控制律直接在物理层面成立，跟踪误差仅由增益决定，与构型无关。

OSC 控制律推导
--------------
机器人动力学方程：
    M(q) q̈ + C(q,q̇) q̇ + g(q) = τ + J^T f_ext

目标：在任务空间施加期望加速度 ẍ_des，则关节力矩为：

    τ = J^T Λ (ẍ_des + μ) + τ_comp

其中：
    Λ     = (J M⁻¹ J^T)⁻¹        # 任务空间惯量矩阵（使力到加速度的映射与质量无关）
    μ     = -Λ J M⁻¹ (C q̇)       # 任务空间科里奥利/离心力补偿（可选）
    τ_comp = g(q) + C(q,q̇) q̇     # 关节空间动力学补偿（重力+科氏力）

期望末端加速度（PD + 速度前馈）：
    ẍ_des = ẍ_ff + Kp·(x_target - x) + Kd·(ẋ_target - ẋ)

MuJoCo 实现说明
---------------
- M(q)       : mj_fullM          → model.nv × model.nv 惯量矩阵
- C(q,q̇)q̇   : data.qfrc_bias    → 已包含重力+科氏力，直接用作 τ_comp
- J          : mj_jacSite        → 位置/旋转雅可比
- ẋ (末端速度): J @ data.qvel    → 由雅可比映射关节速度得到
- 奇异性处理  : SVD 截断伪逆（TSVD）替代 DLS，在奇异方向直接清零而非施加阻尼

手部控制
--------
手部自由度不参与 OSC（OSC 仅控制任务空间，手部维度不在其中），
保留独立的关节空间 PD 控制器处理手部，接口与旧版完全兼容。
"""

import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import Optional


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
        kp_hand:      机械手关节比例增益（关节空间 PD）[N·m/rad]。
        kd_hand:      机械手关节微分增益 [N·m·s/rad]。
        ff_scale:     速度前馈缩放系数 [0~1]。
                      1.0 = 完全前馈目标速度（低延迟轨迹跟踪）；
                      0.0 = 纯 PD（退化为经典 OSC）。
                      建议：平滑目标用 0.8~1.0，有噪声目标用 0.3~0.6。
        vel_filter_alpha: 目标速度低通滤波系数 [0~1]。
                          越小越平滑，越大响应越快。建议 0.4~0.7。
        singular_thresh:  SVD 截断阈值，低于此奇异值的方向被忽略。
                          建议 [0.01, 0.1]，值越大奇异处越保守。
        null_kp:      零空间姿态恢复增益（将冗余自由度拉向参考构型）。
                      0 = 不使用零空间控制。
        torque_min/max: 关节力矩饱和限制（从 base 读取，此处为占位）。
    """
    kp_pos:           float = 400.0
    kd_pos:           float = 40.0
    kp_rot:           float = 100.0
    kd_rot:           float = 20.0
    kp_hand:          np.ndarray = field(default_factory=lambda: np.full(6, 400000.))
    kd_hand:          np.ndarray = field(default_factory=lambda: np.full(6,   400.))
    ff_scale:         float = 0.9
    vel_filter_alpha: float = 0.5
    singular_thresh:  float = 0.01
    null_kp:          float = 10.0   # 零空间恢复增益；设为 0 可禁用

# ====================== 位置控制器实现 ======================
# 提供三套预设增益配置，适用于不同的控制需求：
# 1. Stable_OSCGains：适合点到点运动，增益较低，强调稳定性。
Stable_OSCGains = OSCGains(
    kp_pos=500,
    kd_pos=35,            # 轻微欠阻尼
    kp_rot=120,
    kd_rot=16,
    ff_scale=0.95,
    vel_filter_alpha=0.4,
    singular_thresh=0.02,
    null_kp=3,
)

# 2. FastTracking_OSCGains：适合平滑轨迹跟踪，增益较高，强调响应速度。
FastTracking_OSCGains = OSCGains(
    kp_pos=600,           # 高刚度
    kd_pos=30,            # 欠阻尼（1.2*sqrt(600)≈29）
    kp_rot=150,
    kd_rot=15,
    ff_scale=1.0,         # 完全前馈
    vel_filter_alpha=0.3, # 快速响应，依赖平滑输入
    singular_thresh=0.01, # 较激进
    null_kp=5,            # 弱零空间约束
)

# 3. FastPointToPoint_OSCGains：适合快速点到点运动，增益更高，前馈更弱以避免跳变时过度振荡。
FastPointToPoint_OSCGains = OSCGains(
    kp_pos=800,
    kd_pos=28,            # 明显欠阻尼 ~sqrt(800)
    kp_rot=200,
    kd_rot=20,
    ff_scale=0.0,         # 纯 PD，无前馈（目标跳变时更稳定）
    vel_filter_alpha=0.5,
    singular_thresh=0.05,
    null_kp=0,            # 禁用零空间，全自由度用于速度
)

class PositionController:
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
        self.base  = base
        self.model = model
        self.gains = gains if gains is not None else OSCGains()

        # ----- 1. 关节索引解析 -----
        self.arm_qpos_ids,  self.arm_qvel_ids,  self.arm_joint_ids  = \
            self._resolve_joint_ids(base.arm_names)
        self.hand_qpos_ids, self.hand_qvel_ids, self.hand_joint_ids = \
            self._resolve_joint_ids(
                [base.hand_names[k] for k in base.hand_key_order if k in base.hand_names]
            )

        # ----- 2. 关节范围 -----
        self.arm_range  = model.jnt_range[self.arm_joint_ids]
        self.hand_range = model.jnt_range[self.hand_joint_ids]

        # ----- 3. 力矩限制 -----
        self._torque_min = base.torque_min
        self._torque_max = base.torque_max

        # ----- 4. 预分配缓冲区 -----
        nv = model.nv
        self.jac_p  = np.zeros((3, nv))
        self.jac_r  = np.zeros((3, nv))
        self._M_full = np.zeros((nv, nv))           # 完整惯量矩阵
        self._arm_torques  = np.zeros(base.ARM_DOF)
        self._hand_torques = np.zeros(base.HAND_DOF)

        # ----- 5. 末端 Site ID -----
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_id == -1:
            raise ValueError(f"Site '{ee_site_name}' not found in MuJoCo model.")
        self.ee_site_name = ee_site_name

        # ----- 6. 零空间参考构型 -----
        # 用于将冗余自由度（7DOF 臂的第7个）拉向一个"舒适"的构型，
        # 避免 OSC 在冗余方向漫游到关节极限
        self._null_qpos_ref: Optional[np.ndarray] = (
            null_qpos_ref.copy() if null_qpos_ref is not None else None
        )

        # ----- 7. 速度前馈状态 -----
        self._prev_pos_target:  Optional[np.ndarray] = None
        self._prev_quat_target: Optional[np.ndarray] = None
        self._vel_ff_pos = np.zeros(3)
        self._vel_ff_rot = np.zeros(3)

        # ----- 8. 手部目标缓存 -----
        self._hand_target: Optional[np.ndarray] = None

        # 兼容旧版 set_target 接口
        self._arm_target: Optional[np.ndarray] = None

    # =========================================================
    # 公共接口 1：末端位置/姿态控制（OSC 模式）
    # =========================================================

    def set_ee_target(
        self,
        data: mujoco.MjData,
        ee_pos_target: np.ndarray,
        ee_quat_target: Optional[np.ndarray] = None,
        hand_target: Optional[np.ndarray] = None,
    ):
        """
        操作空间控制主接口。

        每帧执行以下控制律：
            ① 估计目标速度（有限差分 + 低通滤波）→ 速度前馈 v_ff
            ② 计算末端位置/姿态误差
            ③ 合成期望末端加速度：ẍ_des = ẍ_ff + Kp·e_pos + Kd·(v_ff - v_ee)
            ④ 计算任务空间惯量矩阵：Λ = (J M⁻¹ J^T)⁻¹
            ⑤ 映射到关节力矩：τ_task = J^T Λ ẍ_des
            ⑥ 加入动力学补偿：τ = τ_task + τ_comp（重力+科氏力）
            ⑦ 零空间控制：τ += N^T τ_null（冗余自由度恢复参考构型）
            ⑧ 力矩饱和后下发

        Args:
            data:           MuJoCo 数据对象。
            ee_pos_target:  目标位置 [x, y, z]。
            ee_quat_target: 目标姿态四元数 [w, x, y, z]（可选）。
            hand_target:    手部目标关节角（可选，None 则保持缓存值）。
        """
        g  = self.gains
        dt = self.model.opt.timestep
        nv = self.model.nv

        # ── 初始化零空间参考构型（首帧） ─────────────────────────────────
        if self._null_qpos_ref is None:
            self._null_qpos_ref = data.qpos[self.arm_qpos_ids].copy()

        # ═══════════════════════════════════════════════════════════════
        # Step 1: 目标速度估计（有限差分 + 低通滤波）
        # ═══════════════════════════════════════════════════════════════
        # 目的：给 Kd 项提供"目标速度"参考，使阻尼项变为：
        #   Kd·(ẋ_target - ẋ_ee) 而非 Kd·(-ẋ_ee)
        # 后者在目标快速运动时会产生错误的阻力，导致跟踪滞后。
        alpha = g.vel_filter_alpha

        if self._prev_pos_target is not None:
            raw_vel = (ee_pos_target - self._prev_pos_target) / dt
            # 一阶低通：抑制离散差分的高频噪声
            self._vel_ff_pos = alpha * raw_vel + (1.0 - alpha) * self._vel_ff_pos
        else:
            self._vel_ff_pos[:] = 0.0

        if ee_quat_target is not None and self._prev_quat_target is not None:
            neg_prev = np.zeros(4)
            mujoco.mju_negQuat(neg_prev, self._prev_quat_target)
            dq_ff = np.zeros(4)
            mujoco.mju_mulQuat(dq_ff, ee_quat_target, neg_prev)
            raw_vel_rot = np.zeros(3)
            mujoco.mju_quat2Vel(raw_vel_rot, dq_ff, 1.0)
            raw_vel_rot /= dt
            self._vel_ff_rot = alpha * raw_vel_rot + (1.0 - alpha) * self._vel_ff_rot
        else:
            self._vel_ff_rot[:] = 0.0

        self._prev_pos_target  = ee_pos_target.copy()
        self._prev_quat_target = ee_quat_target.copy() if ee_quat_target is not None else None

        # ═══════════════════════════════════════════════════════════════
        # Step 2: 计算末端状态（实际位置、速度）
        # ═══════════════════════════════════════════════════════════════
        ee_pos_cur  = data.site_xpos[self.ee_id].copy()

        # 计算末端线速度：v_ee = J_p @ q̇
        # 注意：只取臂的 qvel 列，手部不参与 OSC
        mujoco.mj_jacSite(self.model, data, self.jac_p, self.jac_r, self.ee_id)
        J_p = self.jac_p[:, self.arm_qvel_ids].copy()   # (3, n_arm)
        J_r = self.jac_r[:, self.arm_qvel_ids].copy()   # (3, n_arm)
        arm_qvel = data.qvel[self.arm_qvel_ids]
        ee_vel_cur = J_p @ arm_qvel                      # 当前末端线速度 [m/s]

        # ═══════════════════════════════════════════════════════════════
        # Step 3: 位置误差 + 速度误差 → 期望末端加速度
        # ═══════════════════════════════════════════════════════════════
        # 位置误差（笛卡尔空间）
        e_pos = ee_pos_target - ee_pos_cur

        # 速度误差：(目标速度前馈 - 当前末端速度)
        # 前馈缩放系数 ff_scale 控制前馈强度：
        #   ff_scale=1 → 完全跟随目标速度，适合平滑轨迹
        #   ff_scale=0 → 纯阻尼（-Kd·v_ee），适合点到点运动
        e_vel_pos = g.ff_scale * self._vel_ff_pos - ee_vel_cur

        # 期望末端线加速度（PD 律，单位：m/s²）
        # ẍ_des = Kp·e_pos + Kd·e_vel
        xacc_des = g.kp_pos * e_pos + g.kd_pos * e_vel_pos

        # 姿态控制（可选）
        xacc_des_rot = np.zeros(3)
        if ee_quat_target is not None:
            # 姿态误差：delta_q → 旋转向量
            ee_quat_cur = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat_cur, data.site_xmat[self.ee_id])
            neg_cur = np.zeros(4)
            mujoco.mju_negQuat(neg_cur, ee_quat_cur)
            dq_err = np.zeros(4)
            mujoco.mju_mulQuat(dq_err, ee_quat_target, neg_cur)
            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, dq_err, 1.0)

            # 末端角速度
            ee_angvel_cur = J_r @ arm_qvel
            e_vel_rot = g.ff_scale * self._vel_ff_rot - ee_angvel_cur

            xacc_des_rot = g.kp_rot * e_rot + g.kd_rot * e_vel_rot

        # ═══════════════════════════════════════════════════════════════
        # Step 4: 任务空间惯量矩阵 Λ = (J M⁻¹ J^T)⁻¹
        # ═══════════════════════════════════════════════════════════════
        # Λ 的物理意义：将笛卡尔空间的"力"转换为笛卡尔空间的"加速度"
        # 引入 Λ 后，控制律对机器人的质量分布不敏感，增益物理意义固定
        #
        # 实现：
        #   1. 获取完整关节空间惯量矩阵 M（nv×nv）
        #   2. 只取臂对应的子块 M_arm（n_arm×n_arm）
        #   3. 计算 M_arm⁻¹，再组装 Λ
        mujoco.mj_fullM(self.model, self._M_full, data.qM)
        arm_idx = self.arm_qvel_ids
        M_arm = self._M_full[np.ix_(arm_idx, arm_idx)]   # 臂子块

        # M_arm⁻¹（对称正定矩阵，用 Cholesky 分解更稳定）
        try:
            M_inv = np.linalg.inv(M_arm)
        except np.linalg.LinAlgError:
            # 退化情况（极少发生）：使用伪逆
            M_inv = np.linalg.pinv(M_arm)

        if ee_quat_target is not None:
            # 6D 任务空间（位置 + 旋转）
            J_full = np.vstack([J_p, J_r])              # (6, n_arm)
            xacc_full = np.concatenate([xacc_des, xacc_des_rot])
        else:
            # 3D 任务空间（仅位置）
            J_full = J_p                                 # (3, n_arm)
            xacc_full = xacc_des

        # Λ = (J M⁻¹ J^T)⁻¹，使用 SVD 截断伪逆保证数值稳定性
        # SVD 截断（TSVD）比 DLS 更干净：直接在奇异方向清零而非施加阻尼
        JMinvJT = J_full @ M_inv @ J_full.T
        Lambda = self._svd_pinv(JMinvJT, thresh=self.gains.singular_thresh)

        # ═══════════════════════════════════════════════════════════════
        # Step 5: 任务空间力 → 关节力矩（主任务）
        # ═══════════════════════════════════════════════════════════════
        # F_task = Λ · ẍ_des（笛卡尔力）
        # τ_task = J^T · F_task
        F_task  = Lambda @ xacc_full
        tau_task_full = J_full.T @ F_task               # (n_arm,)

        # ═══════════════════════════════════════════════════════════════
        # Step 6: 动力学补偿（重力 + 科里奥利力）
        # ═══════════════════════════════════════════════════════════════
        # MuJoCo 的 data.qfrc_bias 已包含 C(q,q̇)q̇ + g(q)，
        # 直接提取臂对应分量作为前馈补偿
        tau_comp = data.qfrc_bias[arm_idx]

        # ═══════════════════════════════════════════════════════════════
        # Step 7: 零空间控制（冗余自由度管理）
        # ═══════════════════════════════════════════════════════════════
        # 对于 7DOF 机械臂 + 3D 任务，存在 4 个冗余自由度。
        # 零空间投影 N = I - J†J 将额外力矩投影到不影响末端运动的方向，
        # 用于将关节拉向参考构型（避免接近关节极限或奇异构型）。
        tau_null = np.zeros(len(arm_idx))
        if self.gains.null_kp > 0 and self._null_qpos_ref is not None:
            # 零空间投影矩阵：N = I - M⁻¹ J^T Λ J
            # 推导：τ_null 经过 N 投影后，对末端加速度贡献为零
            J_bar = M_inv @ J_full.T @ Lambda            # 动力学一致伪逆 J†
            N_mat = np.eye(len(arm_idx)) - J_bar @ J_full  # 零空间投影

            # 零空间力矩：拉向参考构型，加阻尼防止零空间振荡
            q_err_null  = self._null_qpos_ref - data.qpos[self.arm_qpos_ids]
            qd_null     = data.qvel[arm_idx]
            # 零空间阻尼取 kp 的 1/10（临界阻尼估计）
            tau_null_raw = self.gains.null_kp * q_err_null - (self.gains.null_kp * 0.1) * qd_null
            tau_null = N_mat.T @ tau_null_raw

        # ═══════════════════════════════════════════════════════════════
        # Step 8: 合并力矩、饱和、下发
        # ═══════════════════════════════════════════════════════════════
        tau_arm = tau_task_full + tau_comp + tau_null

        # 写入臂力矩缓冲并饱和
        self._arm_torques[:] = np.clip(
            tau_arm,
            self._torque_min[: self.base.ARM_DOF],
            self._torque_max[: self.base.ARM_DOF],
        )

        # 手部：独立关节 PD（手部不参与 OSC）
        self._update_hand(data, hand_target)

        self.base.apply_control(data, self._arm_torques, self._hand_torques)

    # =========================================================
    # 公共接口 2：关节空间控制（兼容旧版接口，退化为关节 PD）
    # =========================================================

    def set_target(
        self,
        data: mujoco.MjData,
        arm_target: np.ndarray,
        hand_target: Optional[np.ndarray] = None,
    ):
        """
        关节空间 PD 控制（兼容旧版接口）。

        注意：此方法绕过 OSC，直接在关节空间做 PD。
        适合初始化阶段将机械臂移动到起始位置，或纯关节空间任务。
        正常跟踪任务请使用 set_ee_target。

        Args:
            data:        MuJoCo 数据对象。
            arm_target:  机械臂目标关节角度。
            hand_target: 手部目标关节角（可选）。
        """
        if self._arm_target is None:
            self._arm_target = data.qpos[self.arm_qpos_ids].copy()

        self._arm_target = np.clip(
            arm_target, self.arm_range[:, 0], self.arm_range[:, 1]
        )

        # 简单关节 PD（复用 qfrc_bias 做重力补偿）
        e_q  = self._arm_target - data.qpos[self.arm_qpos_ids]
        e_qd = -data.qvel[self.arm_qvel_ids]
        # 取 OSC 增益的关节空间等效（kp_pos 量级做参考）
        kp_j = self.gains.kp_pos * 100.0   # 关节空间刚度（经验比例）
        kd_j = self.gains.kd_pos * 10.0
        tau_arm = kp_j * e_q + kd_j * e_qd + data.qfrc_bias[self.arm_qvel_ids]

        self._arm_torques[:] = np.clip(
            tau_arm,
            self._torque_min[: self.base.ARM_DOF],
            self._torque_max[: self.base.ARM_DOF],
        )

        self._update_hand(data, hand_target)
        self.base.apply_control(data, self._arm_torques, self._hand_torques)

    # =========================================================
    # 公共接口 3：重置状态
    # =========================================================

    def reset_targets(self, data: mujoco.MjData):
        """
        重置所有控制器状态。

        在轨迹切换、急停或重新初始化时调用。
        - 清空速度前馈历史（防止切换时速度跳变）
        - 将零空间参考构型更新为当前实际构型

        Args:
            data: MuJoCo 数据对象。
        """
        self._prev_pos_target  = None
        self._prev_quat_target = None
        self._vel_ff_pos[:]    = 0.0
        self._vel_ff_rot[:]    = 0.0
        self._hand_target      = data.qpos[self.hand_qpos_ids].copy()
        self._arm_target       = data.qpos[self.arm_qpos_ids].copy()
        # 更新零空间参考为当前构型，避免复位时出现零空间力矩突变
        self._null_qpos_ref    = data.qpos[self.arm_qpos_ids].copy()

    # =========================================================
    # 私有方法
    # =========================================================

    def _update_hand(self, data: mujoco.MjData, hand_target: Optional[np.ndarray]):
        """
        手部关节 PD 控制（内部调用）。

        手部自由度不纳入 OSC 任务空间，使用独立的关节 PD 控制。
        hand_target=None 时保持上次缓存目标（不跟随实际值，避免跟随滞后）。
        """
        if self._hand_target is None:
            self._hand_target = data.qpos[self.hand_qpos_ids].copy()
        if hand_target is not None:
            self._hand_target = np.clip(
                hand_target, self.hand_range[:, 0], self.hand_range[:, 1]
            )

        e_q  = self._hand_target - data.qpos[self.hand_qpos_ids]
        e_qd = data.qvel[self.hand_qvel_ids]
        tau  = self.gains.kp_hand * e_q - self.gains.kd_hand * e_qd

        self._hand_torques[:] = np.clip(
            tau,
            self._torque_min[self.base.ARM_DOF :],
            self._torque_max[self.base.ARM_DOF :],
        )

    @staticmethod
    def _svd_pinv(A: np.ndarray, thresh: float) -> np.ndarray:
        """
        SVD 截断伪逆（Truncated SVD Pseudo-inverse）。

        相较于 DLS（阻尼最小二乘），TSVD 的优势：
        - 奇异方向直接清零（物理意义明确：不在可控方向施力）
        - 非奇异方向不受阻尼影响（不牺牲正常方向的精度）
        - 阈值 thresh 含义直观：低于此奇异值的方向被视为奇异

        Args:
            A:      待求伪逆的方阵（通常为 J M⁻¹ J^T）。
            thresh: 奇异值截断阈值。

        Returns:
            A 的截断伪逆。
        """
        U, s, Vt = np.linalg.svd(A)
        # 只保留奇异值大于阈值的方向
        s_inv = np.where(s > thresh, 1.0 / s, 0.0)
        return (Vt.T * s_inv) @ U.T

    def _resolve_joint_ids(self, actuator_names):
        """
        将执行器名称解析为 MuJoCo 内部的 qpos、qvel、joint 三组索引。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (qpos_ids, qvel_ids, joint_ids)
        """
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