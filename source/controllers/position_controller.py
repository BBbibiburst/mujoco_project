"""
位置控制器模块（基类 + OSC/IK 子类实现）.

架构：
- BasePositionController: 抽象基类，封装关节索引解析、手部 PD、工具方法等公共逻辑。
- OSCController: 基于操作空间动力学的力矩控制器（OSC）。
- IKController: 基于数值 IK + 关节 PD 的位置控制器。

统一接口：
- set_ee_target(data, pos, quat, hand)  → 末端笛卡尔空间控制，同时更新前馈历史
- set_joint_target(data, arm, hand)     → 关节空间控制
- hold(data)                            → 保持当前目标，重算力矩，不更新前馈历史
- reset(data)                           → 重置所有状态
- get_ee_pose(data)                     → 获取当前末端位姿
- get_joint_state(data)                 → 获取当前关节角和速度

RL 使用模式（推荐）：
    # 构造时不需要指定 RL 决策频率，自动测量
    ctrl = OSCController(base, model)

    # RL 步（任意频率）：更新目标，同时做一步仿真控制
    ctrl.set_ee_target(data, pos, quat)

    # 仿真步：保持目标，重算力矩
    for _ in range(sim_steps_per_policy_step):
        ctrl.hold(data)
        mujoco.mj_step(model, data)

设计要点：
- hold 与 set_* 严格分离：hold 不触碰前馈历史，避免速度估计污染。
- 速度前馈 dt 使用自动测量的真实策略周期，而非固定配置值，避免配置错误导致速度估计失真。
- IKController 的关节 PD 包含重力补偿（qfrc_bias），与 OSCController 行为一致。
- 预设增益通过工厂函数创建，避免模块级可变 ndarray 共享。

依赖：
    numpy, mujoco, dataclasses, time

使用示例：
    from source.controllers.position_controller import OSCController, stable_osc_gains
    from source.controllers.hand_arm_controller import HandArmController
    # 初始化
    base = HandArmController(model)
    ctrl = OSCController(base, model)
    # RL 步
    ctrl.set_ee_target(data, pos_target, quat_target, hand_target)
    # 仿真步
    for _ in range(sim_steps_per_policy_step):
        ctrl.hold(data)
        mujoco.mj_step(model, data)
"""

import numpy as np
import mujoco
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum, auto


# ====================== 枚举 ======================

class _CtrlMode(Enum):
    """
    控制模式追踪.
    用于检测切换时清除历史状态（速度前馈、力矩历史），避免模式切换跳变。
    """
    NONE = auto()
    JOINT_PD = auto()
    EE = auto()


# ====================== 增益配置 ======================

@dataclass
class OSCGains:
    """
    OSC 控制器增益配置.

    Attributes:
        kp_pos: 位置比例增益（笛卡尔刚度）[N/m]。
        kd_pos: 位置微分增益（笛卡尔阻尼）[N·s/m]。临界阻尼参考：2*sqrt(kp_pos)。
        kp_rot: 姿态比例增益 [N·m/rad]。
        kd_rot: 姿态微分增益 [N·m·s/rad]。
        kp_joint: 关节空间比例增益（仅 set_joint_target 使用）[N·m/rad]。
        kd_joint: 关节空间微分增益 [N·m·s/rad]。
        kp_hand: 手部关节比例增益（数组，支持各手指不同增益）。
        kd_hand: 手部关节微分增益。
        ff_scale: 速度前馈缩放系数 [0~1]。1.0=完全前馈，0.0=纯 PD。
        vel_filter_alpha: 目标速度低通滤波系数 [0~1]。越小越平滑，越大响应越快。
        singular_thresh: SVD 截断阈值，低于此值的奇异方向被忽略。建议 [0.01, 0.1]。
        null_kp: 零空间姿态恢复比例增益。0=禁用零空间控制。
        null_kd: 零空间阻尼增益，None 时自动计算为 2*sqrt(null_kp)。
        torque_rate_limit: 关节力矩变化率限制 [N·m/step]。
    """
    kp_pos: float = 400.0
    kd_pos: float = 40.0
    kp_rot: float = 100.0
    kd_rot: float = 20.0

    kp_joint: float = 40000.0
    kd_joint: float = 400.0

    kp_hand: np.ndarray = field(default_factory=lambda: np.full(6, 400000.0))
    kd_hand: np.ndarray = field(default_factory=lambda: np.full(6, 400.0))

    ff_scale: float = 0.9
    vel_filter_alpha: float = 0.5
    singular_thresh: float = 0.01

    null_kp: float = 10.0
    null_kd: Optional[float] = None
    torque_rate_limit: float = 50.0

    def __post_init__(self):
        if self.null_kd is None:
            self.null_kd = 2.0 * np.sqrt(self.null_kp)


@dataclass
class PDGains:
    """
    IK 控制器增益配置.

    Attributes:
        kp_arm: 机械臂关节比例增益 [N·m/rad]。
        kd_arm: 机械臂关节微分增益 [N·m·s/rad]。
        kp_hand: 手部关节比例增益。
        kd_hand: 手部关节微分增益。
    """
    kp_arm: np.ndarray = field(default_factory=lambda: np.full(7, 40000.0))
    kd_arm: np.ndarray = field(default_factory=lambda: np.full(7, 400.0))
    kp_hand: np.ndarray = field(default_factory=lambda: np.full(6, 40000.0))
    kd_hand: np.ndarray = field(default_factory=lambda: np.full(6, 400.0))


# ====================== 预设 OSC 增益（工厂函数） ======================

def stable_osc_gains() -> OSCGains:
    """稳定增益：保守的阻尼和力矩变化率，适合一般操作任务。"""
    return OSCGains(
        kp_pos=500, kd_pos=35,
        kp_rot=120, kd_rot=16,
        kp_joint=50000.0, kd_joint=450.0,
        ff_scale=0.95, vel_filter_alpha=0.4,
        singular_thresh=0.02,
        null_kp=3, null_kd=3.5,
        torque_rate_limit=30.0,
    )


def fast_tracking_osc_gains() -> OSCGains:
    """快速跟踪增益：高刚度+完全前馈，适合平滑轨迹跟踪。"""
    return OSCGains(
        kp_pos=600, kd_pos=30,
        kp_rot=150, kd_rot=15,
        kp_joint=60000.0, kd_joint=400.0,
        ff_scale=1.0, vel_filter_alpha=0.3,
        singular_thresh=0.01,
        null_kp=5, null_kd=4.5,
        torque_rate_limit=80.0,
    )


def fast_point_to_point_osc_gains() -> OSCGains:
    """点到点增益：高刚度+纯 PD（无前馈），适合目标跳变场景。"""
    return OSCGains(
        kp_pos=800, kd_pos=28,
        kp_rot=200, kd_rot=20,
        kp_joint=80000.0, kd_joint=350.0,
        ff_scale=0.0, vel_filter_alpha=0.5,
        singular_thresh=0.05,
        null_kp=0, null_kd=0.0,
        torque_rate_limit=100.0,
    )


# ====================== 基类 ======================

class BasePositionController(ABC):
    """
    位置控制器抽象基类.

    封装所有子类共用的基础设施：
    - 关节索引解析（qpos / qvel / joint IDs）
    - 关节范围与力矩限制
    - 手部 PD 控制
    - 末端位姿与关节状态读取

    子类必须实现：
        set_ee_target, set_joint_target, hold, reset

    Attributes:
        base: 硬件/模型抽象接口（提供 arm_names, hand_names, apply_control 等）。
        model: MuJoCo 模型对象。
        arm_qpos_ids: 机械臂 qpos 索引 (ARM_DOF,)。
        arm_qvel_ids: 机械臂 qvel 索引 (ARM_DOF,)。
        arm_joint_ids: 机械臂 joint 索引 (ARM_DOF,)。
        arm_range: 机械臂关节限位 (ARM_DOF, 2)，[lower, upper]。
        hand_qpos_ids / hand_qvel_ids / hand_joint_ids / hand_range: 同上，针对手部。
        ee_id: 末端 Site 的 MuJoCo ID。
        jac_p / jac_r: 雅可比缓冲区（预分配，nv 列）。
        _arm_torques / _hand_torques: 力矩缓冲区（预分配）。
        _arm_target / _hand_target: 当前目标缓存。
    """

    def __init__(
        self,
        base,
        model: mujoco.MjModel,
        ee_site_name: str = "right_hand_site",
    ):
        self.base = base
        self.model = model

        # 关节索引
        self.arm_qpos_ids, self.arm_qvel_ids, self.arm_joint_ids = \
            self._resolve_joint_ids(base.arm_names)
        self.hand_qpos_ids, self.hand_qvel_ids, self.hand_joint_ids = \
            self._resolve_joint_ids(
                [base.hand_names[k] for k in base.hand_key_order if k in base.hand_names]
            )

        # 关节范围
        self.arm_range = model.jnt_range[self.arm_joint_ids]
        self.hand_range = model.jnt_range[self.hand_joint_ids]

        # 力矩限制
        self._torque_min = base.torque_min
        self._torque_max = base.torque_max

        # 预分配缓冲区（避免控制循环中频繁分配）
        nv = model.nv
        self.jac_p = np.zeros((3, nv))
        self.jac_r = np.zeros((3, nv))
        self._arm_torques = np.zeros(base.ARM_DOF)
        self._hand_torques = np.zeros(base.HAND_DOF)

        # 末端 Site ID
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_id == -1:
            raise ValueError(f"Site '{ee_site_name}' not found in MuJoCo model.")
        self.ee_site_name = ee_site_name

        # 目标缓存（首次调用 set_* 时从 data.qpos 懒初始化）
        self._arm_target: Optional[np.ndarray] = None
        self._hand_target: Optional[np.ndarray] = None

    # ====================== 抽象接口 ======================

    @abstractmethod
    def set_ee_target(
        self,
        data: mujoco.MjData,
        ee_pos_target: np.ndarray,
        ee_quat_target: Optional[np.ndarray] = None,
        hand_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        设置末端目标并计算控制力矩（RL 步调用，自动测量真实频率）.

        更新目标缓存和前馈速度历史，计算并应用力矩。
        真实控制周期通过 wall-clock 自动测量，无需手动配置。

        Args:
            data: MuJoCo 数据对象。
            ee_pos_target: 目标位置 [3,]，单位米。
            ee_quat_target: 目标姿态四元数 [w,x,y,z] [4,]，可选（None=仅位置）。
            hand_target: 手部目标关节角 [HAND_DOF,]，可选（None=保持缓存）。

        Returns:
            (arm_torques, hand_torques): 实际应用的力矩 [N·m]。
        """

    @abstractmethod
    def set_joint_target(
        self,
        data: mujoco.MjData,
        arm_target: np.ndarray,
        hand_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        设置关节目标并计算控制力矩（RL 步调用）.

        Args:
            data: MuJoCo 数据对象。
            arm_target: 机械臂目标关节角度 [ARM_DOF,]。
            hand_target: 手部目标关节角 [HAND_DOF,]，可选。

        Returns:
            (arm_torques, hand_torques): 实际应用的力矩 [N·m]。
        """

    @abstractmethod
    def hold(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """
        保持当前缓存目标，重新计算并应用力矩（仿真步调用）.

        与 set_* 的核心区别：
        - 不更新目标缓存
        - 不更新前馈速度历史（避免污染下次 set_ee_target 的速度估计）
        - 基于当前仿真状态重新计算力矩（动力学每步都在变，必须重算）

        Args:
            data: MuJoCo 数据对象。

        Returns:
            (arm_torques, hand_torques): 实际应用的力矩 [N·m]。
        """

    @abstractmethod
    def reset(self, data: mujoco.MjData) -> None:
        """
        重置所有控制器状态.

        在 episode 结束、轨迹切换或急停时调用。
        子类至少应重置：目标缓存、历史速度/力矩状态。

        Args:
            data: MuJoCo 数据对象。
        """

    # ====================== 公共工具方法 ======================

    def get_ee_pose(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前末端位姿.

        Returns:
            (pos [3,], quat [w,x,y,z] [4,])
        """
        pos = data.site_xpos[self.ee_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self.ee_id])
        return pos, quat

    def get_joint_state(
        self, data: mujoco.MjData
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取机械臂当前关节角和角速度.

        封装 qpos/qvel 索引，避免外部直接操作内部索引数组。

        Returns:
            (qpos [ARM_DOF,], qvel [ARM_DOF,])
        """
        return (
            data.qpos[self.arm_qpos_ids].copy(),
            data.qvel[self.arm_qvel_ids].copy(),
        )

    # ====================== 手部控制（共享实现） ======================

    def _update_hand(
        self,
        data: mujoco.MjData,
        hand_target: Optional[np.ndarray],
        kp_hand: np.ndarray,
        kd_hand: np.ndarray,
    ) -> None:
        """
        手部关节 PD 控制（内部共享实现）.

        手部不纳入末端任务空间控制，始终独立执行关节 PD。
        首次调用时以当前关节角懒初始化目标；hand_target 为 None 时沿用缓存目标。

        Args:
            data: MuJoCo 数据。
            hand_target: 手部目标，None 时保持缓存。
            kp_hand: 手部比例增益数组。
            kd_hand: 手部微分增益数组。
        """
        if self._hand_target is None:
            self._hand_target = data.qpos[self.hand_qpos_ids].copy()

        if hand_target is not None:
            self._hand_target = np.clip(
                hand_target, self.hand_range[:, 0], self.hand_range[:, 1]
            )

        e_q = self._hand_target - data.qpos[self.hand_qpos_ids]
        e_qd = -data.qvel[self.hand_qvel_ids]
        tau = kp_hand * e_q + kd_hand * e_qd

        self._hand_torques[:] = np.clip(
            tau,
            self._torque_min[self.base.ARM_DOF:],
            self._torque_max[self.base.ARM_DOF:],
        )

    # ====================== 私有工具方法 ======================

    def _resolve_joint_ids(
        self, actuator_names
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将执行器名称列表解析为 qpos / qvel / joint 三组 MuJoCo 内部索引."""
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


# ====================== OSC 子类 ======================

class OSCController(BasePositionController):
    """
    操作空间控制器（OSC）.

    自动测量真实 RL 控制周期，用于速度前馈计算。
    无需手动配置 policy_hz，自适应 20Hz/30Hz/50Hz 及异步推理场景。
    """

    def __init__(
        self,
        base,
        model: mujoco.MjModel,
        gains: Optional[OSCGains] = None,
        ee_site_name: str = "right_hand_site",
        null_qpos_ref: Optional[np.ndarray] = None,
    ):
        super().__init__(base, model, ee_site_name)

        self.gains = gains if gains is not None else OSCGains()

        # 初始策略周期（仅用于第一次调用前，会被自动测量覆盖）
        self.policy_dt: float = 0.05  # 默认 20Hz 初值

        # 完整惯量矩阵缓存
        self._M_full = np.zeros((model.nv, model.nv))

        # 零空间参考
        self._null_qpos_ref = (
            null_qpos_ref.copy() if null_qpos_ref is not None else None
        )

        # 前馈速度历史
        self._prev_pos_target = None
        self._prev_quat_target = None
        self._vel_ff_pos = np.zeros(3)
        self._vel_ff_rot = np.zeros(3)

        # EE hold 缓存
        self._cached_ee_pos_target = None
        self._cached_ee_quat_target = None

        # 力矩变化率限制历史
        self._prev_tau = None

        # 当前控制模式
        self._ctrl_mode = _CtrlMode.NONE

        # 策略周期自动测量
        self._last_policy_time: Optional[float] = None

    # ============================================================
    # 策略周期自动测量
    # ============================================================

    def _update_policy_dt(self) -> None:
        """
        自动测量真实 RL 控制周期.

        使用 wall-clock 时间估计两次 set_ee_target 间隔，
        并进行低通滤波，避免抖动。
        过滤异常值（暂停/断点/推理延迟尖峰），确保测量鲁棒。
        """
        now = time.perf_counter()

        if self._last_policy_time is not None:
            dt_measured = now - self._last_policy_time

            # 过滤异常值（防止暂停/断点/推理延迟尖峰污染）
            if 1e-4 < dt_measured < 1.0:
                alpha = 0.2
                self.policy_dt = (
                    alpha * dt_measured
                    + (1.0 - alpha) * self.policy_dt
                )

        self._last_policy_time = now

    # ============================================================
    # 公共接口
    # ============================================================

    def set_ee_target(
        self,
        data: mujoco.MjData,
        ee_pos_target: np.ndarray,
        ee_quat_target: Optional[np.ndarray] = None,
        hand_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """RL 频率调用，自动测量真实周期."""

        if self._ctrl_mode != _CtrlMode.EE:
            self._prev_tau = None
            self._ctrl_mode = _CtrlMode.EE

        # 自动测量真实策略周期
        self._update_policy_dt()

        if self._null_qpos_ref is None:
            self._null_qpos_ref = data.qpos[self.arm_qpos_ids].copy()

        g = self.gains

        # 更新速度前馈（使用自动测量的 policy_dt）
        self._update_velocity_feedforward(
            ee_pos_target,
            ee_quat_target,
            g,
        )

        # 缓存 EE 目标
        self._cached_ee_pos_target = ee_pos_target.copy()
        self._cached_ee_quat_target = (
            None if ee_quat_target is None else ee_quat_target.copy()
        )

        # 更新手部目标
        if hand_target is not None:
            self._hand_target = np.clip(
                hand_target,
                self.hand_range[:, 0],
                self.hand_range[:, 1],
            )

        return self._compute_and_apply_osc(
            data,
            ee_pos_target,
            ee_quat_target,
            g,
        )

    def set_joint_target(
        self,
        data: mujoco.MjData,
        arm_target: np.ndarray,
        hand_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """关节空间 PD."""

        if self._ctrl_mode != _CtrlMode.JOINT_PD:
            self._prev_tau = None
            self._ctrl_mode = _CtrlMode.JOINT_PD

        g = self.gains

        self._arm_target = np.clip(
            arm_target,
            self.arm_range[:, 0],
            self.arm_range[:, 1],
        )

        e_q = self._arm_target - data.qpos[self.arm_qpos_ids]
        e_qd = -data.qvel[self.arm_qvel_ids]

        tau_arm = (
            g.kp_joint * e_q
            + g.kd_joint * e_qd
            + data.qfrc_bias[self.arm_qvel_ids]
        )

        tau_arm = self._apply_torque_limits(tau_arm, g)
        self._arm_torques[:] = tau_arm

        self._update_hand(
            data,
            hand_target,
            g.kp_hand,
            g.kd_hand,
        )

        self.base.apply_control(
            data,
            self._arm_torques,
            self._hand_torques,
        )

        return (
            self._arm_torques.copy(),
            self._hand_torques.copy(),
        )

    def hold(
        self,
        data: mujoco.MjData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """仿真频率调用."""

        if self._ctrl_mode == _CtrlMode.EE:
            if self._cached_ee_pos_target is None:
                if self._arm_target is None:
                    self._arm_target = data.qpos[self.arm_qpos_ids].copy()
                return self._hold_joint_pd(data)
            return self._hold_osc(data)

        if self._arm_target is None:
            self._arm_target = data.qpos[self.arm_qpos_ids].copy()

        return self._hold_joint_pd(data)

    def reset(self, data: mujoco.MjData) -> None:
        """重置控制器."""

        self._prev_pos_target = None
        self._prev_quat_target = None

        self._vel_ff_pos.fill(0.0)
        self._vel_ff_rot.fill(0.0)

        self._cached_ee_pos_target = None
        self._cached_ee_quat_target = None

        self._null_qpos_ref = data.qpos[self.arm_qpos_ids].copy()
        self._arm_target = data.qpos[self.arm_qpos_ids].copy()
        self._hand_target = data.qpos[self.hand_qpos_ids].copy()

        self._prev_tau = None
        self._ctrl_mode = _CtrlMode.NONE

        # 清空策略周期测量状态
        self._last_policy_time = None
        self.policy_dt = 0.05  # 恢复默认初值

    # ============================================================
    # Hold
    # ============================================================

    def _hold_osc(
        self,
        data: mujoco.MjData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """EE 模式保持."""

        return self._compute_and_apply_osc(
            data,
            self._cached_ee_pos_target,
            self._cached_ee_quat_target,
            self.gains,
        )

    def _hold_joint_pd(
        self,
        data: mujoco.MjData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """关节模式保持."""

        g = self.gains

        e_q = self._arm_target - data.qpos[self.arm_qpos_ids]
        e_qd = -data.qvel[self.arm_qvel_ids]

        tau_arm = (
            g.kp_joint * e_q
            + g.kd_joint * e_qd
            + data.qfrc_bias[self.arm_qvel_ids]
        )

        tau_arm = self._apply_torque_limits(tau_arm, g)
        self._arm_torques[:] = tau_arm

        self._update_hand(
            data,
            None,
            g.kp_hand,
            g.kd_hand,
        )

        self.base.apply_control(
            data,
            self._arm_torques,
            self._hand_torques,
        )

        return (
            self._arm_torques.copy(),
            self._hand_torques.copy(),
        )

    # ============================================================
    # OSC 核心
    # ============================================================

    def _compute_and_apply_osc(
        self,
        data: mujoco.MjData,
        ee_pos_target: np.ndarray,
        ee_quat_target: Optional[np.ndarray],
        g: OSCGains,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """OSC 主入口."""

        mujoco.mj_jacSite(
            self.model,
            data,
            self.jac_p,
            self.jac_r,
            self.ee_id,
        )

        J_p = self.jac_p[:, self.arm_qvel_ids].copy()
        J_r = self.jac_r[:, self.arm_qvel_ids].copy()
        qd = data.qvel[self.arm_qvel_ids]

        xacc_des, xacc_rot = self._compute_task_acceleration(
            data.site_xpos[self.ee_id].copy(),
            ee_pos_target,
            J_p,
            J_r,
            qd,
            g,
            ee_quat_target,
            data,
        )

        tau_arm = self._compute_osc_torque(
            data,
            J_p,
            J_r,
            xacc_des,
            xacc_rot,
            g,
        )

        tau_arm = self._apply_torque_limits(tau_arm, g)
        self._arm_torques[:] = tau_arm

        self._update_hand(
            data,
            None,
            g.kp_hand,
            g.kd_hand,
        )

        self.base.apply_control(
            data,
            self._arm_torques,
            self._hand_torques,
        )

        return (
            self._arm_torques.copy(),
            self._hand_torques.copy(),
        )

    def _update_velocity_feedforward(
        self,
        ee_pos_target: np.ndarray,
        ee_quat_target: Optional[np.ndarray],
        g: OSCGains,
    ) -> None:
        """速度前馈估计（使用自动测量的真实 policy_dt）."""

        alpha = g.vel_filter_alpha
        dt = self.policy_dt

        # 平移速度
        if self._prev_pos_target is not None:
            raw_vel = (ee_pos_target - self._prev_pos_target) / dt
            self._vel_ff_pos = (
                alpha * raw_vel
                + (1.0 - alpha) * self._vel_ff_pos
            )
        else:
            self._vel_ff_pos.fill(0.0)

        # 角速度
        if (
            ee_quat_target is not None
            and self._prev_quat_target is not None
        ):
            q_prev_inv = np.zeros(4)
            mujoco.mju_negQuat(
                q_prev_inv,
                self._prev_quat_target,
            )

            dq = np.zeros(4)
            mujoco.mju_mulQuat(
                dq,
                ee_quat_target,
                q_prev_inv,
            )

            raw_w = np.zeros(3)
            mujoco.mju_quat2Vel(raw_w, dq, 1.0)
            raw_w /= dt

            self._vel_ff_rot = (
                alpha * raw_w
                + (1.0 - alpha) * self._vel_ff_rot
            )
        else:
            self._vel_ff_rot.fill(0.0)

        self._prev_pos_target = ee_pos_target.copy()
        self._prev_quat_target = (
            None if ee_quat_target is None
            else ee_quat_target.copy()
        )

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
        """计算任务空间期望加速度."""

        # 位置
        e_pos = ee_pos_target - ee_pos
        ee_vel = J_p @ qd

        e_vel = g.ff_scale * self._vel_ff_pos - ee_vel

        xacc_des = (
            g.kp_pos * e_pos
            + g.kd_pos * e_vel
        )

        # 姿态
        xacc_rot = np.zeros(3)

        if ee_quat_target is not None:
            quat_cur = np.zeros(4)
            mujoco.mju_mat2Quat(
                quat_cur,
                data.site_xmat[self.ee_id],
            )

            quat_des = ee_quat_target.copy()

            # 同半球
            if np.dot(quat_des, quat_cur) < 0.0:
                quat_des *= -1.0

            quat_cur_inv = np.zeros(4)
            mujoco.mju_negQuat(quat_cur_inv, quat_cur)

            dq = np.zeros(4)
            mujoco.mju_mulQuat(
                dq,
                quat_des,
                quat_cur_inv,
            )

            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, dq, 1.0)

            ee_w = J_r @ qd
            e_w = g.ff_scale * self._vel_ff_rot - ee_w

            xacc_rot = (
                g.kp_rot * e_rot
                + g.kd_rot * e_w
            )

        return xacc_des, xacc_rot

    def _compute_osc_torque(
        self,
        data: mujoco.MjData,
        J_p: np.ndarray,
        J_r: np.ndarray,
        xacc_des: np.ndarray,
        xacc_rot: np.ndarray,
        g: OSCGains,
    ) -> np.ndarray:
        """OSC 力矩."""

        arm_idx = self.arm_qvel_ids

        mujoco.mj_fullM(
            self.model,
            self._M_full,
            data.qM,
        )

        M = self._M_full[np.ix_(arm_idx, arm_idx)]

        if np.any(np.abs(xacc_rot) > 1e-12):
            J = np.vstack((J_p, J_r))
            xacc = np.concatenate((xacc_des, xacc_rot))
        else:
            J = J_p
            xacc = xacc_des

        Minv_JT = np.linalg.solve(M, J.T)

        Lambda = self._svd_pinv(
            J @ Minv_JT,
            g.singular_thresh,
        )

        Lambda = 0.5 * (Lambda + Lambda.T)

        bias = data.qfrc_bias[arm_idx]

        F_task = Lambda @ (
            xacc + J @ np.linalg.solve(M, bias)
        )

        tau_task = J.T @ F_task

        tau_null = self._compute_null_space_torque(
            data,
            J,
            M,
            Lambda,
            arm_idx,
            g,
        )

        J_bar = Minv_JT @ Lambda
        N = np.eye(len(arm_idx)) - J_bar @ J

        tau_bias_null = N.T @ bias

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
        """零空间恢复."""

        if (
            g.null_kp <= 0.0
            or self._null_qpos_ref is None
        ):
            return np.zeros(len(arm_idx))

        J_bar = np.linalg.solve(M, J.T) @ Lambda
        N = np.eye(len(arm_idx)) - J_bar @ J

        q = data.qpos[self.arm_qpos_ids]
        qd = data.qvel[arm_idx]

        tau_null_raw = (
            g.null_kp * (self._null_qpos_ref - q)
            - g.null_kd * qd
        )

        return N.T @ tau_null_raw

    def _apply_torque_limits(
        self,
        tau_raw: np.ndarray,
        g: OSCGains,
    ) -> np.ndarray:
        """变化率限制 + 饱和限制."""

        if self._prev_tau is not None:
            delta = np.clip(
                tau_raw - self._prev_tau,
                -g.torque_rate_limit,
                g.torque_rate_limit,
            )
            tau = self._prev_tau + delta
        else:
            tau = tau_raw.copy()

        tau = np.clip(
            tau,
            self._torque_min[: self.base.ARM_DOF],
            self._torque_max[: self.base.ARM_DOF],
        )

        self._prev_tau = tau.copy()
        return tau

    @staticmethod
    def _svd_pinv(
        A: np.ndarray,
        thresh: float,
    ) -> np.ndarray:
        """TSVD 伪逆."""

        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        s_inv = np.zeros_like(s)
        mask = s > thresh
        s_inv[mask] = 1.0 / s[mask]

        return (Vt.T * s_inv) @ U.T


# ====================== IK 子类 ======================

class IKController(BasePositionController):
    """
    IK + 关节 PD 混合控制器.

    两层架构：
    1. IK 层（DLS 数值求解）：将末端目标转换为关节目标增量，原位更新 _arm_target。
    2. PD 层（关节控制）：执行关节 PD + 重力补偿，跟踪 IK 输出的关节目标。

    适用于对动力学要求不高的点到点运动、初始化或作为 OSC 的降级备份。

    注意：IKController 无速度前馈，hold 直接复用缓存的关节目标重算 PD 力矩。
    在 20Hz/500Hz 的 RL 场景下，hold 期间 _arm_target 不变，
    PD 控制自然收敛到目标关节角。

    Attributes:
        gains: PD 增益配置。
        damping: DLS 阻尼系数（控制奇异附近的 dq 幅度）。
    """

    def __init__(
        self,
        base,
        model: mujoco.MjModel,
        gains: Optional[PDGains] = None,
        ee_site_name: str = "right_hand_site",
        damping: float = 0.05,
    ):
        """
        Args:
            base: 硬件抽象接口。
            model: MuJoCo 模型。
            gains: PD 增益配置，None 时使用默认值。
            ee_site_name: 末端 Site 名称。
            damping: DLS 阻尼系数。值越大奇异附近越保守，但精度略低。
        """
        super().__init__(base, model, ee_site_name)
        self.gains = gains if gains is not None else PDGains()
        self.damping = damping

    # ====================== 公共接口 ======================

    def set_ee_target(
        self,
        data: mujoco.MjData,
        ee_pos_target: np.ndarray,
        ee_quat_target: Optional[np.ndarray] = None,
        hand_target: Optional[np.ndarray] = None,
        max_steps: int = 100,
        tol: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        末端目标控制（IK + 关节 PD，RL 步调用）.

        IK 求解更新 _arm_target，然后执行关节 PD + 重力补偿。
        后续 hold 调用将直接使用更新后的 _arm_target，不再重新求解 IK。

        Args:
            data: MuJoCo 数据。
            ee_pos_target: 目标位置 [3,]。
            ee_quat_target: 目标姿态 [w,x,y,z] [4,]，可选。
            hand_target: 手部目标 [HAND_DOF,]，可选。
            max_steps: IK 最大迭代次数。
            tol: IK 收敛阈值（位置+姿态误差 L2 范数）。

        Returns:
            (arm_torques, hand_torques)
        """
        if self._arm_target is None:
            self._arm_target = data.qpos[self.arm_qpos_ids].copy()

        # IK 求解，原位更新 _arm_target
        self._solve_ik(data, ee_pos_target, ee_quat_target, self._arm_target, max_steps, tol)

        return self._joint_pd_and_apply(data, hand_target)

    def set_joint_target(
        self,
        data: mujoco.MjData,
        arm_target: np.ndarray,
        hand_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        直接关节目标控制（绕过 IK）.

        适用于已知关节角目标的任务（如归位、预设姿态）。包含重力补偿（qfrc_bias）。

        Args:
            data: MuJoCo 数据。
            arm_target: 目标关节角 [ARM_DOF,]。
            hand_target: 手部目标 [HAND_DOF,]，可选。

        Returns:
            (arm_torques, hand_torques)
        """
        self._arm_target = np.clip(arm_target, self.arm_range[:, 0], self.arm_range[:, 1])
        return self._joint_pd_and_apply(data, hand_target)

    def hold(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """
        保持当前缓存的关节目标，重新计算并应用力矩（仿真步调用）.

        IKController 无前馈历史，hold 与 set_joint_target 的唯一区别是
        不更新 _arm_target。在高频仿真步中持续施力，PD 控制自然收敛。
        若尚未设置目标，以当前关节角为目标（原地保持）。
        """
        if self._arm_target is None:
            self._arm_target = data.qpos[self.arm_qpos_ids].copy()
        return self._joint_pd_and_apply(data, hand_target=None)

    def reset(self, data: mujoco.MjData) -> None:
        """重置目标缓存（以当前关节角为初始目标）."""
        self._arm_target = data.qpos[self.arm_qpos_ids].copy()
        self._hand_target = data.qpos[self.hand_qpos_ids].copy()

    # ====================== 私有方法 ======================

    def _joint_pd_and_apply(
        self,
        data: mujoco.MjData,
        hand_target: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        关节 PD + 重力补偿 + 手部更新 + 应用（set_* 和 hold 共用）.

        包含 qfrc_bias 重力补偿，与 OSCController 的关节 PD 模式行为一致，
        确保低增益下也能抵抗重力，没有稳态误差。
        """
        g = self.gains
        e_q = self._arm_target - data.qpos[self.arm_qpos_ids]
        e_qd = -data.qvel[self.arm_qvel_ids]
        tau_arm = g.kp_arm * e_q + g.kd_arm * e_qd + data.qfrc_bias[self.arm_qvel_ids]

        self._arm_torques[:] = np.clip(
            tau_arm,
            self._torque_min[:self.base.ARM_DOF],
            self._torque_max[:self.base.ARM_DOF],
        )

        self._update_hand(data, hand_target, g.kp_hand, g.kd_hand)
        self.base.apply_control(data, self._arm_torques, self._hand_torques)
        return self._arm_torques.copy(), self._hand_torques.copy()

    def _solve_ik(
        self,
        data: mujoco.MjData,
        pos_target: np.ndarray,
        quat_target: Optional[np.ndarray],
        q_target: np.ndarray,
        max_steps: int,
        tol: float,
    ) -> None:
        """
        DLS（阻尼最小二乘）逆运动学求解器（原位更新 q_target）.

        dq = Jᵀ (J Jᵀ + λ² I)⁻¹ err
        λ 为阻尼系数，在奇异附近限制 dq 幅度，防止发散。
        收敛后（err_norm < tol）提前退出迭代。
        """
        for _ in range(max_steps):
            err_pos = pos_target - data.site_xpos[self.ee_id]
            err_norm = np.linalg.norm(err_pos)

            err_rot = np.zeros(3)
            if quat_target is not None:
                quat_cur = np.zeros(4)
                mujoco.mju_mat2Quat(quat_cur, data.site_xmat[self.ee_id])
                qt_align = quat_target.copy()
                if np.dot(qt_align, quat_cur) < 0:
                    qt_align = -qt_align
                neg_cur = np.zeros(4)
                mujoco.mju_negQuat(neg_cur, quat_cur)
                dq = np.zeros(4)
                mujoco.mju_mulQuat(dq, qt_align, neg_cur)
                mujoco.mju_quat2Vel(err_rot, dq, 1.0)
                err_norm += np.linalg.norm(err_rot)

            if err_norm < tol:
                break

            mujoco.mj_jacSite(self.model, data, self.jac_p, self.jac_r, self.ee_id)
            J_p = self.jac_p[:, self.arm_qvel_ids]
            J_r = self.jac_r[:, self.arm_qvel_ids]

            if quat_target is not None:
                J = np.vstack([J_p, J_r])
                err = np.concatenate([err_pos, err_rot])
            else:
                J = J_p
                err = err_pos

            A = J @ J.T + self.damping**2 * np.eye(J.shape[0])
            try:
                dq_joint = J.T @ np.linalg.solve(A, err)
            except np.linalg.LinAlgError:
                dq_joint = J.T @ np.linalg.pinv(A) @ err

            q_target[:] = np.clip(
                q_target + dq_joint,
                self.arm_range[:, 0],
                self.arm_range[:, 1],
            )