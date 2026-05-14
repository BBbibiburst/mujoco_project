"""
环境配置数据类.

将原 RobotConfig 按职责拆分为三个独立数据类，
再由 RobotConfig 聚合，保持向后兼容。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from source.controllers.position_controller import OSCGains, PDGains
from .config import DefaultTextures


# ====================== 仿真参数 ======================

@dataclass
class SimConfig:
    """纯物理仿真参数."""
    control_freq: float = 20.0      # 策略控制频率 [Hz]
    sim_freq: float = 1000.0        # 物理仿真频率 [Hz]
    max_episode_steps: int = 500    # 单回合最大步数

    @property
    def n_sim_steps_per_control(self) -> int:
        """每个控制步对应的仿真步数."""
        return max(1, int(self.sim_freq / self.control_freq))


# ====================== 动作/控制器参数 ======================

@dataclass
class ActionConfig:
    """动作空间与底层控制器参数."""

    # 动作表示
    # "joint" : 7DOF 臂关节增量 + 6DOF 手部增量 = 13 维
    # "ee"    : 末端位姿增量（位置3D + 姿态3D）+ 6DOF 手部增量 = 12 维
    action_mode: str = "joint"

    # 底层控制器
    # "osc" : 操作空间控制（推荐，平滑连续）
    # "ik"  : 逆运动学（适合离散目标点）
    controller_type: str = "osc"

    # 动作缩放（单位：米 / 弧度）
    action_scale: float = 0.05
    action_scale_rot: Optional[float] = None   # None → 使用 action_scale
    action_scale_hand: Optional[float] = 0.005 # None → 使用 action_scale

    # 控制器增益（None → 使用控制器内部默认值）
    osc_gains: Optional[OSCGains] = None
    ik_gains: Optional[PDGains] = None

    # 初始构型（None → 使用模型 qpos0）
    init_arm_qpos: Optional[np.ndarray] = None
    init_hand_qpos: Optional[np.ndarray] = None


# ====================== 机器人硬件参数 ======================

@dataclass
class RobotHardwareConfig:
    """机器人本体与传感器参数."""
    rot_xyz_deg: Tuple[float, float, float] = (-90, 0, 0)
    attach_point_name: str = "right_hand"
    tactile_backend: str = "simple_avg"  # "simple" | "simple_avg" | "physics" | "physics_avg"
    physics: Optional[object] = None     # PhysicsConfig，避免循环导入用 object


# ====================== 场景外观参数 ======================

@dataclass
class SceneConfig:
    """场景外观与桌子配置."""
    has_table: bool = True
    table_size: Tuple[float, float, float] = (0.8, 1.2, 0.05)
    table_pos: Tuple[float, float, float] = (0.5, 0.0, 0.55)
    table_surface_texture: Optional[str] = field(
        default_factory=lambda: DefaultTextures.TABLE_SURFACE
    )
    table_leg_texture: Optional[str] = field(
        default_factory=lambda: DefaultTextures.TABLE_LEG
    )
    table_surface_rgba: Tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)
    table_leg_rgba: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)


# ====================== 聚合配置（对外主接口）======================

@dataclass
class RobotConfig:
    """
    机器人环境完整配置（聚合类）.

    使用示例::

        cfg = RobotConfig(
            sim=SimConfig(control_freq=30.0),
            action=ActionConfig(action_mode="ee"),
        )
    """
    hardware: RobotHardwareConfig = field(default_factory=RobotHardwareConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)

    # ------------------------------------------------------------------
    # 向后兼容属性：旧代码直接访问 cfg.action_mode / cfg.control_freq 等
    # 所有属性均代理到对应子配置
    # ------------------------------------------------------------------

    @property
    def action_mode(self) -> str:
        return self.action.action_mode

    @property
    def controller_type(self) -> str:
        return self.action.controller_type

    @property
    def action_scale(self) -> float:
        return self.action.action_scale

    @property
    def action_scale_rot(self) -> Optional[float]:
        return self.action.action_scale_rot

    @property
    def action_scale_hand(self) -> Optional[float]:
        return self.action.action_scale_hand

    @property
    def osc_gains(self) -> Optional[OSCGains]:
        return self.action.osc_gains

    @property
    def ik_gains(self) -> Optional[PDGains]:
        return self.action.ik_gains

    @property
    def init_arm_qpos(self) -> Optional[np.ndarray]:
        return self.action.init_arm_qpos

    @property
    def init_hand_qpos(self) -> Optional[np.ndarray]:
        return self.action.init_hand_qpos

    @property
    def control_freq(self) -> float:
        return self.sim.control_freq

    @property
    def sim_freq(self) -> float:
        return self.sim.sim_freq

    @property
    def max_episode_steps(self) -> int:
        return self.sim.max_episode_steps

    @property
    def n_sim_steps_per_control(self) -> int:
        return self.sim.n_sim_steps_per_control

    @property
    def rot_xyz_deg(self):
        return self.hardware.rot_xyz_deg

    @property
    def attach_point_name(self) -> str:
        return self.hardware.attach_point_name

    @property
    def tactile_backend(self) -> str:
        return self.hardware.tactile_backend

    @property
    def physics(self):
        return self.hardware.physics

    @property
    def has_table(self) -> bool:
        return self.scene.has_table

    @property
    def table_size(self):
        return self.scene.table_size

    @property
    def table_pos(self):
        return self.scene.table_pos

    @property
    def table_surface_texture(self) -> Optional[str]:
        return self.scene.table_surface_texture

    @property
    def table_leg_texture(self) -> Optional[str]:
        return self.scene.table_leg_texture

    @property
    def table_surface_rgba(self):
        return self.scene.table_surface_rgba

    @property
    def table_leg_rgba(self):
        return self.scene.table_leg_rgba