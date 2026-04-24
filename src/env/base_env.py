"""
通用强化学习环境基类.

基于 Gymnasium 接口规范，封装机械臂+灵巧手的 MuJoCo 仿真环境。
所有具体任务环境继承此基类，只需重写少量方法即可快速开发新任务。

继承后需要实现的方法：
    - _build_scene(spec)       : 向 MjSpec 添加任务所需的物体、相机、传感器
    - _get_obs(data)           : 构建并返回观测向量
    - _compute_reward(data)    : 计算当前步奖励
    - _is_terminated(data)     : 判断是否达到成功终止条件
    - _is_truncated(data)      : 判断是否超时或触发安全截断
    - _reset_scene(data)       : 重置任务特定状态（物体位置等）

可选重写：
    - _get_info(data)          : 返回额外调试信息字典
    - observation_space        : 属性，返回 gym.Space（默认 Box）
    - action_space             : 属性，返回 gym.Space（默认 Box）
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Union

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    # 提供最小化占位符，允许在无 gymnasium 时也能运行
    class spaces:
        class Box:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                self.low = np.full(shape, low, dtype=dtype) if shape else np.array(low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype) if shape else np.array(high, dtype=dtype)
                self.shape = shape or self.low.shape
                self.dtype = dtype
            def sample(self):
                return np.random.uniform(self.low, self.high).astype(self.dtype)
        
        # ✅ FIX: 补充 Dict 占位类
        class Dict:
            def __init__(self, spaces_dict: Dict[str, Any]):
                self.spaces = spaces_dict
            
            def sample(self):
                return {k: v.sample() for k, v in self.spaces.items()}

from src.robot.robot_arm_system import get_combined_spec, PhysicsConfig
from src.controllers.position_controller import (
    OSC_PositionController, OSCGains,
    IK_PositionController, PDGains,
)
from src.controllers.hand_arm_controller import HandArmController
from src.sensors.tactile_sensor import TactileReader


# ====================== 环境配置数据类 ======================

@dataclass
class RobotConfig:
    """机器人与仿真基础配置."""
    # 机器人参数
    rot_xyz_deg: Tuple[float, float, float] = (-90, 0, 0)
    attach_point_name: str = "right_hand"
    tactile_backend: str = "simple_avg"   # "physics" | "simple_avg" | "none"
    physics: Optional[PhysicsConfig] = None

    # 仿真参数
    control_freq: float = 20.0            # 策略控制频率 [Hz]（每步调用一次 step）
    sim_freq: float = 1000.0              # 物理仿真频率 [Hz]（由 model.opt.timestep 决定）
    max_episode_steps: int = 500          # 单回合最大步数

    # 动作空间模式：决定动作如何解析
    #   "osc_pose" : 6D 位姿增量 (3位移 + 3旋转) + 手部 = 12维
    #   "osc_pos"  : 3D 位置增量 + 手部 = 9维（姿态自由）
    #   "joint_pd" : 关节空间增量 = 13维
    action_mode: str = "osc_pose"

    # 底层控制器类型：决定用哪种控制器执行
    #   "osc" : OSC_PositionController（支持 set_ee_target / set_target）
    #   "ik"  : IK_PositionController（支持 set_ee_target / set_target）
    controller_type: str = "osc"

    # 动作缩放因子
    #   位置控制：单位 米
    #   姿态控制：单位 弧度
    #   关节控制：单位 弧度
    action_scale: float = 0.05
    # 姿态增量单独缩放（可选，若 None 则使用 action_scale）
    action_scale_rot: Optional[float] = None

    # 控制器增益
    osc_gains: Optional[OSCGains] = None  # OSC 控制器增益，None 时使用默认
    ik_gains: Optional[PDGains] = None    # IK 控制器增益，None 时使用默认

    # 初始构型（弧度）
    init_arm_qpos: Optional[np.ndarray] = None   # None 时使用模型 qpos0 默认值
    init_hand_qpos: Optional[np.ndarray] = None  # None 时使用模型 qpos0 默认值

    @property
    def n_sim_steps_per_control(self) -> int:
        """每个控制步对应的仿真步数."""
        return max(1, int(self.sim_freq / self.control_freq))


@dataclass
class EnvStats:
    """回合统计信息（只读，由基类自动维护）."""
    episode_count: int = 0
    total_steps: int = 0
    episode_steps: int = 0
    episode_reward: float = 0.0
    success_count: int = 0


# ====================== 基类 ======================

class RobotArmEnvBase(ABC):
    """
    机械臂+灵巧手强化学习环境基类.

    流程图：
        reset() → _build_scene() → _reset_scene()
                                        ↓
        step(action) → _apply_action() → sim_step×N → _get_obs()
                                                           ↓
                                              _compute_reward() + _is_terminated()

    基类负责：
        - 模型加载与编译（调用 get_combined_spec）
        - 控制器初始化（HandArmController + OSC/IK）
        - 仿真步进（按 control_freq 与 sim_freq 比例推进）
        - 动作空间处理（末端位移/姿态或关节增量）
        - 统计信息维护

    类负责：
        - 场景搭建（物体、相机）
        - 观测空间定义
        - 奖励函数
        - 终止判断
    """

    # ---- 子类需要声明的常量（可覆盖） ----
    ARM_DOF: int = 7
    HAND_DOF: int = 6
    TOTAL_DOF: int = 13

    def __init__(self, config: Optional[RobotConfig] = None):
        self.cfg = config or RobotConfig()
        self.stats = EnvStats()

        # 以下属性在 _init_simulation() 中初始化
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.reader: Optional[TactileReader] = None
        self.hw: Optional[HandArmController] = None
        self.controller: Optional[Union[OSC_PositionController, IK_PositionController]] = None

        self._initialized = False

    # ====================== 抽象接口（子类必须实现） ======================

    @abstractmethod
    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        """
        向未编译的 MjSpec 中添加任务所需元素.

        在模型编译前调用，可添加：
            - 目标物体（add_body + geom + free joint）
            - 障碍物
            - 摄像头
            - 额外传感器

        Args:
            spec: 未编译的合并规格对象（已包含机械臂+手爪）。

        Example:
            >>> def _build_scene(self, spec):
            ...     body = spec.worldbody.add_body(name="cube", pos=[0.4, 0, 0.05])
            ...     body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025]*3)
            ...     body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        """
        ...

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """
        构建并返回当前观测向量.

        在每个 step 结束后以及 reset 时调用。
        可通过 self.data / self.model / self.reader 获取仿真状态。

        Returns:
            np.ndarray: 观测向量，形状和类型与 observation_space 一致。
        """
        ...

    @abstractmethod
    def _compute_reward(self) -> float:
        """
        计算当前步奖励.

        Returns:
            float: 奖励值（正奖励为好，负奖励为惩罚）。
        """
        ...

    @abstractmethod
    def _is_terminated(self) -> bool:
        """
        判断回合是否因成功而终止.

        Returns:
            bool: True 表示任务成功完成。
        """
        ...

    @abstractmethod
    def _reset_scene(self) -> None:
        """
        重置任务特定状态（物体位置、目标点等）.

        在每次 reset() 时调用，应随机化或重置场景。
        仅需处理任务相关状态；机器人构型由基类负责重置。
        """
        ...

    # ====================== 可选重写 ======================

    def _is_truncated(self) -> bool:
        """超时截断（默认：超过 max_episode_steps）."""
        return self.stats.episode_steps >= self.cfg.max_episode_steps

    def _get_info(self) -> Dict[str, Any]:
        """额外调试信息（默认：返回统计数据）."""
        return {
            "episode_steps": self.stats.episode_steps,
            "episode_reward": self.stats.episode_reward,
            "episode_count": self.stats.episode_count,
        }

    @property
    def observation_space(self):
        """观测空间（默认：7+6+6=19维 Box，子类可覆盖）."""
        obs_dim = self.ARM_DOF + self.HAND_DOF + 6  # qpos(7) + hand_qpos(6) + ee_pose(6)
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    @property
    def action_space(self):
        """
        动作空间.

        根据 action_mode 返回对应维度：
            - "osc_pose" : 末端位移(3) + 末端旋转(3) + 手部(6) = 12维
            - "osc_pos"  : 末端位移(3) + 手部(6) = 9维
            - "joint_pd" : 臂关节增量(7) + 手部增量(6) = 13维
        """
        mode = self.cfg.action_mode
        if mode == "osc_pose":
            action_dim = 3 + 3 + self.HAND_DOF   # 12
        elif mode == "osc_pos":
            action_dim = 3 + self.HAND_DOF       # 9
        elif mode == "joint_pd":
            action_dim = self.ARM_DOF + self.HAND_DOF  # 13
        else:
            raise ValueError(
                f"Unknown action_mode: '{mode}'. "
                f"Expected 'osc_pose', 'osc_pos', or 'joint_pd'."
            )
        return spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    # ====================== 公开接口 ======================

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境，返回初始观测.

        Args:
            seed: 随机数种子（可选）。

        Returns:
            (obs, info): 初始观测和调试信息。
        """
        if seed is not None:
            np.random.seed(seed)

        # 首次调用时初始化仿真
        if not self._initialized:
            self._init_simulation()
            self._initialized = True

        # 重置仿真状态
        mujoco.mj_resetData(self.model, self.data)

        # 重置机器人到初始构型
        self._reset_robot_pose()

        # 前向推算（更新 xpos, site_xpos 等）
        mujoco.mj_forward(self.model, self.data)

        # ---- 重建控制器，彻底消除上一回合状态残留 ----
        self._rebuild_controller()

        # 任务特定重置
        self._reset_scene()

        # 再次前向推算（反映场景重置后的状态）
        mujoco.mj_forward(self.model, self.data)

        # 更新统计
        self.stats.episode_count += 1
        self.stats.episode_steps = 0
        self.stats.episode_reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    def _rebuild_controller(self) -> None:
        """重新初始化控制器（用于 reset 时彻底重置状态）."""
        ctrl_type = self.cfg.controller_type

        if ctrl_type == "osc":
            gains = self.cfg.osc_gains if self.cfg.osc_gains is not None else OSCGains()
            self.controller = OSC_PositionController(
                base=self.hw,
                model=self.model,
                gains=gains,
            )
        elif ctrl_type == "ik":
            gains = self.cfg.ik_gains if self.cfg.ik_gains is not None else PDGains()
            self.controller = IK_PositionController(
                base=self.hw,
                model=self.model,
                gains=gains,
            )
        else:
            raise ValueError(
                f"Unknown controller_type: '{ctrl_type}'. "
                f"Expected 'osc' or 'ik'."
            )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步控制，推进仿真.

        Args:
            action: 动作向量（归一化到 [-1, 1]）。

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        if not self._initialized:
            raise RuntimeError("请先调用 reset() 初始化环境。")

        # 应用动作（控制机器人）
        self._apply_action(action)

        # 推进仿真（多个物理步对应一个控制步）
        for _ in range(self.cfg.n_sim_steps_per_control):
            mujoco.mj_step(self.model, self.data)

        # 收集结果
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._get_info()

        # 更新统计
        self.stats.episode_steps += 1
        self.stats.total_steps += 1
        self.stats.episode_reward += reward
        if terminated:
            self.stats.success_count += 1

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """启动 MuJoCo 被动查看器（阻塞，用于调试）."""
        if not self._initialized:
            raise RuntimeError("请先调用 reset() 初始化环境。")
        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            while v.is_running():
                mujoco.mj_step(self.model, self.data)
                v.sync()

    def close(self) -> None:
        """释放资源."""
        self._initialized = False
        self.model = None
        self.data = None

    # ====================== 便捷查询接口 ======================

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回末端 (pos[3], quat[4])."""
        return self.controller.get_ee_pose(self.data)

    def get_arm_qpos(self) -> np.ndarray:
        """返回机械臂关节角度 (7,)."""
        return self.data.qpos[self.controller.arm_qpos_ids].copy()

    def get_hand_qpos(self) -> np.ndarray:
        """返回灵巧手关节角度 (6,)."""
        return self.data.qpos[self.controller.hand_qpos_ids].copy()

    def get_body_pos(self, body_name: str) -> np.ndarray:
        """返回指定 body 的世界坐标位置."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' 不存在于模型中。")
        return self.data.xpos[body_id].copy()

    def get_tactile(self) -> Optional[Dict]:
        """返回触觉传感器图像字典（如果启用）."""
        if self.reader is None:
            return None
        return self.reader.read_image(self.data)

    # ====================== 私有实现 ======================

    def _init_simulation(self) -> None:
        """初始化 MuJoCo 仿真（加载模型、初始化控制器）."""
        print(f"[{self.__class__.__name__}] 正在初始化仿真环境...")

        # 1. 获取合并后的未编译 spec
        spec, reader = get_combined_spec(
            rot_xyz_deg=self.cfg.rot_xyz_deg,
            attach_point_name=self.cfg.attach_point_name,
            physics=self.cfg.physics,
            tactile_backend=self.cfg.tactile_backend,
        )

        # 2. 子类添加任务场景元素
        self._build_scene(spec)

        # 3. 编译模型
        print(f"[{self.__class__.__name__}] 编译模型中...")
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # 4. 绑定触觉传感器
        reader.bind(self.model)
        self.reader = reader

        # 5. 初始化 HandArmController（底层运动学/动力学接口，不存状态，无需重建）
        self.hw = HandArmController(self.model)

        # 6. 初始化控制器
        self._rebuild_controller()

        print(
            f"[{self.__class__.__name__}] 初始化完成。"
            f" nv={self.model.nv}, nu={self.model.nu}"
        )

    def _reset_robot_pose(self) -> None:
        """
        将机器人重置到初始构型.
        """
        arm_ids  = self.controller.arm_qpos_ids
        hand_ids = self.controller.hand_qpos_ids

        # --- 机械臂 ---
        if self.cfg.init_arm_qpos is not None:
            self.data.qpos[arm_ids] = self.cfg.init_arm_qpos
        else:
            # 回退到模型默认构型（XML 中 <key> 或 <default> 定义的 qpos0）
            self.data.qpos[arm_ids] = self.model.qpos0[arm_ids]

        # --- 灵巧手 ---
        if self.cfg.init_hand_qpos is not None:
            self.data.qpos[hand_ids] = self.cfg.init_hand_qpos
        else:
            self.data.qpos[hand_ids] = self.model.qpos0[hand_ids]

        # --- 速度清零（防止残留速度导致初始抖动）---
        self.data.qvel[arm_ids]  = 0.0
        self.data.qvel[hand_ids] = 0.0

    def _apply_action(self, action: np.ndarray) -> None:
        """
        将归一化动作映射到控制器指令.

        根据 action_mode 解析动作，统一调用控制器的 set_ee_target 或 set_target。
        控制器类型（osc/ik）与动作解析方式解耦。
        """
        action = np.clip(action, -1.0, 1.0)
        mode = self.cfg.action_mode

        if mode == "osc_pose":
            self._apply_osc_pose_action(action)
        elif mode == "osc_pos":
            self._apply_osc_pos_action(action)
        elif mode == "joint_pd":
            self._apply_joint_pd_action(action)
        else:
            raise ValueError(
                f"Unknown action_mode: '{mode}'. "
                f"Expected 'osc_pose', 'osc_pos', or 'joint_pd'."
            )

    def _apply_osc_pose_action(self, action: np.ndarray) -> None:
        """
        6D 位姿增量动作解析.

        action[0:3]  → 末端位置增量（×action_scale）
        action[3:6]  → 末端姿态增量（×action_scale_rot 或 action_scale）
        action[6:]   → 手部目标增量（×action_scale）
        """
        ee_pos, ee_quat = self.get_ee_pose()

        # 位置增量
        delta_pos = action[:3] * self.cfg.action_scale
        new_ee_pos = ee_pos + delta_pos

        # 姿态增量：轴角 → 四元数
        scale_rot = self.cfg.action_scale_rot if self.cfg.action_scale_rot is not None else self.cfg.action_scale
        delta_rpy = action[3:6] * scale_rot
        
        delta_quat = np.zeros(4)
        angle = np.linalg.norm(delta_rpy)
        if angle > 1e-6:
            axis = delta_rpy / angle
            mujoco.mju_axisAngle2Quat(delta_quat, axis, angle)
        else:
            delta_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # 组合姿态：new_quat = delta_quat ⊗ current_quat
        new_ee_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_ee_quat, delta_quat, ee_quat)
        mujoco.mju_normalize4(new_ee_quat)

        # 手部控制
        hand_delta = action[6:] * self.cfg.action_scale
        new_hand = self.get_hand_qpos() + hand_delta

        # 统一调用 set_ee_target（OSC 或 IK 控制器都支持）
        self.controller.set_ee_target(
            self.data,
            ee_pos_target=new_ee_pos,
            ee_quat_target=new_ee_quat,
            hand_target=new_hand,
        )

    def _apply_osc_pos_action(self, action: np.ndarray) -> None:
        """
        3D 位置增量动作解析（姿态自由）.

        action[0:3]  → 末端位置增量（×action_scale）
        action[3:]   → 手部目标增量（×action_scale）
        """
        ee_pos, _ = self.get_ee_pose()

        # 位置增量
        delta_pos = action[:3] * self.cfg.action_scale
        new_ee_pos = ee_pos + delta_pos

        # 手部控制
        hand_delta = action[3:] * self.cfg.action_scale
        new_hand = self.get_hand_qpos() + hand_delta

        # 姿态自由：不传 ee_quat_target
        self.controller.set_ee_target(
            self.data,
            ee_pos_target=new_ee_pos,
            ee_quat_target=None,
            hand_target=new_hand,
        )

    def _apply_joint_pd_action(self, action: np.ndarray) -> None:
        """
        关节空间增量动作解析.

        action[:7]   → 臂关节角度增量（×action_scale）
        action[7:]   → 手部关节增量（×action_scale）
        """
        current_arm = self.get_arm_qpos()
        current_hand = self.get_hand_qpos()

        arm_delta = action[:self.ARM_DOF] * self.cfg.action_scale
        hand_delta = action[self.ARM_DOF:] * self.cfg.action_scale

        # 统一调用 set_target（OSC 或 IK 控制器都支持）
        self.controller.set_target(
            self.data,
            arm_target=current_arm + arm_delta,
            hand_target=current_hand + hand_delta,
        )