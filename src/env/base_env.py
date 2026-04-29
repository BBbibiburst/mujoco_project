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
import gymnasium as gym
from gymnasium import spaces
from src.robot.robot_arm_system import get_combined_spec, PhysicsConfig
from src.controllers.position_controller import OSCController, IKController, OSCGains, PDGains
from src.controllers.hand_arm_controller import HandArmController
from src.sensors.tactile_sensor import TactileReader


# ====================== 环境配置数据类 ======================

@dataclass
class RobotConfig:
    """机器人与仿真基础配置."""
    # 机器人参数
    rot_xyz_deg: Tuple[float, float, float] = (-90, 0, 0)
    attach_point_name: str = "right_hand"
    tactile_backend: str = "simple_avg"   # "simple" "simple_avg" "physics" "physics_avg"
    physics: Optional[PhysicsConfig] = None

    # 仿真参数
    control_freq: float = 20.0            # 策略控制频率 [Hz]（每步调用一次 step）
    sim_freq: float = 1000.0              # 物理仿真频率 [Hz]（由 model.opt.timestep 决定）
    max_episode_steps: int = 500          # 单回合最大步数

    # 动作空间模式：决定动作如何解析
    #   "joint" : 7Dof 机械臂 + 手部 6Dof = 13维
    #   "ee" : 末端位姿增量（位置 + 姿态）+ 手部 6Dof = 12维
    action_mode: str = "joint"

    # 底层控制器类型：决定用哪种控制器执行
    #   "osc" : 基于操作空间控制的 OSCController（推荐，适合连续平滑控制）
    #   "ik"  : 基于逆运动学的 IKController（适合离散目标点，可能不够平滑）
    controller_type: str = "osc"

    # 动作缩放因子
    #   位置控制：单位 米
    #   姿态控制：单位 弧度
    #   关节控制：单位 弧度
    action_scale: float = 0.05
    # 姿态增量单独缩放（可选，若 None 则使用 action_scale）
    action_scale_rot: Optional[float] = None
    # 手部推杆增量单独缩放（单位 米，满量程 0.01 m）
    # 推杆每步合理步长约 0.001 m（满量程的 10%），远小于臂的 action_scale。
    # None 时 fallback 到 action_scale，但 action_scale=0.05 对推杆而言过大（5×满量程），建议单独设置 action_scale_hand=0.001。
    action_scale_hand: Optional[float] = 0.005

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

class RobotArmEnvBase(gym.Env, ABC):
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

    metadata = {"render_modes": ["human"]}  # gym.Env 要求
    # ---- 子类需要声明的常量（可覆盖） ----
    ARM_DOF: int = 7
    HAND_DOF: int = 6
    TOTAL_DOF: int = 13

    def __init__(self, config: Optional[RobotConfig] = None):
        super().__init__()
        self.cfg = config or RobotConfig()
        self.stats = EnvStats()

        # 以下属性在 _init_simulation() 中初始化
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.reader: Optional[TactileReader] = None
        self.hw: Optional[HandArmController] = None
        self.controller: Optional[Union[OSCController, IKController]] = None

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
            spec: 未编译的合并规格对象（已包含机械臂+灵巧手）。

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
            float: 奖励值（正奖励为奖励，负奖励为惩罚）。
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
            - "joint" : 7Dof 机械臂 + 手部 6Dof = 13维
            - "ee" : 末端位姿增量（位置 + 姿态）+ 手部 6Dof = 12维
         维度顺序统一为 [arm, hand]，其中 arm 根据 mode式解析为关节增量或末端位姿增量，hand 始终为关节增量。动作值归一化到 [-1, 1]，由 _apply_action 解析为实际控制指令。
            "joint" 模式下 action[:7] 是臂关节增量，action[7:] 是手部关节增量；
            "ee" 模式下 action[:6] 是末端位姿增量（位置 + 姿态），action[6:] 是手部关节增量。
         这种设计使得控制器类型（osc/ik）与动作解析方式解耦，便于在不同任务中灵活选择控制方式和动作表示。
        """
        mode = self.cfg.action_mode
        if mode == "joint":
            action_dim = self.ARM_DOF + self.HAND_DOF  # 机械臂 7Dof + 手部 6Dof = 13维
        elif mode == "ee":
            action_dim = 6 + self.HAND_DOF  # 末端位姿增量（位置+姿态）+ 手部 6Dof = 12维
        else:
            raise ValueError(
                f"Unknown action_mode: '{mode}'. "
                f"Expected 'joint' or 'ee'."
            )
        return spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    # ====================== 公开接口 ======================

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境，返回初始观测.
        
        Args:
            seed: 随机数种子（可选）。
            options: 额外选项（Gymnasium 标准接口，可选）。
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

        # ---- 初始化持久目标（防止零动作时漂移）----
        # 以当前实际末端位姿为起点，后续所有增量在此基础上累积
        _pos, _quat = self.get_ee_pose()
        self._target_pos: np.ndarray = _pos.copy()
        self._target_quat: np.ndarray = _quat.copy()
        self._target_hand: np.ndarray = self.get_hand_qpos().copy()

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
            self.controller = OSCController(
                base=self.hw,
                model=self.model,
                gains=gains,
            )
        elif ctrl_type == "ik":
            gains = self.cfg.ik_gains if self.cfg.ik_gains is not None else PDGains()
            self.controller = IKController(
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

        # 应用动作并推进仿真
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        # 在控制频率与仿真频率不匹配时，保持目标并推进额外的仿真步，确保动作持续生效，仿真状态更新充分。
        for _ in range(self.cfg.n_sim_steps_per_control - 1):
            # 重要！保持当前目标不变，持续推进仿真，否则动作只在第一步生效，后续仿真漂移。
            self._keep_target()
            # 继续推进仿真，更新状态（如触觉传感器）但不重新应用动作，避免污染前馈历史。
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
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # 4. 绑定触觉传感器
        reader.bind(self.model)
        self.reader = reader

        # 5. 初始化 HandArmController
        self.hw = HandArmController(self.model)

        # 6. 初始化控制器
        self._rebuild_controller()

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

        根据 action_mode 解析动作，统一调用控制器的set_joint_target或其他接口，控制器类型（osc/ik）与动作解析方式解耦。
        控制器类型（osc/ik）与动作解析方式解耦。
        """
        action = np.clip(action, -1.0, 1.0)
        mode = self.cfg.action_mode

        if mode == "joint":
            self._apply_joint_action(action)
        elif mode == "ee":
            self._apply_ee_action(action)
        else:
            raise ValueError(
                f"Unknown action_mode: '{mode}'. "
                f"Expected 'joint' or 'ee'."
            )

    def _apply_joint_action(self, action: np.ndarray) -> None:
        """
        关节空间增量动作解析.

        action[:7]   → 臂关节角度增量（×action_scale）
        action[7:]   → 手部关节增量（×action_scale）
        """
        current_arm = self.get_arm_qpos()
        current_hand = self.get_hand_qpos()

        arm_delta = action[:self.ARM_DOF] * self.cfg.action_scale
        # 手部推杆位移增量（满量程 0.01 m，需独立缩放）
        scale_hand = self.cfg.action_scale_hand if self.cfg.action_scale_hand is not None else self.cfg.action_scale
        hand_delta = action[self.ARM_DOF:] * scale_hand

        # 统一调用 
        self.controller.set_joint_target(
            self.data,
            arm_target=current_arm + arm_delta,
            hand_target=current_hand + hand_delta,
        )
    
    def _apply_ee_action(self, action: np.ndarray) -> None:
        """
        末端空间增量动作解析

        action[:6]   -> 位置 + 姿态增量
        action[6:]   -> 手部关节增量
        """
        current_pos, current_quat = self.get_ee_pose()
        current_hand = self.get_hand_qpos()

        # ----------------------------
        # 位置增量
        # ----------------------------
        pos_delta = action[:3] * self.cfg.action_scale
        self._target_pos = current_pos + pos_delta

        # ----------------------------
        # 姿态增量
        # ----------------------------
        rot_scale = (
            self.cfg.action_scale_rot
            if self.cfg.action_scale_rot is not None
            else self.cfg.action_scale
        )

        rot_delta = action[3:6] * rot_scale

        angle = np.linalg.norm(rot_delta)

        if angle < 1e-8:
            delta_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = rot_delta / angle
            delta_quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(delta_quat, axis, angle)

        # 注意：局部坐标系增量
        target_quat = np.zeros(4)
        mujoco.mju_mulQuat(target_quat, current_quat, delta_quat)

        self._target_quat = target_quat

        # ----------------------------
        # 手部增量
        # ----------------------------
        scale_hand = (
            self.cfg.action_scale_hand
            if self.cfg.action_scale_hand is not None
            else self.cfg.action_scale
        )

        hand_delta = action[6:] * scale_hand

        self.controller.set_ee_target(
            self.data,
            ee_pos_target=self._target_pos,
            ee_quat_target=self._target_quat,
            hand_target=current_hand + hand_delta,
        )
        
    def _keep_target(self) -> None:
        """保持当前目标不变，持续推进仿真.
        重要！否则动作只在第一步生效，后续仿真漂移。
        """
        self.controller.hold(self.data)
    
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回末端执行器当前位姿（位置 + 四元数）."""
        return self.controller.get_ee_pose(self.data)