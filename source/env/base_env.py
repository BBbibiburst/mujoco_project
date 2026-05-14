"""
通用强化学习环境基类.

基于 Gymnasium 接口规范，封装机械臂+灵巧手的 MuJoCo 仿真环境。
所有具体任务环境继承此基类，只需重写少量方法即可快速开发新任务。

继承后需要实现的方法：
    - _build_scene(spec)   : 向 MjSpec 添加任务所需物体、相机、传感器
    - _get_obs()           : 构建并返回观测向量
    - _compute_reward()    : 计算当前步奖励
    - _is_terminated()     : 判断是否达到成功终止条件
    - _reset_scene()       : 重置任务特定状态（物体位置等）

可选重写：
    - _is_truncated()      : 超时或安全截断（默认：超过 max_episode_steps）
    - _get_info()          : 返回额外调试信息字典
    - observation_space    : 属性，返回 gym.Space（默认 Box 19 维）
    - action_space         : 属性，返回 gym.Space（默认 Box）
    - _create_scene_builder() : 返回自定义 SceneBuilder
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from source.robot.robot_arm_system import get_combined_spec
from source.controllers.position_controller import (
    OSCController, IKController, fast_tracking_osc_gains, PDGains,
)
from source.controllers.hand_arm_controller import HandArmController
from source.sensors.tactile_sensor import TactileReader

from .env_config import RobotConfig
from .scene_builder import BaseSceneBuilder


# ====================== 回合统计 ======================

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

    流程：
        reset() → _build_scene() → _reset_scene()
                                        ↓
        step(action) → _apply_action() → sim_step×N → _get_obs()
                                                           ↓
                                              _compute_reward() + _is_terminated()
    """

    metadata = {"render_modes": ["human"]}

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

        # 场景构建器（可被子类替换）
        self._scene_builder: BaseSceneBuilder = self._create_scene_builder()

        # Renderer（懒创建，由 _get_renderer() 管理）
        self._renderer: Optional[mujoco.Renderer] = None
        self._renderer_model_id: int = -1
        self._renderer_h: int = 240
        self._renderer_w: int = 320

        # 缓存 spaces
        self._observation_space: Optional[spaces.Space] = None
        self._action_space: Optional[spaces.Box] = None

        # 末端目标（在 reset 中初始化）
        self._target_pos: np.ndarray = np.zeros(3)
        self._target_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
        self._target_hand: np.ndarray = np.zeros(self.HAND_DOF)

        self._initialized: bool = False

    # ====================== 抽象接口 ======================

    @abstractmethod
    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        """向未编译的 MjSpec 中添加任务所需元素（物体、相机、传感器）."""
        ...

    @abstractmethod
    def _get_obs(self) -> Union[np.ndarray, Dict]:
        """构建并返回当前观测（在每步结束后及 reset 时调用）."""
        ...

    @abstractmethod
    def _compute_reward(self) -> float:
        """计算当前步奖励."""
        ...

    @abstractmethod
    def _is_terminated(self) -> bool:
        """判断是否因成功而终止."""
        ...

    @abstractmethod
    def _reset_scene(self) -> None:
        """重置任务特定状态（物体位置、目标点等）."""
        ...

    # ====================== 可选重写 ======================

    def _create_scene_builder(self) -> BaseSceneBuilder:
        """返回场景构建器实例，子类可替换为自定义构建器."""
        return BaseSceneBuilder()

    def _is_truncated(self) -> bool:
        """超时截断（默认：超过 max_episode_steps）."""
        return self.stats.episode_steps >= self.cfg.max_episode_steps

    def _get_info(self) -> Dict[str, Any]:
        """额外调试信息（默认：回合统计数据）."""
        return {
            "episode_steps":  self.stats.episode_steps,
            "episode_reward": self.stats.episode_reward,
            "episode_count":  self.stats.episode_count,
        }

    @property
    def observation_space(self) -> spaces.Box:
        """默认观测空间：qpos(7) + hand_qpos(6) + ee_pose(6) = 19 维."""
        if self._observation_space is None:
            obs_dim = self.ARM_DOF + self.HAND_DOF + 6
            self._observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: spaces.Space) -> None:
        self._observation_space = value

    @property
    def action_space(self) -> spaces.Box:
        """
        动作空间.
        - "joint": 7+6=13 维
        - "ee":    6+6=12 维
        """
        if self._action_space is None:
            if self.cfg.action_mode == "joint":
                action_dim = self.ARM_DOF + self.HAND_DOF
            elif self.cfg.action_mode == "ee":
                action_dim = 6 + self.HAND_DOF
            else:
                raise ValueError(f"Unknown action_mode: '{self.cfg.action_mode}'")
            self._action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )
        return self._action_space

    @action_space.setter
    def action_space(self, value: spaces.Space) -> None:
        self._action_space = value

    # ====================== 公开接口 ======================

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        if not self._initialized:
            self._init_simulation()
            self._initialized = True

        mujoco.mj_resetData(self.model, self.data)
        self._reset_robot_pose()
        mujoco.mj_forward(self.model, self.data)
        self._rebuild_controller()
        self._reset_scene()
        mujoco.mj_forward(self.model, self.data)

        # 以当前实际末端位姿为初始目标，防止零动作漂移
        _pos, _quat = self.get_ee_pose()
        self._target_pos  = _pos.copy()
        self._target_quat = _quat.copy()
        self._target_hand = self.get_hand_qpos().copy()

        self.stats.episode_count  += 1
        self.stats.episode_steps   = 0
        self.stats.episode_reward  = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        if not self._initialized:
            raise RuntimeError("请先调用 reset() 初始化环境。")

        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)

        for _ in range(self.cfg.n_sim_steps_per_control - 1):
            self._keep_target()
            mujoco.mj_step(self.model, self.data)

        obs        = self._get_obs()
        reward     = self._compute_reward()
        terminated = self._is_terminated()
        truncated  = self._is_truncated()
        info       = self._get_info()

        self.stats.episode_steps  += 1
        self.stats.total_steps    += 1
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
                self._keep_target()
                mujoco.mj_step(self.model, self.data)
                v.sync()

    def close(self) -> None:
        """释放所有资源."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_model_id = -1
        self._initialized = False
        self.model      = None
        self.data       = None
        self.reader     = None
        self.hw         = None
        self.controller = None

    # ====================== 便捷查询接口 ======================

    def get_arm_qpos(self) -> np.ndarray:
        """机械臂关节角度 (7,)."""
        return self.data.qpos[self.controller.arm_qpos_ids].copy()

    def get_hand_qpos(self) -> np.ndarray:
        """灵巧手关节角度 (6,)."""
        return self.data.qpos[self.controller.hand_qpos_ids].copy()

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """末端执行器位姿（位置 + 四元数）."""
        return self.controller.get_ee_pose(self.data)

    def get_body_pos(self, body_name: str) -> np.ndarray:
        """指定 body 的世界坐标位置."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' 不存在于模型中。")
        return self.data.xpos[body_id].copy()

    def get_site_pos(self, site_name: str) -> np.ndarray:
        """指定 site 的世界坐标位置."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' 不存在于模型中。")
        return self.data.site_xpos[site_id].copy()

    def get_tactile(self) -> Optional[Dict]:
        """原始触觉图像字典（如果启用）."""
        return self.reader.read_image(self.data) if self.reader else None

    # ====================== Renderer 管理 ======================

    def _get_renderer(self, height: int = 240, width: int = 320) -> mujoco.Renderer:
        """
        懒创建 Renderer，model 重建时自动重建.

        子类直接调用此方法获取 renderer，无需手动管理生命周期。
        """
        model_id = id(self.model)
        if (self._renderer is None
                or self._renderer_model_id != model_id
                or self._renderer_h != height
                or self._renderer_w != width):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_model_id = model_id
            self._renderer_h = height
            self._renderer_w = width
        return self._renderer

    def render_camera(self, camera_name: str, height: int = 240, width: int = 320) -> np.ndarray:
        """渲染指定相机的 RGB 图像，供子类直接调用."""
        renderer = self._get_renderer(height, width)
        renderer.update_scene(self.data, camera=camera_name)
        return renderer.render()

    # ====================== 私有实现 ======================

    def _init_simulation(self) -> None:
        spec, reader = get_combined_spec(
            rot_xyz_deg=self.cfg.rot_xyz_deg,
            attach_point_name=self.cfg.attach_point_name,
            physics=self.cfg.physics,
            tactile_backend=self.cfg.tactile_backend,
        )

        # 场景构建（基础 + 任务特定）
        self._table_height = self._scene_builder.build(spec, self.cfg.scene)
        self._build_scene(spec)

        self.model = spec.compile()
        self.data  = mujoco.MjData(self.model)

        reader.bind(self.model)
        self.reader = reader

        self.hw = HandArmController(self.model)
        self._rebuild_controller()

    def _rebuild_controller(self) -> None:
        """重建控制器（每次 reset 时调用，彻底清除上一回合状态）."""
        ctrl_type = self.cfg.controller_type

        if ctrl_type == "osc":
            gains = self.cfg.osc_gains or fast_tracking_osc_gains()
            self.controller = OSCController(base=self.hw, model=self.model, gains=gains)
        elif ctrl_type == "ik":
            gains = self.cfg.ik_gains or PDGains()
            self.controller = IKController(base=self.hw, model=self.model, gains=gains)
        else:
            raise ValueError(f"Unknown controller_type: '{ctrl_type}'")

    def _reset_robot_pose(self) -> None:
        arm_ids  = self.controller.arm_qpos_ids
        hand_ids = self.controller.hand_qpos_ids

        self.data.qpos[arm_ids]  = (
            self.cfg.init_arm_qpos if self.cfg.init_arm_qpos is not None
            else self.model.qpos0[arm_ids]
        )
        self.data.qpos[hand_ids] = (
            self.cfg.init_hand_qpos if self.cfg.init_hand_qpos is not None
            else self.model.qpos0[hand_ids]
        )

        self.data.qvel[self.controller.arm_qvel_ids]  = 0.0
        self.data.qvel[self.controller.hand_qvel_ids] = 0.0

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -1.0, 1.0)
        if self.cfg.action_mode == "joint":
            self._apply_joint_action(action)
        elif self.cfg.action_mode == "ee":
            self._apply_ee_action(action)
        else:
            raise ValueError(f"Unknown action_mode: '{self.cfg.action_mode}'")

    def _apply_joint_action(self, action: np.ndarray) -> None:
        scale_hand = self.cfg.action_scale_hand or self.cfg.action_scale
        self.controller.set_joint_target(
            self.data,
            arm_target=self.get_arm_qpos() + action[:self.ARM_DOF] * self.cfg.action_scale,
            hand_target=self.get_hand_qpos() + action[self.ARM_DOF:] * scale_hand,
        )

    def _apply_ee_action(self, action: np.ndarray) -> None:
        cur_pos, cur_quat = self.get_ee_pose()
        scale_rot  = self.cfg.action_scale_rot or self.cfg.action_scale
        scale_hand = self.cfg.action_scale_hand or self.cfg.action_scale

        self._target_pos = cur_pos + action[:3] * self.cfg.action_scale

        rot_delta = action[3:6] * scale_rot
        angle = np.linalg.norm(rot_delta)
        if angle < 1e-8:
            delta_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            delta_quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(delta_quat, rot_delta / angle, angle)

        self._target_quat = np.zeros(4)
        mujoco.mju_mulQuat(self._target_quat, cur_quat, delta_quat)

        self.controller.set_ee_target(
            self.data,
            ee_pos_target=self._target_pos,
            ee_quat_target=self._target_quat,
            hand_target=self.get_hand_qpos() + action[6:] * scale_hand,
        )

    def _keep_target(self) -> None:
        """保持当前目标不变，持续推进仿真（防止多步内漂移）."""
        self.controller.hold(self.data)