"""
Block Lifting 任务环境.

任务描述：
    机械臂+灵巧手从桌面抓取立方体，将其提升到指定高度以上。

观测空间（扁平化 Dict，SB3 兼容）：
    - camera_rgb:      (240, 320, 3)
    - tactile_bottom:  (5, 10, 7)
    - tactile_middle:  (5, 8, 5)
    - tactile_top:     (5, 6, 5)
    - proprioception:  (13,)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from .base_env import RobotArmEnvBase
from .env_config import RobotConfig
from .tactile_obs import TactileObsHelper

# ====================== 相机配置 ======================


@dataclass
class CameraConfig:
    """单个相机的位姿配置."""

    name: str
    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]  # (w, x, y, z)


# ====================== 任务配置 ======================


@dataclass
class BlockLiftingConfig:
    """Block Lifting 任务专用配置."""

    obs_camera_name: str = "frontview"

    # 物体配置（obj_size 为 MuJoCo half-size，实际边长 = 2 × obj_size）
    obj_size: float = 0.025
    obj_mass: float = 0.1
    obj_color: Tuple = (0.9, 0.2, 0.1, 1.0)

    # 目标高度（相对于桌面）
    target_lift_height: float = 0.15
    target_color: Tuple = (0.1, 0.8, 0.1, 0.4)

    # 物体生成区域（相对于桌面中心）
    obj_spawn_range: Tuple = (0.26, 0.39)  # (half_x, half_y)
    obj_spawn_center: Tuple = (0.50, 0.0)  # (x, y)

    # 手部开合目标关节角
    hand_open: np.ndarray = field(default_factory=lambda: np.zeros(6))
    hand_close: np.ndarray = field(default_factory=lambda: np.full(6, 0.01))

    # 相机列表
    cameras: List[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig("frontview", (1.6, 0.0, 1.2), (0.56, 0.43, 0.43, 0.56)),
            CameraConfig("birdview", (-0.2, 0.0, 3.0), (0.7071, 0.0, 0.0, 0.7071)),
            CameraConfig("agentview", (1.0, 0.0, 1.0), (0.7071, 0.0, 0.7071, 0.0)),
            CameraConfig(
                "sideview", (-0.0565, 1.276, 1.488), (0.0099, 0.0069, 0.5912, 0.8064)
            ),
        ]
    )

    # 调试：方块生成区域示意（仅 MuJoCo 可视化可见，相机不可见）
    spawn_area_debug: bool = True
    spawn_area_color: Tuple = (0.0, 0.6, 1.0, 0.15)  # 半透明蓝色

    # 终止判断
    drop_threshold_offset: float = -0.025  # 物体底部触桌即失败

    # 奖励权重
    lift_base_reward: float = 2.0
    lift_progress_reward: float = 5.0
    lift_success_reward: float = 10.0
    grasp_stability_weight: float = 1.0
    grasp_stability_threshold: float = 50.0

    # 时间负奖励（每步惩罚）
    time_penalty: float = -0.01

    # 跌落负奖励（物体掉落时额外惩罚）
    drop_penalty: float = -5.0


# ====================== Block Lifting 环境 ======================


class BlockLiftingEnv(RobotArmEnvBase):
    """
    Block Lifting 任务强化学习环境.
    任务：从桌面抓取立方体，将其提升到目标高度以上。
    """

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[BlockLiftingConfig] = None,
    ):
        self.task_cfg = task_config or BlockLiftingConfig()

        cam_names = [c.name for c in self.task_cfg.cameras]
        assert self.task_cfg.obs_camera_name in cam_names, (
            f"obs_camera_name '{self.task_cfg.obs_camera_name}' "
            f"not in cameras: {cam_names}"
        )
        self._cam_name: str = self.task_cfg.obs_camera_name

        super().__init__(robot_config or RobotConfig())

        # 触觉助手（在 _init_simulation 后通过 _post_init_setup 绑定）
        self._tactile = TactileObsHelper()

        # MuJoCo ID 缓存
        self._obj_body_id: int = -1
        self._obj_free_jnt_qposadr: int = -1
        self._target_marker_body_id: int = -1
        self._spawn_area_geom_id: int = -1
        self._spawn_area_border_ids: List[int] = []

        # 回合辅助指标
        self._max_height: float = 0.0
        self._is_dropped: bool = False
        self._grasp_success: bool = False

    # ====================== 观测与动作空间 ======================

    @property
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "camera_rgb": spaces.Box(0, 255, (240, 320, 3), dtype=np.uint8),
                "proprioception": spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32),
                **TactileObsHelper.observation_spaces(),
            }
        )

    # ====================== 必须实现的抽象方法 ======================

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        """添加立方体、目标高度 marker 和相机."""
        wb = spec.worldbody
        tc = self.task_cfg
        cube_z = self._table_height + tc.obj_size

        # 立方体
        obj = wb.add_body(
            name="target_object",
            pos=[tc.obj_spawn_center[0], tc.obj_spawn_center[1], cube_z],
        )
        obj.add_geom(
            name="obj_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.obj_size] * 3,
            rgba=list(tc.obj_color),
            mass=tc.obj_mass,
            friction=[1.0, 0.5, 0.05],
            condim=4,
            conaffinity=15,
        )
        obj.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="obj_free_joint")

        # 目标高度 marker（mocap body，无碰撞）
        target_marker = wb.add_body(
            name="target_marker",
            pos=[
                tc.obj_spawn_center[0],
                tc.obj_spawn_center[1],
                self._table_height + tc.target_lift_height,
            ],
            mocap=True,
        )
        target_marker.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.15, 0.15, 0.001],
            rgba=list(tc.target_color),
            contype=0,
            conaffinity=0,
        )

        # 相机
        for cam in tc.cameras:
            spec.worldbody.add_camera(
                name=cam.name, mode=0, pos=list(cam.pos), quat=list(cam.quat)
            )

        # 调试：方块生成区域示意（仅 MuJoCo 可视化可见，相机不可见）
        # 显示整个 obj_spawn_range 定义的大区域，而非单个方块位置
        if tc.spawn_area_debug:
            half_x, half_y = tc.obj_spawn_range
            cx, cy = tc.obj_spawn_center
            # 使用 group=3 使其不在默认相机渲染组中（默认 group=0,1,2）
            wb.add_geom(
                name="spawn_area_debug",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_x, half_y, 0.001],
                pos=[cx, cy, self._table_height + 0.002],
                rgba=list(tc.spawn_area_color),
                contype=0,
                conaffinity=0,
                group=3,  # 仅 MuJoCo 可视化可见，相机不可见
            )
            # 边界线框（更清楚地显示区域范围）
            border_color = (0.0, 0.6, 1.0, 0.4)
            # 四条边线
            wb.add_geom(
                name="spawn_area_border_x1",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_x, 0.002, 0.0015],
                pos=[cx, cy - half_y, self._table_height + 0.003],
                rgba=list(border_color),
                contype=0, conaffinity=0, group=3,
            )
            wb.add_geom(
                name="spawn_area_border_x2",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_x, 0.002, 0.0015],
                pos=[cx, cy + half_y, self._table_height + 0.003],
                rgba=list(border_color),
                contype=0, conaffinity=0, group=3,
            )
            wb.add_geom(
                name="spawn_area_border_y1",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.002, half_y, 0.0015],
                pos=[cx - half_x, cy, self._table_height + 0.003],
                rgba=list(border_color),
                contype=0, conaffinity=0, group=3,
            )
            wb.add_geom(
                name="spawn_area_border_y2",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.002, half_y, 0.0015],
                pos=[cx + half_x, cy, self._table_height + 0.003],
                rgba=list(border_color),
                contype=0, conaffinity=0, group=3,
            )

    def _get_obs(self) -> Dict[str, Any]:
        """相机 RGB + 触觉图像 + 本体感觉."""
        tactile = self._tactile.get_grouped(self.data)
        return {
            "camera_rgb": self.render_camera(self._cam_name),
            "tactile_bottom": tactile["bottom"],
            "tactile_middle": tactile["middle"],
            "tactile_top": tactile["top"],
            "proprioception": np.concatenate(
                [self.get_arm_qpos(), self.get_hand_qpos()]
            ).astype(np.float32),
        }

    def _compute_reward(self) -> float:
        """
        奖励函数：
        1. 离地基础奖（物体离开桌面）
        2. 上升过程奖（线性，随高度增加）
        3. 通关大奖（达到目标高度）
        4. 抓握稳定性奖（触觉有激活且物体离地）
        5. 时间负奖励（每步惩罚，鼓励快速完成）
        6. 跌落负奖励（物体掉落时惩罚）
        """
        tc = self.task_cfg
        current_height = self._get_obj_height()
        self._max_height = max(self._max_height, current_height)

        lift_threshold = tc.obj_size
        reward = 0.0

        # 1. 时间负奖励（每步都扣）
        reward += tc.time_penalty

        # 2. 离地基础奖
        if current_height > lift_threshold:
            reward += tc.lift_base_reward
            progress = (current_height - lift_threshold) / max(
                tc.target_lift_height - lift_threshold, 1e-3
            )
            reward += np.clip(progress, 0.0, 1.0) * tc.lift_progress_reward

        # 3. 通关大奖
        if current_height >= tc.target_lift_height:
            reward += tc.lift_success_reward

        # 4. 抓握稳定性奖
        if (
            self._tactile.is_active(self.data, threshold=tc.grasp_stability_threshold)
            and current_height > lift_threshold
        ):
            reward += tc.grasp_stability_weight
            self._grasp_success = True

        # 5. 跌落负奖励（物体掉落时额外惩罚）
        if self._is_dropped:
            reward += tc.drop_penalty

        return reward

    def _is_terminated(self) -> Tuple[bool, bool]:
        """成功（达到目标高度）或失败（掉落/提升后落回桌面）时终止."""
        height = self._get_obj_height()
        tc = self.task_cfg

        # 更新历史最高高度
        self._max_height = max(self._max_height, height)

        # 缓冲高度 = 方块边长（2 * half_size）
        buffer_height = 2.0 * tc.obj_size

        # 判断是否曾经被提升离开桌面（超过一个边长高度）
        if not self._was_lifted and self._max_height > buffer_height:
            self._was_lifted = True

        # 成功：达到目标高度
        if height >= tc.target_lift_height:
            return True, True

        # 失败1：掉落到桌面以下（原有逻辑）
        if height < tc.drop_threshold_offset:
            self._is_dropped = True
            return True, False

        # 失败2：曾经被提升后又落回桌面（低于一个边长高度）
        if self._was_lifted and height <= buffer_height:
            self._is_dropped = True  # 复用 _is_dropped 标记表示失败
            return True, False

        return False, False

    def _reset_scene(self) -> None:
        """随机化物体位置和旋转，更新 marker，绑定触觉助手，重置辅助指标."""
        # 每次 reset 时重新缓存 ID（防止 model 重建后失效）
        self._cache_ids()
        self._was_lifted: bool = False

        # 绑定触觉助手
        self._tactile.bind(self.reader)

        tc = self.task_cfg
        cube_z = self._table_height + tc.obj_size

        # 随机采样物体位置
        half_x, half_y = tc.obj_spawn_range
        cx, cy = tc.obj_spawn_center
        lo = np.array([cx - half_x, cy - half_y, cube_z])
        hi = np.array([cx + half_x, cy + half_y, cube_z])
        obj_pos = lo + self.np_random.random(3) * (hi - lo)

        # 随机采样旋转四元数（仅绕 Z 轴，保持底面朝下）
        rand_quat = self._random_quaternion_z_axis()

        # 更新 qpos
        if self._obj_free_jnt_qposadr >= 0:
            adr = self._obj_free_jnt_qposadr
            self.data.qpos[adr : adr + 3] = obj_pos
            self.data.qpos[adr + 3 : adr + 7] = rand_quat  # 随机 Z 轴旋转

            jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free_joint"
            )
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0

        # 更新 target_marker XY 跟随物体
        if self._target_marker_body_id >= 0:
            mocap_id = self.model.body_mocapid[self._target_marker_body_id]
            if mocap_id >= 0:
                self.data.mocap_pos[mocap_id] = [
                    obj_pos[0],
                    obj_pos[1],
                    self._table_height + tc.target_lift_height,
                ]

        # 记录方块初始位置和旋转（供 _get_info 使用）
        self._obj_initial_pos = obj_pos.copy()
        self._obj_initial_quat = rand_quat.copy()

        # 重置辅助指标
        self._max_height = 0.0
        self._is_dropped = False
        self._grasp_success = False

    def _get_info(self) -> Dict[str, Any]:
        """
        返回包含方块位置、高度等任务信息的调试字典。
        继承基类默认的回合统计数据，并追加任务特定信息。
        """
        # 获取基类提供的回合统计信息
        info = super()._get_info()

        tc = self.task_cfg
        current_pos = self.get_block_position()
        current_height = self._get_obj_height()

        # 方块相关信息
        info["block"] = {
            "initial_position": self._obj_initial_pos.tolist() if hasattr(self, '_obj_initial_pos') else None,
            "initial_quaternion": self._obj_initial_quat.tolist() if hasattr(self, '_obj_initial_quat') else None,
            "current_position": current_pos.tolist(),
            "current_height": float(current_height),
            "target_height": float(tc.target_lift_height),
            "max_height": float(self._max_height),
            "is_lifted": bool(current_height >= tc.target_lift_height),
            "was_lifted": bool(self._was_lifted),
            "is_dropped": bool(self._is_dropped),
            "grasp_success": bool(self._grasp_success),
            "obj_size": float(tc.obj_size),
            "obj_mass": float(tc.obj_mass),
        }

        # 末端执行器信息
        try:
            ee_pos, ee_quat = self.get_ee_pose()
            info["end_effector"] = {
                "position": ee_pos.tolist(),
                "quaternion": ee_quat.tolist(),
            }
        except Exception:
            info["end_effector"] = None

        # 目标 marker 位置
        try:
            target_marker_pos = self.get_body_pos("target_marker")
            info["target_marker"] = {
                "position": target_marker_pos.tolist(),
            }
        except Exception:
            info["target_marker"] = None

        return info

    # ====================== 公开辅助方法 ======================

    def get_obj_height(self) -> float:
        """物体当前高度（相对于桌面）."""
        return self._get_obj_height()

    def is_lifted(self) -> bool:
        """物体是否达到目标高度."""
        return self._get_obj_height() >= self.task_cfg.target_lift_height

    def get_max_height(self) -> float:
        """本 episode 达到的最大高度."""
        return self._max_height

    def is_dropped(self) -> bool:
        """物体是否已掉落."""
        return self._is_dropped

    def is_grasp_success(self) -> bool:
        """是否成功抓握（触觉有激活且物体离地）."""
        return self._grasp_success

    def verify_tactile(self) -> None:
        """打印触觉传感器分辨率（调试用）."""
        self._tactile.verify_shapes(self.data)

    def get_block_position(self) -> np.ndarray:
        """
        获取方块当前的三维位置 [x, y, z]（相对于世界坐标系）。
        注意：z 值包含桌面高度。
        """
        if self._obj_body_id < 0:
            self._cache_ids()  # 确保 ID 已缓存
        return self.data.xpos[self._obj_body_id].copy()  # .copy() 避免修改原数据
    
    def get_block_quaternion(self) -> np.ndarray:
        """获取方块当前旋转四元数 [w, x, y, z]."""
        if self._obj_body_id < 0:
            self._cache_ids()
        return self.data.xquat[self._obj_body_id].copy()

    def get_mid_point_position(self) -> np.ndarray:
        thumb = self.get_site_pos("inspirehand_fingertip_thumb")
        finger3 = self.get_site_pos("inspirehand_fingertip_3")
        finger2 = self.get_site_pos("inspirehand_fingertip_2")
        self.midpoint = (thumb + (finger3 + finger2) / 2.0) / 2.0
        return self.midpoint

    # ====================== 私有方法 ======================

    def _get_obj_height(self) -> float:
        """物体中心 Z 减桌面高度."""
        if self._obj_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._obj_body_id][2] - self._table_height

    def _cache_ids(self) -> None:
        """缓存常用 MuJoCo ID（每次 reset 无条件刷新）."""
        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object"
        )
        assert self._obj_body_id >= 0, (
            "target_object body not found. "
            "Make sure _build_scene() adds a body named 'target_object'."
        )

        jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free_joint"
        )
        self._obj_free_jnt_qposadr = (
            self.model.jnt_qposadr[jnt_id] if jnt_id >= 0 else -1
        )

        self._target_marker_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker"
        )

        # 缓存生成区域 geom ID（大区域固定位置，不跟随单个方块）
        self._spawn_area_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "spawn_area_debug"
        )
        self._spawn_area_border_ids: List[int] = []
        for border_name in [
            "spawn_area_border_x1", "spawn_area_border_x2",
            "spawn_area_border_y1", "spawn_area_border_y2",
        ]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, border_name)
            if bid >= 0:
                self._spawn_area_border_ids.append(bid)

    def _random_quaternion_z_axis(self) -> np.ndarray:
        """
        生成仅绕 Z 轴的随机单位四元数 [w, x, y, z]。
        保持物体底面朝下，适合抓取任务。
        """
        angle = self.np_random.uniform(0, 2 * np.pi)  # 0~360度均匀分布
        half_angle = angle / 2.0
        return np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])