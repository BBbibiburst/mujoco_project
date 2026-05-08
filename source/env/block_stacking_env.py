"""
Block Stacking 任务环境.

任务描述：
    机械臂+灵巧手将桌面上的一个立方体（cube_top）堆叠到另一个立方体（cube_base）上。

观测空间（扁平化 Dict，SB3 兼容）：
    - camera_rgb:      (240, 320, 3)  俯视相机 RGB 图像
    - tactile_bottom:  (5, 10, 7)     5手指 × 10行 × 7列  底部指节
    - tactile_middle:  (5, 8, 5)      5手指 × 8行 × 5列   中部指节
    - tactile_top:     (5, 6, 5)      5手指 × 6行 × 5列   顶部指节
    - proprioception:  (13,)          机械臂7DOF + 手6DOF 关节角度

动作空间（根据 base_env.py 的 action_mode 定义）：
    - "joint" : 13维 机械臂7Dof + 手部6Dof 关节增量
    - "ee" : 12维 末端位姿增量（位置3D + 姿态3D）+ 手部6Dof 关节增量
"""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import mujoco
import numpy as np
from .base_env import RobotArmEnvBase, RobotConfig
from gymnasium import spaces

# 导入触觉传感器常量和工具函数
from source.sensors.tactile_sensor import (
    TactileReader,
    DISPLAY_ORDER,
    FINGER_PHALANX_ORDER,
)

# ====================== 相机配置 ======================


@dataclass
class CameraConfig:
    """单个相机的位姿配置."""

    name: str
    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]  # (w, x, y, z)


# ====================== 任务配置 ======================


@dataclass
class BlockStackingConfig:
    """Block Stacking 任务专用配置."""

    obs_camera_name: str = "frontview"

    # 物体配置
    cube_size: float = 0.025
    cube_mass: float = 0.1
    cube_base_color: Tuple = (0.2, 0.6, 0.9, 1.0)   # 蓝色 - 底部立方体
    cube_top_color: Tuple = (0.9, 0.3, 0.2, 1.0)    # 红色 - 顶部立方体

    # 堆叠配置
    stack_threshold: float = 0.005  # 水平偏移阈值（认为堆叠成功的最大xy偏差）
    stack_height_tolerance: float = 0.003  # 高度容差

    # 目标标记颜色
    target_color: Tuple = (0.1, 0.8, 0.1, 0.3)

    # 物体放置区域配置（两个立方体分别在不同的区域）
    spawn_range: Tuple = (0.12, 0.10)  # (half_x, half_y) 每个立方体的采样范围
    base_spawn_center: Tuple = (0.42, -0.12)   # 底部立方体中心偏移
    top_spawn_center: Tuple = (0.42, 0.12)     # 顶部立方体中心偏移

    # 最小间距（防止初始位置重叠）
    min_cube_distance: float = 0.08

    # 运动配置
    approach_height: float = 0.12
    grasp_height_offset: float = 0.03

    # 手部开合配置（关节角）
    hand_open: np.ndarray = field(default_factory=lambda: np.zeros(6))
    hand_close: np.ndarray = field(default_factory=lambda: np.full(6, 0.01))

    # 相机配置（统一管理，与 _build_scene 保持同步）
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


# ====================== 触觉传感器配置 ======================

# 显式指定手指顺序（5根），不依赖字典插入顺序
_FINGER_NAMES: List[str] = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
_N_FINGERS: int = len(_FINGER_NAMES)  # 5

# 每个指节类型的传感器分辨率 (rows, cols)
_TACTILE_LEVELS: Dict[str, Tuple[int, int]] = {
    "bottom": (10, 7),
    "middle": (8, 5),
    "top": (6, 5),
}

# 指节名称 → 指节类型（用于调试验证）
_PHALANX_NAME_TO_LEVEL: Dict[str, str] = {}
for _finger, _phalanges in FINGER_PHALANX_ORDER.items():
    for _idx, _name in enumerate(_phalanges):
        _PHALANX_NAME_TO_LEVEL[_name] = {0: "bottom", 1: "middle", 2: "top"}[_idx]


# ====================== Block Stacking 环境 ======================


class BlockStackingEnv(RobotArmEnvBase):
    """
    Block Stacking 任务强化学习环境.
    任务：将 cube_top 堆叠到 cube_base 上.
    """

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[BlockStackingConfig] = None,
    ):
        self.task_cfg = task_config or BlockStackingConfig()

        # 验证 obs_camera_name 在相机列表中存在
        cam_names = [c.name for c in self.task_cfg.cameras]
        assert self.task_cfg.obs_camera_name in cam_names, (
            f"obs_camera_name '{self.task_cfg.obs_camera_name}' "
            f"not in cameras list {cam_names}"
        )
        self._cam_name: str = self.task_cfg.obs_camera_name
        self._camera_names: List[str] = cam_names

        if robot_config is None:
            robot_config = RobotConfig()

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        robot_config.table_surface_texture = str(
            PROJECT_ROOT / "assets/textures/ceramic.png"
        )
        robot_config.table_leg_texture = str(PROJECT_ROOT / "assets/textures/metal.png")

        super().__init__(robot_config)

        self._base_init_pos = np.zeros(3)
        self._top_init_pos = np.zeros(3)

        # MuJoCo ID 缓存（-1 表示未缓存）
        self._base_body_id: int = -1
        self._top_body_id: int = -1
        self._base_free_jnt_qposadr: int = -1
        self._top_free_jnt_qposadr: int = -1
        self._cached_model_ptr: int = -1  # 用于检测 model 是否重建

        # 相机渲染器（在 _reset_scene 中按需重建）
        self._renderer: Optional[mujoco.Renderer] = None

    # ====================== 高度计算======================

    def _compute_cube_z(self) -> float:
        """
        计算立方体中心的世界坐标系 Z 高度.

        立方体放在桌面上，底部贴桌面，中心高度 = 桌面高度 + 物体半高

        Returns:
            cube_z: 立方体中心的世界 Z
        """
        return self._table_height + self.task_cfg.cube_size

    def _compute_stacked_z(self) -> float:
        """
        计算堆叠后顶部立方体的目标中心高度.

        底部立方体中心在 table_height + cube_size
        顶部立方体中心在底部立方体中心 + 2*cube_size

        Returns:
            stacked_z: 堆叠后顶部立方体中心的目标世界 Z
        """
        return self._table_height + 3 * self.task_cfg.cube_size

    # ====================== 观测与动作空间 ======================

    @property
    def observation_space(self) -> spaces.Dict:
        img_h, img_w = 240, 320
        return spaces.Dict(
            {
                "camera_rgb": spaces.Box(0, 255, (img_h, img_w, 3), dtype=np.uint8),
                "tactile_bottom": spaces.Box(
                    0, 255, (_N_FINGERS, 10, 7), dtype=np.uint8
                ),
                "tactile_middle": spaces.Box(
                    0, 255, (_N_FINGERS, 8, 5), dtype=np.uint8
                ),
                "tactile_top": spaces.Box(0, 255, (_N_FINGERS, 6, 5), dtype=np.uint8),
                "proprioception": spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32),
            }
        )

    # ====================== 必须实现的抽象方法 ======================

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        """向 spec 添加任务特定元素（在基类桌子基础上扩展）."""
        wb = spec.worldbody
        tc = self.task_cfg

        cube_z = self._compute_cube_z()
        stacked_z = self._compute_stacked_z()

        # 默认位置
        default_base_pos = [tc.base_spawn_center[0], tc.base_spawn_center[1], cube_z]
        default_top_pos = [tc.top_spawn_center[0], tc.top_spawn_center[1], cube_z]

        # 1. 底部立方体（cube_base）
        base_obj = wb.add_body(
            name="cube_base", pos=default_base_pos
        )
        base_obj.add_geom(
            name="cube_base_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.cube_size] * 3,
            rgba=list(tc.cube_base_color),
            mass=tc.cube_mass,
            friction=[1.0, 0.005, 0.0001],
            condim=4,
        )
        base_obj.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="cube_base_joint")

        # 2. 顶部立方体（cube_top）
        top_obj = wb.add_body(
            name="cube_top", pos=default_top_pos
        )
        top_obj.add_geom(
            name="cube_top_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.cube_size] * 3,
            rgba=list(tc.cube_top_color),
            mass=tc.cube_mass,
            friction=[1.0, 0.005, 0.0001],
            condim=4,
        )
        top_obj.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="cube_top_joint")

        # 3. 目标位置 marker（在 cube_base 正上方，可视化目标堆叠位置）
        # 注意：这里使用 base_obj.add_body，使其成为 cube_base 的子物体
        target_marker = base_obj.add_body(
            name="target_marker", 
            pos=[0, 0, 2 * tc.cube_size]  # 这里的坐标是相对于 cube_base 中心的相对坐标
        )
        target_marker.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.cube_size * 1.1, tc.cube_size * 1.1, 0.001],
            rgba=list(tc.target_color),
            contype=0,
            conaffinity=0,
        )

        # 4. 相机（统一从 task_cfg.cameras 读取，不再硬编码）
        for cam in tc.cameras:
            spec.worldbody.add_camera(
                name=cam.name, mode=0, pos=list(cam.pos), quat=list(cam.quat)
            )

    def _get_obs(self) -> Dict[str, Any]:
        """构造观测：相机图像 + 触觉图像 + 本体感觉."""
        camera_rgb = self._render_camera()
        tactile = self._get_tactile_grouped()
        proprioception = np.concatenate(
            [self.get_arm_qpos(), self.get_hand_qpos()]
        ).astype(np.float32)

        return {
            "camera_rgb": camera_rgb,
            "tactile_bottom": tactile["bottom"],
            "tactile_middle": tactile["middle"],
            "tactile_top": tactile["top"],
            "proprioception": proprioception,
        }

    def _compute_reward(self) -> float:
        """计算奖励：鼓励 cube_top 堆叠到 cube_base 上."""
        base_pos = self._get_base_pos()
        top_pos = self._get_top_pos()
        tc = self.task_cfg

        # 目标堆叠位置（cube_base 正上方）
        target_pos = base_pos.copy()
        target_pos[2] = self._compute_stacked_z()

        # 水平距离奖励（鼓励 top 在 base 正上方）
        horizontal_dist = np.linalg.norm(top_pos[:2] - base_pos[:2])
        horizontal_reward = -horizontal_dist

        # 高度奖励（鼓励 top 达到目标高度）
        target_z = self._compute_stacked_z()
        height_diff = target_z - top_pos[2]
        height_reward = -abs(height_diff)

        # 如果成功堆叠，给予大奖励
        if self._is_stacked():
            return 10.0

        # 组合奖励
        return horizontal_reward + 0.5 * height_reward

    def _is_terminated(self) -> bool:
        """终止条件：物体掉落桌面以下，或成功堆叠."""
        base_pos = self._get_base_pos()
        top_pos = self._get_top_pos()

        # 任一立方体掉落到桌面以下
        if base_pos[2] < self._table_height - 0.05 or top_pos[2] < self._table_height - 0.05:
            return True

        # 成功堆叠后终止（可选，如果需要持续保持则注释掉）
        if self._is_stacked():
            return True

        return False

    def _is_truncated(self) -> bool:
        """截断条件：达到最大步数等（由基类管理）."""
        return False

    def _reset_scene(self) -> None:
        """随机化两个立方体位置，重建缓存和渲染器."""
        tc = self.task_cfg

        # 无条件刷新 ID 缓存
        self._cache_ids()

        # 若 model 已重建，强制重建 Renderer
        model_ptr = id(self.model)
        if self._renderer is None or self._cached_model_ptr != model_ptr:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
            self._cached_model_ptr = model_ptr

        cube_z = self._compute_cube_z()
        half_x, half_y = tc.spawn_range

        # 底部立方体采样范围
        base_cx, base_cy = tc.base_spawn_center
        base_pos_range = np.array(
            [
                [base_cx - half_x, base_cy - half_y, cube_z],
                [base_cx + half_x, base_cy + half_y, cube_z],
            ]
        )

        # 顶部立方体采样范围
        top_cx, top_cy = tc.top_spawn_center
        top_pos_range = np.array(
            [
                [top_cx - half_x, top_cy - half_y, cube_z],
                [top_cx + half_x, top_cy + half_y, cube_z],
            ]
        )

        # 采样位置，确保两个立方体之间有足够的距离
        max_attempts = 50
        for _ in range(max_attempts):
            base_pos = self._sample_pos(base_pos_range)
            top_pos = self._sample_pos(top_pos_range)

            dist = np.linalg.norm(base_pos[:2] - top_pos[:2])
            if dist >= tc.min_cube_distance:
                break

        self._base_init_pos = base_pos.copy()
        self._top_init_pos = top_pos.copy()

        # 更新底部立方体 qpos
        if self._base_free_jnt_qposadr >= 0:
            adr = self._base_free_jnt_qposadr
            self.data.qpos[adr : adr + 3] = base_pos
            self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_base_joint")
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0

        # 更新顶部立方体 qpos
        if self._top_free_jnt_qposadr >= 0:
            adr = self._top_free_jnt_qposadr
            self.data.qpos[adr : adr + 3] = top_pos
            self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_top_joint")
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0

    # ====================== 视觉与触觉观测 ======================

    def _render_camera(self) -> np.ndarray:
        """渲染观测相机 RGB 图像."""
        if self._renderer is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        self._renderer.update_scene(self.data, camera=self._cam_name)
        return self._renderer.render()

    def _get_tactile_grouped(self) -> Dict[str, np.ndarray]:
        """获取触觉图像，按指节类型分组，输出 shape: (N_FINGERS, H, W)."""
        if self.reader is None:
            return self._empty_tactile_grouped()

        try:
            tactile_imgs = self.reader.read_image(self.data)
            if not tactile_imgs:
                return self._empty_tactile_grouped()

            result: Dict[str, np.ndarray] = {}
            for level, level_idx in [("bottom", 0), ("middle", 1), ("top", 2)]:
                expected_h, expected_w = _TACTILE_LEVELS[level]
                level_images = []
                for finger in _FINGER_NAMES:
                    phalanx_name = FINGER_PHALANX_ORDER[finger][level_idx]
                    if phalanx_name in tactile_imgs:
                        img = tactile_imgs[phalanx_name]
                        if img.shape == (expected_w, expected_h):
                            img = img.T
                        elif img.shape != (expected_h, expected_w):
                            import cv2

                            img = cv2.resize(
                                img,
                                (expected_w, expected_h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        level_images.append(img.astype(np.uint8))
                    else:
                        level_images.append(
                            np.zeros((expected_h, expected_w), dtype=np.uint8)
                        )

                result[level] = np.stack(level_images, axis=0)  # (_N_FINGERS, H, W)

            return result

        except Exception as e:
            if getattr(self, "_debug", False):
                print("[Tactile ERROR]", repr(e))
            self._tactile_error_count = getattr(self, "_tactile_error_count", 0) + 1
            return self._empty_tactile_grouped()

    def _empty_tactile_grouped(self) -> Dict[str, np.ndarray]:
        """返回全零的分组触觉图像（维度与 observation_space 一致）."""
        return {
            "bottom": np.zeros((_N_FINGERS, 10, 7), dtype=np.uint8),
            "middle": np.zeros((_N_FINGERS, 8, 5), dtype=np.uint8),
            "top": np.zeros((_N_FINGERS, 6, 5), dtype=np.uint8),
        }

    def _verify_tactile_shapes(self) -> None:
        """调试用：验证触觉传感器实际分辨率与 _TACTILE_LEVELS 是否一致."""
        if self.reader is None:
            print("[Verify] reader is None")
            return

        tactile_imgs = self.reader.read_image(self.data)
        if not tactile_imgs:
            print("[Verify] no tactile images")
            return

        print("=== 触觉传感器实际分辨率 ===")
        for name in DISPLAY_ORDER:
            if name in tactile_imgs:
                img = tactile_imgs[name]
                level = _PHALANX_NAME_TO_LEVEL.get(name, "unknown")
                print(f"  {name} ({level}): shape={img.shape}, max={img.max():.1f}")
            else:
                print(f"  {name}: MISSING")

        print("\n=== 预期分辨率 (rows, cols) ===")
        for level, (h, w) in _TACTILE_LEVELS.items():
            print(f"  {level}: ({h}, {w})")

    # ====================== 动作应用 ======================

    def _apply_action(self, action: np.ndarray) -> None:
        """应用动作并更新任务阶段."""
        super()._apply_action(action)

    # ====================== 内部辅助方法 ======================

    def _get_base_pos(self) -> np.ndarray:
        """获取底部立方体质心位置."""
        if self._base_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._base_body_id].copy()

    def _get_base_quat(self) -> np.ndarray:
        """获取底部立方体姿态四元数."""
        if self._base_body_id < 0:
            self._cache_ids()
        return self.data.xquat[self._base_body_id].copy()

    def _get_top_pos(self) -> np.ndarray:
        """获取顶部立方体质心位置."""
        if self._top_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._top_body_id].copy()

    def _get_top_quat(self) -> np.ndarray:
        """获取顶部立方体姿态四元数."""
        if self._top_body_id < 0:
            self._cache_ids()
        return self.data.xquat[self._top_body_id].copy()

    def _is_stacked(self) -> bool:
        """
        检查是否成功堆叠.

        条件：
        1. cube_top 的水平位置接近 cube_base 的水平位置
        2. cube_top 的高度接近目标堆叠高度
        3. cube_top 的速度较小（稳定放置）
        """
        base_pos = self._get_base_pos()
        top_pos = self._get_top_pos()
        tc = self.task_cfg

        # 水平偏移检查
        horizontal_dist = np.linalg.norm(top_pos[:2] - base_pos[:2])
        if horizontal_dist > tc.stack_threshold:
            return False

        # 高度检查
        target_z = self._compute_stacked_z()
        if abs(top_pos[2] - target_z) > tc.stack_height_tolerance:
            return False

        # 稳定性检查（速度较小）
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_top_joint")
        if jnt_id >= 0:
            dof_adr = self.model.jnt_dofadr[jnt_id]
            top_vel = self.data.qvel[dof_adr : dof_adr + 6]
            if np.linalg.norm(top_vel) > 0.1:  # 速度阈值
                return False

        return True

    def _cache_ids(self) -> None:
        """
        缓存常用 MuJoCo ID.

        在每次 _reset_scene 开头无条件调用，确保 model 重建后
        缓存不会持有过期 ID（不再依赖 -1 哨兵懒检查）.
        """
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_base"
        )
        assert self._base_body_id >= 0, (
            "cube_base body not found in model. "
            "Make sure _build_scene() adds a body named 'cube_base'."
        )

        self._top_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_top"
        )
        assert self._top_body_id >= 0, (
            "cube_top body not found in model. "
            "Make sure _build_scene() adds a body named 'cube_top'."
        )

        base_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_base_joint"
        )
        self._base_free_jnt_qposadr = (
            self.model.jnt_qposadr[base_jnt_id] if base_jnt_id >= 0 else -1
        )

        top_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_top_joint"
        )
        self._top_free_jnt_qposadr = (
            self.model.jnt_qposadr[top_jnt_id] if top_jnt_id >= 0 else -1
        )

    def _sample_pos(self, pos_range: np.ndarray) -> np.ndarray:
        """在 pos_range 定义的长方体区域内均匀采样 3D 位置."""
        lo, hi = pos_range[0], pos_range[1]
        return lo + self.np_random.random(3) * (hi - lo)