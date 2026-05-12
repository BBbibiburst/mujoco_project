"""
Block Lifting 任务环境.

任务描述：
    机械臂+灵巧手从桌面上抓取一个立方体，将其提升到指定高度以上。

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
class BlockLiftingConfig:
    """Block Lifting 任务专用配置."""

    obs_camera_name: str = "frontview"

    # 物体配置
    obj_size: float = 0.025
    obj_mass: float = 0.1
    obj_color: Tuple = (0.9, 0.2, 0.1, 1.0)

    # 目标高度配置
    target_lift_height: float = 0.15  # 立方体需要被提升到的目标高度（相对于桌面）
    target_color: Tuple = (0.1, 0.8, 0.1, 0.4)

    # 物体放置区域配置
    obj_spawn_range: Tuple = (0.30, 0.15)  # (half_x, half_y) 相对于桌面中心的偏移范围
    obj_spawn_center: Tuple = (0.45, 0.0)  # 桌面中心偏移 (x, y)

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

        self._obj_init_pos = np.zeros(3)

        # MuJoCo ID 缓存（-1 表示未缓存）
        self._obj_body_id: int = -1
        self._obj_free_jnt_qposadr: int = -1
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
        return self._table_height + self.task_cfg.obj_size

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

        # 物体初始位置（在 reset 中会随机化，这里用默认值）
        default_obj_pos = [tc.obj_spawn_center[0], tc.obj_spawn_center[1], cube_z]

        # 1. 目标物体（立方体，放在桌面上）
        obj = wb.add_body(
            name="target_object", pos=default_obj_pos
        )
        obj.add_geom(
            name="obj_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.obj_size] * 3,
            rgba=list(tc.obj_color),
            mass=tc.obj_mass,
            friction=[1.0, 0.5, 0.001],
            condim=4,
            conaffinity=15,
        )
        obj.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="obj_free_joint")

        # 2. 目标高度 marker（可视化目标高度平面，无碰撞）
        # 在目标高度处放置一个半透明的平面标记
        target_marker = wb.add_body(
            name="target_marker", 
            pos=[tc.obj_spawn_center[0], tc.obj_spawn_center[1], 
                 self._table_height + tc.target_lift_height]
        )
        target_marker.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.15, 0.15, 0.001],  # 较大的平面标记
            rgba=list(tc.target_color),
            contype=0,
            conaffinity=0,
        )

        # 3. 相机（统一从 task_cfg.cameras 读取，不再硬编码）
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
        """计算奖励：鼓励物体高度超过目标高度."""
        obj_pos = self._get_obj_pos()
        target_z = self._table_height + self.task_cfg.target_lift_height

        # 基于高度的奖励
        height_reward = max(0.0, obj_pos[2] - target_z)

        # 如果物体被成功提升，给予额外奖励
        if obj_pos[2] >= target_z:
            height_reward += 1.0

        return height_reward

    def _is_terminated(self) -> bool:
        """终止条件：物体掉落（低于桌面）或成功提升并保持."""
        obj_pos = self._get_obj_pos()

        # 物体掉落到桌面以下
        if obj_pos[2] < self._table_height - 0.05:
            return True

        return False

    def _is_truncated(self) -> bool:
        """截断条件：达到最大步数等（由基类管理）."""
        return False

    def _reset_scene(self) -> None:
        """随机化物体位置（限制在桌面指定区域内），重建缓存和渲染器."""
        tc = self.task_cfg

        # 无条件刷新 ID 缓存（防止 model 重建后缓存失效）
        self._cache_ids()

        # 若 model 已重建（指针变化），强制重建 Renderer
        model_ptr = id(self.model)
        if self._renderer is None or self._cached_model_ptr != model_ptr:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
            self._cached_model_ptr = model_ptr

        # 计算立方体中心高度
        cube_z = self._compute_cube_z()

        # 在指定范围内随机采样物体位置
        half_x, half_y = tc.obj_spawn_range
        center_x, center_y = tc.obj_spawn_center

        obj_pos_range = np.array(
            [
                [center_x - half_x, center_y - half_y, cube_z],
                [center_x + half_x, center_y + half_y, cube_z],
            ]
        )

        obj_pos = self._sample_pos(obj_pos_range)
        self._obj_init_pos = obj_pos.copy()

        # 更新物体 qpos
        if self._obj_free_jnt_qposadr >= 0:
            adr = self._obj_free_jnt_qposadr
            # 设置位置和四元数 (Identity)
            self.data.qpos[adr : adr + 3] = obj_pos
            self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

            # 关键：通过 joint_id 找到对应的 6 维速度空间并清零（包含旋转角速度）
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free_joint")
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0  # 清除平移(3) + 旋转(3)

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

    def _get_obj_pos(self) -> np.ndarray:
        """获取物体质心位置."""
        if self._obj_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._obj_body_id].copy()

    def _get_obj_quat(self) -> np.ndarray:
        """获取物体姿态四元数."""
        if self._obj_body_id < 0:
            self._cache_ids()
        return self.data.xquat[self._obj_body_id].copy()

    def _cache_ids(self) -> None:
        """
        缓存常用 MuJoCo ID.

        在每次 _reset_scene 开头无条件调用，确保 model 重建后
        缓存不会持有过期 ID（不再依赖 -1 哨兵懒检查）.
        """
        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object"
        )
        assert self._obj_body_id >= 0, (
            "target_object body not found in model. "
            "Make sure _build_scene() adds a body named 'target_object'."
        )

        jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free_joint"
        )
        self._obj_free_jnt_qposadr = (
            self.model.jnt_qposadr[jnt_id] if jnt_id >= 0 else -1
        )

    def _sample_pos(self, pos_range: np.ndarray) -> np.ndarray:
        """在 pos_range 定义的长方体区域内均匀采样 3D 位置."""
        lo, hi = pos_range[0], pos_range[1]
        return lo + self.np_random.random(3) * (hi - lo)

    def get_obj_height(self) -> float:
        """获取物体当前高度（相对于桌面）."""
        obj_pos = self._get_obj_pos()
        return obj_pos[2] - self._table_height

    def is_lifted(self) -> bool:
        """检查物体是否已被提升到目标高度以上."""
        return self.get_obj_height() >= self.task_cfg.target_lift_height