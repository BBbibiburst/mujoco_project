"""
抓取放置任务环境.

任务描述：
    机械臂+灵巧手从 bin1 中抓取一个立方体，将其搬运到 bin2 中并放置。

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
# ====================== 奖励相关常量 ======================

# 阶段阈值
_GRASP_HEIGHT_THRESH: float = 0.04   # 物体抬升超过此高度认为已抓起（相对 bin 底板）
_PLACE_XY_THRESH: float    = 0.06   # 物体 XY 中心距 bin2 中心小于此值认为放置成功
_PLACE_Z_THRESH: float     = 0.05   # 物体 Z 高度在 bin2 底板上方此范围内认为放置成功

# 奖励权重
_W_APPROACH: float  = 2.0    # 末端接近物体
_W_LIFT:     float  = 3.0    # 物体抬升高度
_W_TRANSPORT:float  = 3.0    # 物体接近目标
_W_PLACE:    float  = 5.0    # 物体接近目标 XY（放置阶段细化）
_W_TIME:     float  = -0.005 # 时间惩罚（每步）
_W_SUCCESS:  float  = 10.0   # 成功稀疏奖励
_W_GRASP_BONUS: float = 1.0  # 首次抬起物体里程碑奖励


@dataclass
class PickAndPlaceConfig:
    """抓取放置任务专用配置."""

    obs_camera_name: str = "frontview"

    # 物体配置
    obj_size: float = 0.025
    obj_mass: float = 0.1
    obj_color: Tuple = (0.9, 0.2, 0.1, 1.0)

    # 目标配置
    target_min_dist: float = 0.15
    target_color: Tuple = (0.1, 0.8, 0.1, 0.4)

    # Bin 配置
    bin1_pos: Tuple = (0.42, -0.18, 0.0)
    bin2_pos: Tuple = (0.42, 0.18, 0.0)
    bin_size: Tuple = (0.10, 0.14, 0.02)  # (half_x, half_y, half_z)
    bin_wall_thickness: float = 0.008
    bin1_rgba: Tuple = (0.7, 0.5, 0.3, 1.0)
    bin2_rgba: Tuple = (0.4, 0.25, 0.1, 1.0)

    # 运动配置
    lift_height: float = 0.15
    grasp_height_offset: float = 0.03
    approach_height: float = 0.12

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


# ====================== 抓取放置环境 ======================


class PickAndPlaceEnv(RobotArmEnvBase):
    """
    抓取放置任务强化学习环境.
    任务：从 bin1 抓取立方体，放置到 bin2 中。
    """

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[PickAndPlaceConfig] = None,
    ):
        self.task_cfg = task_config or PickAndPlaceConfig()

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
        self._target_pos = np.zeros(3)

        # MuJoCo ID 缓存（-1 表示未缓存）
        self._obj_body_id: int = -1
        self._obj_free_jnt_qposadr: int = -1
        self._cached_model_ptr: int = -1  # 用于检测 model 是否重建

        # 相机渲染器（在 _reset_scene 中按需重建）
        self._renderer: Optional[mujoco.Renderer] = None

    # ====================== 高度计算======================

    def _compute_scene_heights(self) -> Tuple[float, float, float]:
        """
        集中计算 bin、cube、marker 的世界坐标系 Z 高度。

        _add_bin 内部结构（bin body 原点 = 墙体中心）：
            底板中心  pos = [0, 0, -sz]，半厚 = wall
            底板顶面  = bin_body_z - sz + wall

        令底板顶面 == table_z，解得：
            bin_body_z = table_z + sz - wall

        NOTE: 此方法依赖 self._table_height，只能在基类 __init__ 完成后调用。

        Returns:
            bin_body_z  : bin body 原点的世界 Z（两个 bin 共用）
            cube_z      : cube 中心的世界 Z
            marker_z    : target marker 中心的世界 Z
        """
        tc = self.task_cfg
        sz = tc.bin_size[2]
        wall = tc.bin_wall_thickness
        t = self._table_height

        bin_body_z = t + sz + wall  # 底板底面贴桌面，bin 完全在桌面之上
        bin_floor_top = t + 2 * wall  # 底板顶面 = 桌面 + 底板全厚
        cube_z = bin_floor_top + tc.obj_size
        marker_z = bin_floor_top + 0.001

        return bin_body_z, cube_z, marker_z

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

    # ====================== Bin 构建辅助方法 ======================

    def _add_bin(
        self,
        worldbody,
        name: str,
        pos: List[float],
        size: Tuple[float, float, float] = (0.12, 0.16, 0.05),
        wall: float = 0.008,
        rgba: Tuple[float, float, float, float] = (0.55, 0.35, 0.2, 1.0),
    ):
        """
        创建 open-top bin。

        bin body 原点位于墙体中心：
            底板中心在 pos=[0, 0, -sz]，半厚 = wall
            底板顶面  = body_z - sz + wall
        """
        sx, sy, sz = size

        bin_body = worldbody.add_body(name=name, pos=pos)

        # 底板
        bin_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, 0, -sz],
            size=[sx, sy, wall],
            rgba=rgba,
            friction=[1.2, 0.01, 0.0001],
        )
        # 前墙
        bin_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, sy, 0],
            size=[sx, wall, sz],
            rgba=rgba,
            friction=[1.2, 0.01, 0.0001],
        )
        # 后墙
        bin_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, -sy, 0],
            size=[sx, wall, sz],
            rgba=rgba,
            friction=[1.2, 0.01, 0.0001],
        )
        # 左墙
        bin_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-sx, 0, 0],
            size=[wall, sy, sz],
            rgba=rgba,
            friction=[1.2, 0.01, 0.0001],
        )
        # 右墙
        bin_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[sx, 0, 0],
            size=[wall, sy, sz],
            rgba=rgba,
            friction=[1.2, 0.01, 0.0001],
        )

        return bin_body

    # ====================== 必须实现的抽象方法 ======================

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        """向 spec 添加任务特定元素（在基类桌子基础上扩展）."""
        wb = spec.worldbody
        tc = self.task_cfg

        bin_body_z, cube_z, marker_z = self._compute_scene_heights()

        # 1. 两个 bin
        self._add_bin(
            wb,
            "bin1",
            pos=[tc.bin1_pos[0], tc.bin1_pos[1], bin_body_z],
            size=tc.bin_size,
            wall=tc.bin_wall_thickness,
            rgba=tc.bin1_rgba,
        )
        self._add_bin(
            wb,
            "bin2",
            pos=[tc.bin2_pos[0], tc.bin2_pos[1], bin_body_z],
            size=tc.bin_size,
            wall=tc.bin_wall_thickness,
            rgba=tc.bin2_rgba,
        )

        # 2. 目标物体（初始放在 bin1 中心）
        obj = wb.add_body(
            name="target_object", pos=[tc.bin1_pos[0], tc.bin1_pos[1], cube_z]
        )
        obj.add_geom(
            name="obj_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.obj_size] * 3,
            rgba=list(tc.obj_color),
            mass=tc.obj_mass,
            friction=[1.0, 0.005, 0.0001],
            condim=4,
        )
        obj.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="obj_free_joint")

        # 3. 目标位置 marker（贴在 bin2 底板上，无碰撞）
        target_marker = wb.add_body(
            name="target_marker", pos=[tc.bin2_pos[0], tc.bin2_pos[1], marker_z]
        )
        target_marker.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.obj_size * 1.2, tc.obj_size * 1.2, 0.001],
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


    def _reset_scene(self) -> None:
        """随机化物体和目标位置，重建缓存和渲染器，重置里程碑标志。"""
        # 重置里程碑（每 episode 只触发一次的奖励）
        self._grasp_bonus_given: bool = False
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

        # 集中计算高度，与 _build_scene 保持一致
        _, cube_z, marker_z = self._compute_scene_heights()

        sx, sy, _ = tc.bin_size
        bin1_x, bin1_y, _ = tc.bin1_pos
        bin2_x, bin2_y, _ = tc.bin2_pos

        obj_pos_range = np.array(
            [
                [bin1_x - sx * 0.6, bin1_y - sy * 0.6, cube_z],
                [bin1_x + sx * 0.6, bin1_y + sy * 0.6, cube_z],
            ]
        )
        target_pos_range = np.array(
            [
                [bin2_x - sx * 0.6, bin2_y - sy * 0.6, marker_z],
                [bin2_x + sx * 0.6, bin2_y + sy * 0.6, marker_z],
            ]
        )

        obj_pos = self._sample_pos(obj_pos_range)
        self._obj_init_pos = obj_pos.copy()

        target_pos = self._sample_pos(target_pos_range)
        self._target_pos = target_pos.copy()

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
        缓存常用 MuJoCo ID。

        在每次 _reset_scene 开头无条件调用，确保 model 重建后
        缓存不会持有过期 ID（不再依赖 -1 哨兵懒检查）。
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

# ====================== 状态判断辅助 ======================

    def _get_bin2_center_xy(self) -> np.ndarray:
        """返回 bin2 中心的 XY 坐标。"""
        tc = self.task_cfg
        return np.array([tc.bin2_pos[0], tc.bin2_pos[1]])

    def _get_bin_floor_z(self) -> float:
        """返回 bin 底板顶面世界 Z（两个 bin 共用，与 _compute_scene_heights 一致）。"""
        tc = self.task_cfg
        wall = tc.bin_wall_thickness
        return self._table_height + 2 * wall

    def _is_object_lifted(self) -> bool:
        """物体是否已被抬离 bin1 底板（高于阈值）。"""
        obj_z   = self._get_obj_pos()[2]
        floor_z = self._get_bin_floor_z()
        return (obj_z - floor_z) > _GRASP_HEIGHT_THRESH

    def _is_object_in_bin2(self) -> bool:
        """
        判断物体是否已放置进 bin2。
        条件：XY 在 bin2 范围内，且 Z 在底板附近（未悬空）。
        """
        tc      = self.task_cfg
        obj_pos = self._get_obj_pos()
        floor_z = self._get_bin_floor_z()

        sx, sy, _ = tc.bin_size
        bx, by    = tc.bin2_pos[0], tc.bin2_pos[1]

        in_xy = (abs(obj_pos[0] - bx) < sx * 0.9 and
                abs(obj_pos[1] - by) < sy * 0.9)
        # 物体 Z 中心在底板上方 obj_size 的合理范围内（静置）
        obj_size = tc.obj_size
        in_z  = floor_z < obj_pos[2] < floor_z + obj_size * 2.5

        return in_xy and in_z

    # ====================== 奖励函数 ======================

    def _compute_reward(self) -> float:
        """
        分阶段密集奖励 + 成功稀疏奖励。

        阶段划分（互相渐进，非互斥）：
            1. Approach  : 末端执行器接近物体
            2. Grasp/Lift: 物体被抬起（Z 方向）
            3. Transport : 物体接近 bin2 目标中心
            4. Place     : 物体在 bin2 内稳定放置 → 成功

        所有距离奖励使用负指数形式 –w·d，使奖励连续可微。
        里程碑奖励（_is_object_lifted）在首次触发时给出额外加成。
        """
        reward = 0.0

        obj_pos   = self._get_obj_pos()
        ee_pos, _ = self.get_ee_pose()

        # ------------------------------------------------------------------
        # 1. 接近奖励：末端到物体水平距离
        # ------------------------------------------------------------------
        ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        reward += _W_APPROACH * np.exp(-4.0 * ee_to_obj)

        # ------------------------------------------------------------------
        # 2. 抬升奖励：物体相对 bin1 底板的 Z 高度
        # ------------------------------------------------------------------
        floor_z    = self._get_bin_floor_z()
        lift_height = max(0.0, obj_pos[2] - floor_z - self.task_cfg.obj_size)
        reward     += _W_LIFT * np.tanh(lift_height / 0.08)  # 0.08m 时饱和至 ~0.76

        # ------------------------------------------------------------------
        # 3. 运输奖励：物体到 bin2 中心的 XY 距离（抬起后才生效，避免误导）
        # ------------------------------------------------------------------
        if self._is_object_lifted():
            bin2_center = np.array([
                self.task_cfg.bin2_pos[0],
                self.task_cfg.bin2_pos[1],
                obj_pos[2],   # 忽略 Z，只考虑 XY 搬运
            ])
            obj_to_target = np.linalg.norm(obj_pos - bin2_center)
            reward += _W_TRANSPORT * np.exp(-3.0 * obj_to_target)

            # 里程碑：首次抬起时额外加成
            if not getattr(self, "_grasp_bonus_given", False):
                reward += _W_GRASP_BONUS
                self._grasp_bonus_given = True

        # ------------------------------------------------------------------
        # 4. 放置奖励：物体在 bin2 底板附近的 XY 精度（引导最终放下）
        # ------------------------------------------------------------------
        if self._is_object_in_bin2():
            bin2_xy  = self._get_bin2_center_xy()
            place_xy_err = np.linalg.norm(obj_pos[:2] - bin2_xy)
            reward  += _W_PLACE * np.exp(-5.0 * place_xy_err)

        # ------------------------------------------------------------------
        # 5. 成功稀疏奖励
        # ------------------------------------------------------------------
        if self._is_terminated():
            reward += _W_SUCCESS

        # ------------------------------------------------------------------
        # 6. 时间惩罚（鼓励高效完成）
        # ------------------------------------------------------------------
        reward += _W_TIME

        return float(reward)

    # ====================== 终止条件 ======================

    def _is_terminated(self) -> bool:
        """
        成功终止条件：物体稳定放置在 bin2 内。

        需同时满足：
            - 物体 XY 在 bin2 范围内
            - 物体 Z 在底板上方合理高度（静置而非悬空）
            - 物体速度几乎为零（防止物体飞过 bin2 时瞬间触发）
        """
        if not self._is_object_in_bin2():
            return False

        # 速度约束：物体线速度 < 0.05 m/s（防止飞过误判）
        jnt_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free_joint"
        )
        if jnt_id < 0:
            return False

        dof_adr  = self.model.jnt_dofadr[jnt_id]
        obj_vel  = self.data.qvel[dof_adr : dof_adr + 3]
        if np.linalg.norm(obj_vel) > 0.05:
            return False

        return True

    def _is_truncated(self) -> bool:
        """超时，或物体掉落桌面以下。"""
        if self.stats.episode_steps >= self.cfg.max_episode_steps:
            return True

        # 物体掉落保护：Z 低于桌面 0.1m 以下视为掉落截断
        obj_z = self._get_obj_pos()[2]
        if obj_z < self._table_height - 0.1:
            return True

        return False