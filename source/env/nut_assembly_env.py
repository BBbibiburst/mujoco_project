"""
Nut Assembly 任务环境.

任务描述：
    机械臂+灵巧手将桌面上的螺母（nut）装配到对应的插销（peg）上。
    方形螺母装配到方形插销，圆形螺母装配到圆形插销。

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


# ====================== 枚举：螺母类型 ======================


class NutType(IntEnum):
    """螺母类型."""

    ROUND = 0
    SQUARE = 1


# ====================== 任务配置 ======================


@dataclass
class NutAssemblyConfig:
    """Nut Assembly 任务专用配置."""

    obs_camera_name: str = "frontview"

    # 螺母配置
    nut_size: float = 0.02
    nut_mass: float = 0.05

    # 插销配置
    peg_height: float = 0.1
    peg_radius: float = 0.016
    peg_mass: float = 0.1

    # 颜色配置
    round_nut_color: Tuple = (0.9, 0.5, 0.1, 1.0)    # 橙色 - 圆形螺母
    square_nut_color: Tuple = (0.2, 0.7, 0.3, 1.0)    # 绿色 - 方形螺母
    round_peg_color: Tuple = (0.9, 0.7, 0.3, 1.0)      # 浅橙色 - 圆形插销
    square_peg_color: Tuple = (0.3, 0.8, 0.5, 1.0)      # 浅绿色 - 方形插销

    # 螺母 XML 路径（相对于项目根目录）
    round_nut_xml: str = "assets/objects/round-nut.xml"
    square_nut_xml: str = "assets/objects/square-nut.xml"

    # 插销位置（固定在桌面上）
    round_peg_pos: Tuple = (0.40, -0.15, 0.0)   # (x, y, z_offset_from_table)
    square_peg_pos: Tuple = (0.40, 0.15, 0.0)

    # 螺母初始放置区域配置
    nut_spawn_range: Tuple = (0.08, 0.08)  # (half_x, half_y)
    round_nut_spawn_center: Tuple = (0.50, -0.15)
    square_nut_spawn_center: Tuple = (0.50, 0.15)

    # 装配成功判定阈值
    assembly_threshold: float = 0.015  # 螺母中心到插销中心的水平距离阈值
    assembly_height_threshold: float = 0.01  # 螺母高度相对于插销顶部的容差
    assembly_stability_steps: int = 10  # 需要稳定保持的步数

    # 目标标记颜色
    target_color: Tuple = (0.1, 0.8, 0.1, 0.3)

    # 运动配置
    approach_height: float = 0.15
    grasp_height_offset: float = 0.02

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


# ====================== Nut Assembly 环境 ======================


class NutAssemblyEnv(RobotArmEnvBase):
    """
    Nut Assembly 任务强化学习环境.
    任务：将 round_nut 装配到 round_peg，将 square_nut 装配到 square_peg.
    """

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[NutAssemblyConfig] = None,
    ):
        self.task_cfg = task_config or NutAssemblyConfig()

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

        self._round_nut_init_pos = np.zeros(3)
        self._square_nut_init_pos = np.zeros(3)

        # MuJoCo ID 缓存（-1 表示未缓存）
        self._round_nut_body_id: int = -1
        self._square_nut_body_id: int = -1
        self._round_peg_body_id: int = -1
        self._square_peg_body_id: int = -1
        self._round_nut_jnt_qposadr: int = -1
        self._square_nut_jnt_qposadr: int = -1
        self._cached_model_ptr: int = -1

        # 相机渲染器（在 _reset_scene 中按需重建）
        self._renderer: Optional[mujoco.Renderer] = None

        # 装配成功计数器（用于稳定性检测）
        self._round_assembly_count: int = 0
        self._square_assembly_count: int = 0

    # ====================== 高度计算======================

    def _compute_nut_z(self) -> float:
        """
        计算螺母中心的世界坐标系 Z 高度.

        螺母放在桌面上，底部贴桌面，中心高度 = 桌面高度 + 螺母半高

        Returns:
            nut_z: 螺母中心的世界 Z
        """
        return self._table_height + self.task_cfg.nut_size * 0.5

    def _compute_peg_top_z(self) -> float:
        """
        计算插销顶部的世界坐标系 Z 高度.

        Returns:
            peg_top_z: 插销顶部的世界 Z
        """
        return self._table_height + self.task_cfg.peg_height

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

        nut_z = self._compute_nut_z()
        peg_top_z = self._compute_peg_top_z()

        # 1. 圆形插销（固定在桌面上）
        round_peg = wb.add_body(
            name="round_peg",
            pos=[tc.round_peg_pos[0], tc.round_peg_pos[1], self._table_height + tc.peg_height * 0.5],
        )
        round_peg.add_geom(
            name="round_peg_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[tc.peg_radius, tc.peg_height * 0.5],
            rgba=list(tc.round_peg_color),
            mass=tc.peg_mass,
            friction=[1.0, 0.005, 0.0001],
        )

        # 2. 方形插销（固定在桌面上）
        square_peg = wb.add_body(
            name="square_peg",
            pos=[tc.square_peg_pos[0], tc.square_peg_pos[1], self._table_height + tc.peg_height * 0.5],
        )
        square_peg.add_geom(
            name="square_peg_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.peg_radius, tc.peg_radius, tc.peg_height * 0.5],
            rgba=list(tc.square_peg_color),
            mass=tc.peg_mass,
            friction=[1.0, 0.005, 0.0001],
        )

        # 3. 圆形螺母 - 从外部 XML 加载并附加到场景
        # 策略：加载 XML 后，递归重命名所有 body 添加前缀，避免命名冲突
        round_nut_xml_path = Path(__file__).resolve().parent.parent.parent / tc.round_nut_xml
        if round_nut_xml_path.exists():
            round_nut_spec = mujoco.MjSpec.from_file(str(round_nut_xml_path))
            round_nut_root = round_nut_spec.worldbody.first_body()
            if round_nut_root is not None:
                # 递归重命名所有 body，添加前缀
                self._rename_bodies_recursive(round_nut_root, "round_nut_")

                # 重置根节点位置
                original_pos = np.array(round_nut_root.pos)
                if np.linalg.norm(original_pos) > 1e-6:
                    round_nut_root.pos = [0.0, 0.0, 0.0]

                # 附加到场景
                round_nut_frame = wb.add_frame(
                    name="round_nut_frame",
                    pos=[tc.round_nut_spawn_center[0], tc.round_nut_spawn_center[1], nut_z]
                )
                attached = round_nut_frame.attach_body(round_nut_root, prefix="", suffix="")
                # 添加自由关节使螺母可以移动（XML 中没有关节）
                attached.add_joint(
                    type=mujoco.mjtJoint.mjJNT_FREE, 
                    name="round_nut_joint"
                )
            else:
                self._create_round_nut_programmatically(wb, tc, nut_z)
        else:
            self._create_round_nut_programmatically(wb, tc, nut_z)

        # 4. 方形螺母 - 从外部 XML 加载并附加到场景
        square_nut_xml_path = Path(__file__).resolve().parent.parent.parent / tc.square_nut_xml
        if square_nut_xml_path.exists():
            square_nut_spec = mujoco.MjSpec.from_file(str(square_nut_xml_path))
            square_nut_root = square_nut_spec.worldbody.first_body()
            if square_nut_root is not None:
                # 递归重命名所有 body，添加前缀
                self._rename_bodies_recursive(square_nut_root, "square_nut_")

                # 重置根节点位置
                original_pos = np.array(square_nut_root.pos)
                if np.linalg.norm(original_pos) > 1e-6:
                    square_nut_root.pos = [0.0, 0.0, 0.0]

                # 附加到场景
                square_nut_frame = wb.add_frame(
                    name="square_nut_frame",
                    pos=[tc.square_nut_spawn_center[0], tc.square_nut_spawn_center[1], nut_z]
                )
                attached = square_nut_frame.attach_body(square_nut_root, prefix="", suffix="")
                attached.add_joint(
                    type=mujoco.mjtJoint.mjJNT_FREE, 
                    name="square_nut_joint"
                )
            else:
                self._create_square_nut_programmatically(wb, tc, nut_z)
        else:
            self._create_square_nut_programmatically(wb, tc, nut_z)

        # 5. 目标位置 markers（在插销顶部，可视化目标装配位置）
        # 圆形插销目标
        round_target = wb.add_body(
            name="round_target_marker",
            pos=[tc.round_peg_pos[0], tc.round_peg_pos[1], peg_top_z + tc.nut_size * 0.3],
        )
        round_target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[tc.nut_size * 0.55, 0.001],
            rgba=list(tc.target_color),
            contype=0,
            conaffinity=0,
        )

        # 方形插销目标
        square_target = wb.add_body(
            name="square_target_marker",
            pos=[tc.square_peg_pos[0], tc.square_peg_pos[1], peg_top_z + tc.nut_size * 0.3],
        )
        square_target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.nut_size * 0.55, tc.nut_size * 0.55, 0.001],
            rgba=list(tc.target_color),
            contype=0,
            conaffinity=0,
        )

        # 6. 相机（统一从 task_cfg.cameras 读取，不再硬编码）
        for cam in tc.cameras:
            spec.worldbody.add_camera(
                name=cam.name, mode=0, pos=list(cam.pos), quat=list(cam.quat)
            )

    def _rename_bodies_recursive(self, body, prefix: str) -> None:
        """
        递归重命名 body 及其所有子 body，添加前缀避免命名冲突.

        Args:
            body: 当前 body 节点
            prefix: 要添加的前缀
        """
        if body.name:
            body.name = prefix + body.name

        # 递归处理子 body
        child = body.first_body()
        while child is not None:
            self._rename_bodies_recursive(child, prefix)
            child = child.next_body(child)

    def _create_round_nut_programmatically(self, wb, tc, nut_z):
        """程序化创建圆形螺母（当 XML 加载失败时的回退方案）."""
        round_nut = wb.add_body(
            name="round_nut",
            pos=[tc.round_nut_spawn_center[0], tc.round_nut_spawn_center[1], nut_z],
        )
        # 外圆柱（螺母主体）
        round_nut.add_geom(
            name="round_nut_outer",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[tc.nut_size * 0.5, tc.nut_size * 0.3],
            rgba=list(tc.round_nut_color),
            mass=tc.nut_mass,
            friction=[1.0, 0.005, 0.0001],
        )
        # 添加自由关节以允许螺母移动
        round_nut.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="round_nut_joint")

    def _create_square_nut_programmatically(self, wb, tc, nut_z):
        """程序化创建方形螺母（当 XML 加载失败时的回退方案）."""
        square_nut = wb.add_body(
            name="square_nut",
            pos=[tc.square_nut_spawn_center[0], tc.square_nut_spawn_center[1], nut_z],
        )
        square_nut.add_geom(
            name="square_nut_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.nut_size * 0.5, tc.nut_size * 0.5, tc.nut_size * 0.3],
            rgba=list(tc.square_nut_color),
            mass=tc.nut_mass,
            friction=[1.0, 0.005, 0.0001],
        )
        square_nut.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="square_nut_joint")

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
        """计算奖励：鼓励螺母装配到对应插销上."""
        tc = self.task_cfg
        reward = 0.0

        # 圆形螺母奖励
        round_nut_pos = self._get_round_nut_pos()
        round_peg_pos = self._get_round_peg_pos()
        round_horizontal_dist = np.linalg.norm(round_nut_pos[:2] - round_peg_pos[:2])
        round_height_diff = abs(round_nut_pos[2] - (self._compute_peg_top_z() + tc.nut_size * 0.3))

        # 水平接近奖励
        reward += -round_horizontal_dist
        # 高度接近奖励
        reward += -0.5 * round_height_diff
        # 成功装配奖励
        if self._is_round_assembled():
            reward += 5.0

        # 方形螺母奖励
        square_nut_pos = self._get_square_nut_pos()
        square_peg_pos = self._get_square_peg_pos()
        square_horizontal_dist = np.linalg.norm(square_nut_pos[:2] - square_peg_pos[:2])
        square_height_diff = abs(square_nut_pos[2] - (self._compute_peg_top_z() + tc.nut_size * 0.3))

        reward += -square_horizontal_dist
        reward += -0.5 * square_height_diff
        if self._is_square_assembled():
            reward += 5.0

        return reward

    def _is_terminated(self) -> bool:
        """终止条件：螺母掉落桌面以下，或两个螺母都成功装配."""
        round_nut_pos = self._get_round_nut_pos()
        square_nut_pos = self._get_square_nut_pos()

        # 任一螺母掉落到桌面以下
        if round_nut_pos[2] < self._table_height - 0.05 or square_nut_pos[2] < self._table_height - 0.05:
            return True

        # 两个螺母都成功装配
        if self._is_round_assembled() and self._is_square_assembled():
            return True

        return False

    def _is_truncated(self) -> bool:
        """截断条件：达到最大步数等（由基类管理）."""
        return False

    def _reset_scene(self) -> None:
        """随机化螺母位置，重建缓存和渲染器."""
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

        # 重置装配计数器
        self._round_assembly_count = 0
        self._square_assembly_count = 0

        nut_z = self._compute_nut_z()
        half_x, half_y = tc.nut_spawn_range

        # 圆形螺母采样范围
        round_cx, round_cy = tc.round_nut_spawn_center
        round_pos_range = np.array(
            [
                [round_cx - half_x, round_cy - half_y, nut_z],
                [round_cx + half_x, round_cy + half_y, nut_z],
            ]
        )

        # 方形螺母采样范围
        square_cx, square_cy = tc.square_nut_spawn_center
        square_pos_range = np.array(
            [
                [square_cx - half_x, square_cy - half_y, nut_z],
                [square_cx + half_x, square_cy + half_y, nut_z],
            ]
        )

        # 采样位置
        round_pos = self._sample_pos(round_pos_range)
        square_pos = self._sample_pos(square_pos_range)

        self._round_nut_init_pos = round_pos.copy()
        self._square_nut_init_pos = square_pos.copy()

        # 更新圆形螺母 qpos
        if self._round_nut_jnt_qposadr >= 0:
            adr = self._round_nut_jnt_qposadr
            self.data.qpos[adr : adr + 3] = round_pos
            self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "round_nut_joint")
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0

        # 更新方形螺母 qpos
        if self._square_nut_jnt_qposadr >= 0:
            adr = self._square_nut_jnt_qposadr
            self.data.qpos[adr : adr + 3] = square_pos
            self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "square_nut_joint")
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

        # 更新装配稳定性计数器
        if self._is_round_assembled():
            self._round_assembly_count += 1
        else:
            self._round_assembly_count = 0

        if self._is_square_assembled():
            self._square_assembly_count += 1
        else:
            self._square_assembly_count = 0

    # ====================== 内部辅助方法 ======================

    def _get_round_nut_pos(self) -> np.ndarray:
        """获取圆形螺母质心位置."""
        if self._round_nut_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._round_nut_body_id].copy()

    def _get_square_nut_pos(self) -> np.ndarray:
        """获取方形螺母质心位置."""
        if self._square_nut_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._square_nut_body_id].copy()

    def _get_round_peg_pos(self) -> np.ndarray:
        """获取圆形插销位置."""
        if self._round_peg_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._round_peg_body_id].copy()

    def _get_square_peg_pos(self) -> np.ndarray:
        """获取方形插销位置."""
        if self._square_peg_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._square_peg_body_id].copy()

    def _is_round_assembled(self) -> bool:
        """检查圆形螺母是否成功装配到圆形插销上."""
        tc = self.task_cfg
        nut_pos = self._get_round_nut_pos()
        peg_pos = self._get_round_peg_pos()

        # 水平距离检查
        horizontal_dist = np.linalg.norm(nut_pos[:2] - peg_pos[:2])
        if horizontal_dist > tc.assembly_threshold:
            return False

        # 高度检查（螺母应该在插销顶部附近）
        target_z = self._compute_peg_top_z() + tc.nut_size * 0.3
        if abs(nut_pos[2] - target_z) > tc.assembly_height_threshold:
            return False

        # 稳定性检查
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "round_nut_joint")
        if jnt_id >= 0:
            dof_adr = self.model.jnt_dofadr[jnt_id]
            nut_vel = self.data.qvel[dof_adr : dof_adr + 6]
            if np.linalg.norm(nut_vel) > 0.05:
                return False

        return True

    def _is_square_assembled(self) -> bool:
        """检查方形螺母是否成功装配到方形插销上."""
        tc = self.task_cfg
        nut_pos = self._get_square_nut_pos()
        peg_pos = self._get_square_peg_pos()

        # 水平距离检查
        horizontal_dist = np.linalg.norm(nut_pos[:2] - peg_pos[:2])
        if horizontal_dist > tc.assembly_threshold:
            return False

        # 高度检查
        target_z = self._compute_peg_top_z() + tc.nut_size * 0.3
        if abs(nut_pos[2] - target_z) > tc.assembly_height_threshold:
            return False

        # 稳定性检查
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "square_nut_joint")
        if jnt_id >= 0:
            dof_adr = self.model.jnt_dofadr[jnt_id]
            nut_vel = self.data.qvel[dof_adr : dof_adr + 6]
            if np.linalg.norm(nut_vel) > 0.05:
                return False

        return True

    def _is_stably_assembled(self, nut_type: NutType) -> bool:
        """
        检查螺母是否稳定装配（持续多步）.

        Args:
            nut_type: 螺母类型（ROUND 或 SQUARE）

        Returns:
            bool: 是否稳定装配
        """
        tc = self.task_cfg
        if nut_type == NutType.ROUND:
            return self._round_assembly_count >= tc.assembly_stability_steps
        else:
            return self._square_assembly_count >= tc.assembly_stability_steps

    def _cache_ids(self) -> None:
        """
        缓存常用 MuJoCo ID.

        注意：加载 XML 后递归重命名了所有 body，名称格式为 "round_nut_object" 和 "square_nut_object"
        """
        # 圆形螺母 body（重命名后为 "round_nut_object"）
        self._round_nut_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "round_nut_object"
        )
        if self._round_nut_body_id < 0:
            # 尝试其他可能的名称
            all_bodies = [self.model.body(i).name for i in range(self.model.nbody)]
            print(f"[DEBUG] 所有 body 名称: {all_bodies}")
            raise AssertionError(
                "round_nut body not found. Tried: round_nut_object. "
                f"Available: {str([n for n in all_bodies if 'nut' in n.lower() or 'object' in n.lower()])}"
            )

        # 方形螺母 body（重命名后为 "square_nut_object"）
        self._square_nut_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "square_nut_object"
        )
        if self._square_nut_body_id < 0:
            all_bodies = [self.model.body(i).name for i in range(self.model.nbody)]
            raise AssertionError(
                "square_nut body not found. Tried: square_nut_object. "
                f"Available: {str([n for n in all_bodies if 'nut' in n.lower() or 'object' in n.lower()])}"
            )

        self._round_peg_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "round_peg"
        )
        assert self._round_peg_body_id >= 0, "round_peg body not found"

        self._square_peg_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "square_peg"
        )
        assert self._square_peg_body_id >= 0, "square_peg body not found"

        # 查找关节
        round_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "round_nut_joint"
        )
        self._round_nut_jnt_qposadr = (
            self.model.jnt_qposadr[round_jnt_id] if round_jnt_id >= 0 else -1
        )

        square_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "square_nut_joint"
        )
        self._square_nut_jnt_qposadr = (
            self.model.jnt_qposadr[square_jnt_id] if square_jnt_id >= 0 else -1
        )

    def _sample_pos(self, pos_range: np.ndarray) -> np.ndarray:
        """在 pos_range 定义的长方体区域内均匀采样 3D 位置."""
        lo, hi = pos_range[0], pos_range[1]
        return lo + self.np_random.random(3) * (hi - lo)