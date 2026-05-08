"""
Door Opening 任务环境.

任务描述：
    机械臂+灵巧手转动门把手并打开门.

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
class DoorOpeningConfig:
    """Door Opening 任务专用配置."""

    obs_camera_name: str = "frontview"

    # 门配置
    door_xml: str = "assets/objects/door_lock.xml"

    # 门的位置配置（相对于桌面中心）
    door_pos: Tuple = (0.50, 0.0, 0.0)  # (x, y, z_offset_from_table) - 中心位置
    door_pos_range: Tuple = (0.08, 0.05)  # (half_x, half_y) - 随机化范围
    door_rot_deg: float = 0.0  # 门绕 Z 轴的旋转角度
    door_rot_range_deg: float = 15.0  # 门旋转随机化范围（±度）

    # 关节名称（根据 door_lock.xml 中的实际名称）
    hinge_joint_name: str = "hinge"           # 门扇铰链
    latch_joint_name: str = "latch_joint"     # 门把手/锁舌关节

    # 关键 site 名称
    handle_site_name: str = "handle"          # 门把手抓取点

    # 目标角度配置
    hinge_target_angle: float = 0.35   # 门扇目标打开角度（接近上限 0.4）
    latch_target_angle: float = -1.5   # 门把手转动角度（解锁位置）

    # 成功判定阈值
    hinge_angle_threshold: float = 0.05      # 门扇角度容差
    latch_angle_threshold: float = 0.2       # 门把手角度容差
    success_stability_steps: int = 10        # 需要稳定保持的步数

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


# ====================== Door Opening 环境 ======================


class DoorOpeningEnv(RobotArmEnvBase):
    """
    Door Opening 任务强化学习环境.
    任务：转动门把手（latch_joint）并打开门扇（hinge）.
    """

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[DoorOpeningConfig] = None,
    ):
        self.task_cfg = task_config or DoorOpeningConfig()

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

        # MuJoCo ID 缓存（-1 表示未缓存）
        self._hinge_joint_id: int = -1
        self._latch_joint_id: int = -1
        self._handle_site_id: int = -1
        self._cached_model_ptr: int = -1

        # 相机渲染器（在 _reset_scene 中按需重建）
        self._renderer: Optional[mujoco.Renderer] = None

        # 成功计数器
        self._success_count: int = 0

        # 当前门的位置和旋转（由 _reset_scene 随机化）
        self._current_door_pos = np.zeros(3)
        self._current_door_rot_deg = 0.0

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

        # 1. 从外部 XML 加载门模型并附加到场景
        door_xml_path = Path(__file__).resolve().parent.parent.parent / tc.door_xml
        if door_xml_path.exists():
            door_spec = mujoco.MjSpec.from_file(str(door_xml_path))
            door_root = door_spec.worldbody.first_body()
            if door_root is not None:
                # 递归重命名所有 body，添加前缀避免冲突
                self._rename_bodies_recursive(door_root, "door_")

                # 递归重命名所有 joint，添加前缀
                self._rename_joints_recursive(door_root, "door_")

                # 递归重命名所有 site，添加前缀
                self._rename_sites_recursive(door_root, "door_")

                # 重置根节点位置
                original_pos = np.array(door_root.pos)
                if np.linalg.norm(original_pos) > 1e-6:
                    door_root.pos = [0.0, 0.0, 0.0]

                # 计算门的位置
                # 门模型中 frame 圆柱的半高为 0.3，所以门底部在 body 原点下方 0.3
                # 需要将门抬高 0.3 使其站在桌面上
                door_bottom_offset = 0.3  # frame 圆柱半高
                door_pos = list(tc.door_pos)
                door_pos[2] = self._table_height + door_bottom_offset

                # 附加到场景
                door_frame = wb.add_frame(
                    name="door_frame",
                    pos=door_pos
                )
                # 如果有旋转，设置 frame 的旋转
                if tc.door_rot_deg != 0.0:
                    import math
                    rad = math.radians(tc.door_rot_deg)
                    # 绕 Z 轴旋转的四元数 (w, x, y, z)
                    door_frame.quat = [math.cos(rad/2), 0.0, 0.0, math.sin(rad/2)]

                attached = door_frame.attach_body(door_root, prefix="", suffix="")
            else:
                self._create_door_programmatically(wb, tc)
        else:
            self._create_door_programmatically(wb, tc)

        # 2. 相机（统一从 task_cfg.cameras 读取，不再硬编码）
        for cam in tc.cameras:
            spec.worldbody.add_camera(
                name=cam.name, mode=0, pos=list(cam.pos), quat=list(cam.quat)
            )

    def _rename_bodies_recursive(self, body, prefix: str) -> None:
        """
        递归重命名 body 及其所有子 body，添加前缀避免命名冲突.
        """
        if body.name:
            body.name = prefix + body.name

        child = body.first_body()
        while child is not None:
            self._rename_bodies_recursive(child, prefix)
            child = body.next_body(child)

    def _rename_joints_recursive(self, body, prefix: str) -> None:
        """
        递归重命名 body 及其所有子 body 中的 joint，添加前缀.
        """
        # 重命名当前 body 的 joints
        joint = body.first_joint()
        while joint is not None:
            if joint.name:
                joint.name = prefix + joint.name
            joint = body.next_joint(joint)

        # 递归处理子 body
        child = body.first_body()
        while child is not None:
            self._rename_joints_recursive(child, prefix)
            child = body.next_body(child)

    def _rename_sites_recursive(self, body, prefix: str) -> None:
        """
        递归重命名 body 及其所有子 body 中的 site，添加前缀.
        """
        # 重命名当前 body 的 sites
        site = body.first_site()
        while site is not None:
            if site.name:
                site.name = prefix + site.name
            site = body.next_site(site)

        # 递归处理子 body
        child = body.first_body()
        while child is not None:
            self._rename_sites_recursive(child, prefix)
            child = body.next_body(child)

    def _create_door_programmatically(self, wb, tc):
        """程序化创建简化门模型（当 XML 加载失败时的回退方案）."""
        import math

        # 门框（站在桌面上，半高 0.5）
        door_frame = wb.add_body(
            name="door_frame",
            pos=[tc.door_pos[0], tc.door_pos[1], self._table_height + 0.5],
        )
        door_frame.add_geom(
            name="door_frame_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.5],
            rgba=[0.5, 0.5, 0.5, 1.0],
        )

        # 门扇（可旋转）
        door_leaf = door_frame.add_body(
            name="door_leaf",
            pos=[0.0, 0.0, 0.0],
        )
        door_leaf.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            name="hinge",
            axis=[0, 0, 1],
            range=[0.0, math.pi/2],
        )
        door_leaf.add_geom(
            name="door_leaf_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.4, 0.02, 0.5],
            pos=[0.4, 0.0, 0.0],
            rgba=[0.7, 0.5, 0.3, 1.0],
        )

        # 门把手
        handle = door_leaf.add_body(
            name="handle",
            pos=[0.7, 0.03, 0.0],
        )
        handle.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            name="latch_joint",
            axis=[0, 1, 0],
            range=[-math.pi/2, math.pi/2],
        )
        handle.add_geom(
            name="handle_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.015, 0.05],
            rgba=[0.8, 0.7, 0.2, 1.0],
        )
        handle.add_site(
            name="handle",
            pos=[0.0, 0.0, 0.0],
            size=[0.02],
            rgba=[0, 0, 1, 0],
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
        tc = self.task_cfg
        reward = 0.0

        hinge_angle = self._get_hinge_angle()
        latch_angle = self._get_latch_angle()

        # 1. 把手旋转：使用负指数或平方，距离越近梯度越大，比纯线性更好收敛
        latch_diff = abs(latch_angle - tc.latch_target_angle)
        reward -= latch_diff 

        # 2. 门扇开启：基础引导
        hinge_diff = abs(hinge_angle - tc.hinge_target_angle)
        reward -= 0.5 * hinge_diff

        # 3. 阶段性加成：保持你的逻辑，但让它更清晰
        # 检查是否满足解锁阈值
        is_unlocked = abs(latch_angle - tc.latch_target_angle) < tc.latch_angle_threshold
        
        if is_unlocked:
            reward += 2.0  # 给一个显著的阶段性“小甜点”奖励
            reward -= 2.0 * hinge_diff  # 此时大幅度增加开门的权重，压过其他所有信号
        
        # 4. 成功奖励
        if self._is_task_successful():
            reward += 20.0  # 建议把成功奖励拉高，确保它覆盖掉过程中的所有负分

        # 5. 动作平滑（这是 Qwen 建议中最值得采纳的部分）
        # 假设你在 Step 中能拿到当前的 action
        # reward -= 0.01 * np.square(action).sum() 

        return reward

    def _is_terminated(self) -> bool:
        """终止条件：任务成功."""
        if self._is_task_successful():
            return True
        return False

    def _is_truncated(self) -> bool:
        """截断条件：达到最大步数等（由基类管理）."""
        return False

    def _reset_scene(self) -> None:
        """重置场景：随机化门的位置和角度，重置门把手和门扇角度."""
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

        # 重置成功计数器
        self._success_count = 0

        # 1. 随机化门的位置（水平面内）
        half_x, half_y = tc.door_pos_range
        door_cx, door_cy = tc.door_pos[0], tc.door_pos[1]
        door_pos_range = np.array(
            [
                [door_cx - half_x, door_cy - half_y, 0.0],
                [door_cx + half_x, door_cy + half_y, 0.0],
            ]
        )
        random_offset = self._sample_pos(door_pos_range)

        # 计算实际门位置（Z 由 _build_scene 中的逻辑决定）
        door_bottom_offset = 0.3  # 与 _build_scene 一致
        actual_door_pos = np.array([
            random_offset[0],
            random_offset[1],
            self._table_height + door_bottom_offset
        ])

        # 更新 door_frame 的位置（通过 qpos 或直接修改 body pos）
        # 由于 attach_body 后的 frame 位置在编译后不能直接修改
        # 我们通过修改 free joint（如果有）或重新编译来随机化
        # 简化处理：这里记录随机化后的位置，用于后续计算奖励等
        self._current_door_pos = actual_door_pos.copy()

        # 2. 随机化门的旋转角度
        rot_deg = tc.door_rot_deg + self.np_random.uniform(
            -tc.door_rot_range_deg, tc.door_rot_range_deg
        )
        self._current_door_rot_deg = rot_deg

        # 3. 随机化门把手初始角度（增加难度）
        latch_init = self.np_random.uniform(-0.1, 0.1)  # 微小随机偏移

        # 4. 随机化门扇初始角度（有时门已经微开）
        hinge_init = self.np_random.uniform(0.0, 0.05)  # 0-0.05 弧度微开

        # 重置门把手和门扇的角度（带随机化）
        if self._latch_joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[self._latch_joint_id]
            self.data.qpos[qpos_adr] = latch_init
            qvel_adr = self.model.jnt_dofadr[self._latch_joint_id]
            self.data.qvel[qvel_adr] = 0.0

        if self._hinge_joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[self._hinge_joint_id]
            self.data.qpos[qpos_adr] = hinge_init
            qvel_adr = self.model.jnt_dofadr[self._hinge_joint_id]
            self.data.qvel[qvel_adr] = 0.0

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

        # 更新成功计数器
        if self._is_task_successful():
            self._success_count += 1
        else:
            self._success_count = 0

    # ====================== 内部辅助方法 ======================

    def _get_hinge_angle(self) -> float:
        """获取门扇当前角度."""
        if self._hinge_joint_id < 0:
            self._cache_ids()
        if self._hinge_joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[self._hinge_joint_id]
            return self.data.qpos[qpos_adr]
        return 0.0

    def _get_latch_angle(self) -> float:
        """获取门把手/锁舌当前角度."""
        if self._latch_joint_id < 0:
            self._cache_ids()
        if self._latch_joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[self._latch_joint_id]
            return self.data.qpos[qpos_adr]
        return 0.0

    def _get_handle_pos(self) -> np.ndarray:
        """获取门把手抓取点的世界坐标."""
        if self._handle_site_id < 0:
            self._cache_ids()
        if self._handle_site_id >= 0:
            return self.data.site_xpos[self._handle_site_id].copy()
        return np.zeros(3)

    def _is_task_successful(self) -> bool:
        """检查任务是否成功（门把手转动到位且门已打开）."""
        tc = self.task_cfg

        hinge_angle = self._get_hinge_angle()
        latch_angle = self._get_latch_angle()

        # 检查门把手是否转动到目标角度（解锁）
        latch_success = abs(latch_angle - tc.latch_target_angle) < tc.latch_angle_threshold

        # 检查门是否打开到目标角度
        hinge_success = abs(hinge_angle - tc.hinge_target_angle) < tc.hinge_angle_threshold

        return latch_success and hinge_success

    def _is_stably_successful(self) -> bool:
        """检查任务是否稳定成功（持续多步）."""
        return self._success_count >= self.task_cfg.success_stability_steps

    def _cache_ids(self) -> None:
        """
        缓存常用 MuJoCo ID.

        注意：加载 XML 后递归重命名了所有 joint 和 site，名称格式为 "door_xxx"
        """
        tc = self.task_cfg

        # 查找门扇铰链关节
        self._hinge_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, tc.hinge_joint_name
        )
        if self._hinge_joint_id < 0:
            # 尝试带前缀的名称
            self._hinge_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"door_{tc.hinge_joint_name}"
            )

        # 查找门把手/锁舌关节
        self._latch_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, tc.latch_joint_name
        )
        if self._latch_joint_id < 0:
            self._latch_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"door_{tc.latch_joint_name}"
            )

        # 查找门把手 site
        self._handle_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, tc.handle_site_name
        )
        if self._handle_site_id < 0:
            self._handle_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"door_{tc.handle_site_name}"
            )

        # 调试输出
        if self._hinge_joint_id < 0 or self._latch_joint_id < 0:
            all_joints = [self.model.joint(i).name for i in range(self.model.njnt)]
            print(f"[DEBUG] 所有 joint 名称: {all_joints}")
            raise AssertionError(
                f"Door joints not found. Tried: {tc.hinge_joint_name}, door_{tc.hinge_joint_name}, "
                f"{tc.latch_joint_name}, door_{tc.latch_joint_name}. "
                f"Available: {[n for n in all_joints if 'hinge' in n.lower() or 'latch' in n.lower()]}"
            )

    def _sample_pos(self, pos_range: np.ndarray) -> np.ndarray:
        """在 pos_range 定义的长方体区域内均匀采样 3D 位置."""
        lo, hi = pos_range[0], pos_range[1]
        return lo + self.np_random.random(3) * (hi - lo)