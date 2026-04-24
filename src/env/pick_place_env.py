"""
抓取放置任务环境（视觉-触觉-本体感觉版本，触觉已扁平化）.

任务描述：
    机械臂+灵巧手从桌面上抓取一个立方体，将其搬运到目标位置并放置。

观测空间（扁平化 Dict，SB3 兼容）：
    - camera_rgb:      (240, 320, 3)  俯视相机 RGB 图像
    - tactile_bottom:  (5, 10, 7)     5手指 × 10行 × 7列  底部指节
    - tactile_middle:  (5, 8, 5)      5手指 × 8行 × 5列   中部指节
    - tactile_top:     (5, 6, 5)      5手指 × 6行 × 5列   顶部指节
    - proprioception:  (13,)          机械臂7DOF + 手6DOF 关节角度

动作空间（12维，OSC 6D位姿控制）：
    - 末端xyz位移(3) + 末端rpy旋转(3) + 手部6指增量(6)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

from .base_env import RobotArmEnvBase, RobotConfig

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    from .base_env import spaces

# 导入触觉传感器常量（与 grasp_task_env.py 保持一致）
from src.sensors.tactile_sensor import TactileReader, DISPLAY_ORDER, FINGER_PHALANX_ORDER


# ====================== 任务阶段枚举 ======================

class TaskPhase(IntEnum):
    REACH     = 0
    GRASP     = 1
    TRANSPORT = 2
    PLACE     = 3


# ====================== 任务配置 ======================

@dataclass
class PickPlaceConfig:
    """抓取放置任务专用配置."""

    # 物体配置
    obj_pos_range: np.ndarray = field(
        default_factory=lambda: np.array([[0.35, -0.15, 0.025],
                                          [0.50,  0.15, 0.025]])
    )
    obj_size: float = 0.025
    obj_mass: float = 0.1
    obj_color: Tuple = (0.9, 0.2, 0.1, 1.0)

    # 目标配置
    target_pos_range: np.ndarray = field(
        default_factory=lambda: np.array([[0.35, -0.15, 0.025],
                                          [0.50,  0.15, 0.025]])
    )
    target_min_dist: float = 0.15
    target_color: Tuple = (0.1, 0.8, 0.1, 0.4)

    # 运动配置
    lift_height: float = 0.15
    grasp_height_offset: float = 0.03
    approach_height: float = 0.12

    # 手部开合配置（关节角）
    hand_open:  np.ndarray = field(default_factory=lambda: np.zeros(6))
    hand_close: np.ndarray = field(default_factory=lambda: np.full(6, 0.008))

    # 阶段切换阈值
    reach_threshold: float = 0.04
    grasp_threshold: float = 0.03
    transport_threshold: float = 0.05
    place_threshold: float = 0.04

    # 奖励权重
    r_reach_scale: float = 2.0
    r_grasp_bonus: float = 10.0
    r_transport_scale: float = 3.0
    r_place_bonus: float = 50.0
    r_drop_penalty: float = -5.0
    r_step_penalty: float = -0.01
    r_collision_penalty: float = -0.5

    # 成功判定
    success_dist: float = 0.04
    success_height_max: float = 0.08


# ====================== 触觉传感器配置 ======================

# 显式指定手指顺序，不依赖字典插入顺序
_FINGER_NAMES = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]

# 统一为 (rows, cols) 即 (H, W) 格式，与 tactile_sensor.py 一致
_TACTILE_LEVELS = {
    "bottom": (10, 7),   # 10 rows, 7 cols
    "middle": (8, 5),    # 8 rows, 5 cols
    "top":    (6, 5),    # 6 rows, 5 cols
}

# 指节名称到类型的映射（用于从 DISPLAY_ORDER 推断）
_PHALANX_NAME_TO_LEVEL = {}
for finger, phalanges in FINGER_PHALANX_ORDER.items():
    for idx, name in enumerate(phalanges):
        level = {0: "bottom", 1: "middle", 2: "top"}[idx]
        _PHALANX_NAME_TO_LEVEL[name] = level


# ====================== 抓取放置环境 ======================

class PickPlaceEnv(RobotArmEnvBase):
    """
    抓取放置任务强化学习环境（视觉-触觉-本体感觉版本，SB3 兼容扁平化观测）.
    """

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[PickPlaceConfig] = None,
    ):
        self.task_cfg = task_config or PickPlaceConfig()
        super().__init__(robot_config)

        # 任务状态
        self._phase = TaskPhase.REACH
        self._obj_init_pos = np.zeros(3)
        self._target_pos = np.zeros(3)

        # 缓存 body/geom ID
        self._obj_body_id: int = -1
        self._target_site_id: int = -1
        self._obj_free_jnt_qposadr: int = -1

        # 相机渲染器（延迟初始化）
        self._renderer: Optional[mujoco.Renderer] = None
        self._cam_id: int = -1

    # ====================== 观测与动作空间 ======================

    @property
    def observation_space(self):
        """
        观测空间：扁平化 Dict（SB3 MultiInputPolicy 兼容，无嵌套 Dict）
        {
            camera_rgb:      (240, 320, 3),
            tactile_bottom:  (5, 10, 7),
            tactile_middle:  (5, 8, 5),
            tactile_top:     (5, 6, 5),
            proprioception:  (13,),
        }
        """
        img_h, img_w = 240, 320

        return spaces.Dict({
            "camera_rgb": spaces.Box(
                low=0, high=255, shape=(img_h, img_w, 3), dtype=np.uint8
            ),
            "tactile_bottom": spaces.Box(
                low=0, high=255, shape=(5, 10, 7), dtype=np.uint8
            ),
            "tactile_middle": spaces.Box(
                low=0, high=255, shape=(5, 8, 5), dtype=np.uint8
            ),
            "tactile_top": spaces.Box(
                low=0, high=255, shape=(5, 6, 5), dtype=np.uint8
            ),
            "proprioception": spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
            ),
        })

    # action_space 继承自基类（12维 osc_pose）

    # ====================== 必须实现的抽象方法 ======================

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        """向 spec 添加场景元素."""
        wb = spec.worldbody
        tc = self.task_cfg

        # 1. 灯光
        wb.add_light(
            name="top_light",
            pos=[0.2, 0.0, 1.8],
            dir=[0.0, 0.0, -1.0],
            diffuse=[0.9, 0.9, 0.9],
            ambient=[0.3, 0.3, 0.3],
        )

        # 2. 目标物体
        obj = wb.add_body(
            name="target_object",
            pos=[0.4, 0.0, tc.obj_size],
        )
        obj.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.obj_size] * 3,
            rgba=list(tc.obj_color),
            mass=tc.obj_mass,
            friction=[1.0, 0.005, 0.0001],
            condim=4,
            name="obj_geom",
        )
        obj.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="obj_free_joint")

        # 3. 目标指示器
        target_marker = wb.add_body(
            name="target_marker",
            pos=[0.5, 0.1, 0.001],
        )
        target_marker.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.obj_size * 1.2, tc.obj_size * 1.2, 0.001],
            rgba=list(tc.target_color),
            contype=0,
            conaffinity=0,
        )

        # 4. 俯视相机
        wb.add_camera(
            name="top_cam",
            pos=[0.45, 0.0, 1.2],
            quat=[1, 0, 0, 0],
            fovy=45,
        )

    def _get_obs(self) -> Dict[str, Any]:
        """
        构造观测：相机图像 + 触觉图像（扁平化键名） + 本体感觉.
        """
        # --- 1. 相机图像 ---
        camera_rgb = self._render_camera()

        # --- 2. 触觉图像（按指节类型分组，扁平化输出） ---
        tactile_grouped = self._get_tactile_grouped()

        # --- 3. 本体感觉 ---
        arm_q = self.get_arm_qpos()      # (7,)
        hand_q = self.get_hand_qpos()    # (6,)
        proprioception = np.concatenate([arm_q, hand_q]).astype(np.float32)  # (13,)

        return {
            "camera_rgb": camera_rgb,
            "tactile_bottom": tactile_grouped["bottom"],
            "tactile_middle": tactile_grouped["middle"],
            "tactile_top": tactile_grouped["top"],
            "proprioception": proprioception,
        }

    def _compute_reward(self) -> float:
        """多阶段密集奖励."""
        tc = self.task_cfg
        reward = tc.r_step_penalty

        obj_pos = self._get_obj_pos()
        ee_pos, _ = self.get_ee_pose()
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_target = np.linalg.norm(obj_pos[:2] - self._target_pos[:2])

        if self._phase == TaskPhase.REACH:
            approach_point = obj_pos + np.array([0, 0, tc.approach_height])
            dist = np.linalg.norm(ee_pos - approach_point)
            reward += tc.r_reach_scale * (1.0 - np.tanh(5 * dist))

        elif self._phase == TaskPhase.GRASP:
            reward += tc.r_reach_scale * (1.0 - np.tanh(5 * dist_ee_obj))
            tac = self._get_tactile_feature_scalar()
            tac_max = float(tac.max())
            if tac_max > 0.1:
                reward += 1.0 * tac_max

        elif self._phase == TaskPhase.TRANSPORT:
            reward += tc.r_transport_scale * (1.0 - np.tanh(5 * dist_obj_target))
            if obj_pos[2] > tc.obj_size + 0.05:
                reward += 1.0
            if obj_pos[2] < tc.obj_size * 1.5:
                reward += tc.r_drop_penalty

        elif self._phase == TaskPhase.PLACE:
            dist_full = np.linalg.norm(obj_pos - self._target_pos)
            reward += tc.r_transport_scale * (1.0 - np.tanh(8 * dist_full))

        if self._is_terminated():
            reward += tc.r_place_bonus

        return float(reward)

    def _is_terminated(self) -> bool:
        """物体落在目标位置且高度正常时判定成功."""
        tc = self.task_cfg
        obj_pos = self._get_obj_pos()
        dist_2d = np.linalg.norm(obj_pos[:2] - self._target_pos[:2])
        height_ok = obj_pos[2] < tc.success_height_max
        return bool(dist_2d < tc.success_dist and height_ok and
                    self._phase == TaskPhase.PLACE)

    def _reset_scene(self) -> None:
        """随机化物体和目标位置."""
        tc = self.task_cfg

        if self._obj_body_id == -1:
            self._cache_ids()

        # 初始化相机渲染器
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
            self._cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam"
            )

        # 随机化物体位置
        obj_pos = self._sample_pos(tc.obj_pos_range)
        self._obj_init_pos = obj_pos.copy()

        # 随机化目标
        for _ in range(100):
            target_pos = self._sample_pos(tc.target_pos_range)
            if np.linalg.norm(target_pos[:2] - obj_pos[:2]) >= tc.target_min_dist:
                break
        self._target_pos = target_pos.copy()

        # 更新物体 qpos
        if self._obj_free_jnt_qposadr >= 0:
            adr = self._obj_free_jnt_qposadr
            self.data.qpos[adr:adr+3] = obj_pos
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free_joint"
            )
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr:dof_adr+6] = 0.0

        # 重置阶段
        self._phase = TaskPhase.REACH

    def _get_info(self) -> Dict[str, Any]:
        """扩展信息."""
        info = super()._get_info()
        obj_pos = self._get_obj_pos()
        info.update({
            "phase": self._phase.name,
            "obj_pos": obj_pos.tolist(),
            "target_pos": self._target_pos.tolist(),
            "dist_obj_target": float(np.linalg.norm(obj_pos - self._target_pos)),
            "is_grasped": self._is_object_grasped(),
        })
        return info

    # ====================== 视觉与触觉观测 ======================

    def _render_camera(self) -> np.ndarray:
        """渲染俯视相机 RGB 图像."""
        if self._renderer is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        self._renderer.update_scene(self.data, camera="top_cam")
        rgb = self._renderer.render()
        return rgb

    def _get_tactile_grouped(self) -> Dict[str, np.ndarray]:
        """
        获取触觉图像，按指节类型分组.
        与 grasp_task_env.py 的显示逻辑保持一致，使用 FINGER_PHALANX_ORDER.
        """
        if self.reader is None:
            return self._empty_tactile_grouped()

        try:
            tactile_imgs = self.reader.read_image(self.data)
            if not tactile_imgs:
                return self._empty_tactile_grouped()

            result = {}
            for level, level_idx in [("bottom", 0), ("middle", 1), ("top", 2)]:
                level_images = []
                for finger in _FINGER_NAMES:
                    # 使用 FINGER_PHALANX_ORDER 获取正确的指节名称
                    # 索引: 0=bottom, 1=middle, 2=top
                    phalanx_name = FINGER_PHALANX_ORDER[finger][level_idx]

                    if phalanx_name in tactile_imgs:
                        img = tactile_imgs[phalanx_name]
                        expected_h, expected_w = _TACTILE_LEVELS[level]

                        # 统一按 (rows, cols) 即 (H, W) 处理
                        # 如果传感器返回的是 (W, H) 即 (cols, rows)，需要转置
                        if img.shape == (expected_w, expected_h):
                            img = img.T
                        elif img.shape != (expected_h, expected_w):
                            import cv2
                            img = cv2.resize(
                                img, (expected_w, expected_h),
                                interpolation=cv2.INTER_NEAREST
                            )
                        level_images.append(img.astype(np.uint8))
                    else:
                        expected_h, expected_w = _TACTILE_LEVELS[level]
                        level_images.append(
                            np.zeros((expected_h, expected_w), dtype=np.uint8)
                        )

                result[level] = np.stack(level_images, axis=0)  # (5, H, W)

            return result

        except Exception:
            return self._empty_tactile_grouped()

    def _empty_tactile_grouped(self) -> Dict[str, np.ndarray]:
        """返回全零的分组触觉图像."""
        return {
            "bottom": np.zeros((5, 10, 7), dtype=np.uint8),
            "middle": np.zeros((5, 8, 5), dtype=np.uint8),
            "top": np.zeros((5, 6, 5), dtype=np.uint8),
        }

    def _get_tactile_feature_scalar(self) -> np.ndarray:
        """
        提取标量触觉特征（用于奖励计算）.
        返回6维：每根手指的最大压力值，已归一化到 [0, 1].
        """
        if self.reader is None:
            return np.zeros(6)

        try:
            raw_data = self.reader.read_raw(self.data)
            if not raw_data:
                return np.zeros(6)

            feats = []
            for finger in _FINGER_NAMES:
                phalanges = FINGER_PHALANX_ORDER[finger]
                vals = []
                for name in phalanges:
                    if name in raw_data:
                        vals.append(float(raw_data[name].max()))
                max_force = max(vals) if vals else 0.0
                feats.append(min(max_force / TactileReader.FORCE_MAX_NEWTON, 1.0))

            return np.array(feats, dtype=np.float32)

        except Exception:
            return np.zeros(6)

    def _verify_tactile_shapes(self):
        """
        调试用：验证触觉传感器实际分辨率.
        输出应与 _TACTILE_LEVELS 一致.
        """
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
        self._update_phase()

    # ====================== 内部辅助方法 ======================

    def _update_phase(self) -> None:
        """根据当前状态自动推进任务阶段."""
        tc = self.task_cfg
        ee_pos, _ = self.get_ee_pose()
        obj_pos = self._get_obj_pos()

        if self._phase == TaskPhase.REACH:
            approach_point = obj_pos + np.array([0, 0, tc.approach_height])
            if np.linalg.norm(ee_pos - approach_point) < tc.reach_threshold:
                print(f"  [Phase] REACH -> GRASP")
                self._phase = TaskPhase.GRASP

        elif self._phase == TaskPhase.GRASP:
            if self._is_object_grasped():
                print(f"  [Phase] GRASP -> TRANSPORT")
                self._phase = TaskPhase.TRANSPORT

        elif self._phase == TaskPhase.TRANSPORT:
            horiz_dist = np.linalg.norm(obj_pos[:2] - self._target_pos[:2])
            if (horiz_dist < tc.transport_threshold and
                    obj_pos[2] > tc.obj_size + 0.05):
                print(f"  [Phase] TRANSPORT -> PLACE")
                self._phase = TaskPhase.PLACE

    def _is_object_grasped(self) -> bool:
        """判断物体是否被抓住."""
        tc = self.task_cfg
        ee_pos, _ = self.get_ee_pose()
        obj_pos = self._get_obj_pos()
        dist = np.linalg.norm(ee_pos - obj_pos)

        if dist > tc.grasp_threshold * 2:
            return False

        tac = self._get_tactile_feature_scalar()
        has_contact = tac.max() > 0.05
        return bool(dist < tc.grasp_threshold and has_contact)

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
        """缓存常用的 MuJoCo ID."""
        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object"
        )
        # 加断言，防止静默失败
        assert self._obj_body_id >= 0, (
            "target_object body not found in model. "
            "Make sure _build_scene() adds a body named 'target_object'."
        )

        jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free_joint"
        )
        if jnt_id >= 0:
            self._obj_free_jnt_qposadr = self.model.jnt_qposadr[jnt_id]

    @staticmethod
    def _sample_pos(pos_range: np.ndarray) -> np.ndarray:
        """均匀采样3D位置."""
        lo, hi = pos_range[0], pos_range[1]
        return lo + np.random.random(3) * (hi - lo)