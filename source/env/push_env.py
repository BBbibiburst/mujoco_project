"""
推动任务环境

任务描述：
    机械臂+灵巧手不需要抓取物体，而是用手指接触并推动圆盘/方块到桌面上的目标区域。
    这是接触丰富（contact-rich）任务，触觉传感器直接感知推力大小和接触位置。

阶段：
    APPROACH → CONTACT → PUSH（到达目标后结束）

难点：
    - 推力方向控制（施力方向不对会推偏）
    - 防止物体翻转（圆盘较稳定，方块容易翻）
    - 触觉反馈指示接触点位置

观测/动作空间与 pick_place_env 完全一致。
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

from .base_env import RobotArmEnvBase, RobotConfig

try:
    from gymnasium import spaces
except ImportError:
    from .base_env import spaces

from source.sensors.tactile_sensor import TactileReader, FINGER_PHALANX_ORDER


class PushPhase(IntEnum):
    APPROACH = 0
    CONTACT  = 1
    PUSH     = 2


@dataclass
class PushConfig:
    """推动任务配置."""

    # 物体形状：'box' 或 'cylinder'
    obj_shape: str = "cylinder"

    # 尺寸
    obj_radius: float = 0.035   # cylinder 半径
    obj_height: float = 0.015   # cylinder 高度（扁圆盘）
    obj_box_half: Tuple[float, float, float] = (0.03, 0.03, 0.015)  # box 半尺寸
    obj_mass: float = 0.15

    # 物体初始位置范围
    obj_pos_range: np.ndarray = field(
        default_factory=lambda: np.array([[0.32, -0.12, 0.015],
                                          [0.46,  0.12, 0.015]])
    )

    # 目标位置范围
    target_pos_range: np.ndarray = field(
        default_factory=lambda: np.array([[0.48, -0.15, 0.001],
                                          [0.62,  0.15, 0.001]])
    )
    target_min_dist: float = 0.15   # 物体与目标最小初始距离

    # 手部（推动时保持半开）
    hand_push: np.ndarray = field(default_factory=lambda: np.full(6, 0.004))

    # 阈值
    approach_threshold: float = 0.06   # 末端距物体侧面
    contact_threshold:  float = 0.04   # 触觉有接触
    success_dist:       float = 0.04   # 物体与目标距离

    # 奖励
    r_approach_scale: float = 1.5
    r_contact_bonus:  float = 3.0
    r_push_scale:     float = 5.0
    r_success_bonus:  float = 80.0
    r_flip_penalty:   float = -3.0    # 物体翻倒惩罚
    r_step_penalty:   float = -0.005

    # 成功判定
    success_height_min: float = 0.008   # 物体高度不低于此值（防止翻倒算成功）


_FINGER_NAMES = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
_TACTILE_LEVELS = {"bottom": (10, 7), "middle": (8, 5), "top": (6, 5)}


class PushEnv(RobotArmEnvBase):
    """用手指推动物体到目标位置的接触丰富任务环境."""

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[PushConfig] = None,
    ):
        self.task_cfg = task_config or PushConfig()
        super().__init__(robot_config)

        self._phase = PushPhase.APPROACH
        self._target_pos = np.zeros(3)

        self._obj_body_id: int = -1
        self._obj_jnt_qposadr: int = -1

        self._renderer: Optional[mujoco.Renderer] = None

    @property
    def observation_space(self):
        return spaces.Dict({
            "camera_rgb":     spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8),
            "tactile_bottom": spaces.Box(low=0, high=255, shape=(5, 10, 7),   dtype=np.uint8),
            "tactile_middle": spaces.Box(low=0, high=255, shape=(5, 8, 5),    dtype=np.uint8),
            "tactile_top":    spaces.Box(low=0, high=255, shape=(5, 6, 5),    dtype=np.uint8),
            "proprioception": spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
        })

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        wb = spec.worldbody
        tc = self.task_cfg

        wb.add_light(
            name="top_light", pos=[0.3, 0.0, 1.8],
            dir=[0.0, 0.0, -1.0],
            diffuse=[0.9, 0.9, 0.9], ambient=[0.3, 0.3, 0.3],
        )

        # 推动物体
        if tc.obj_shape == "cylinder":
            obj = wb.add_body(name="push_obj", pos=[0.40, 0.0, tc.obj_height / 2])
            obj.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[tc.obj_radius, tc.obj_height / 2],
                rgba=[0.8, 0.3, 0.8, 1.0],
                mass=tc.obj_mass,
                friction=[0.7, 0.005, 0.0001],
                condim=4,
                name="push_obj_geom",
            )
        else:  # box
            hx, hy, hz = tc.obj_box_half
            obj = wb.add_body(name="push_obj", pos=[0.40, 0.0, hz])
            obj.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[hx, hy, hz],
                rgba=[0.8, 0.3, 0.8, 1.0],
                mass=tc.obj_mass,
                friction=[0.7, 0.005, 0.0001],
                condim=4,
                name="push_obj_geom",
            )
        obj.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="push_obj_free_joint")

        # 目标区域标记（半透明圆形）
        target_vis = wb.add_body(name="push_target_vis", pos=[0.55, 0.0, 0.001])
        target_vis.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[tc.obj_radius * 1.5, 0.001],
            rgba=[0.1, 0.9, 0.3, 0.35],
            contype=0, conaffinity=0,
            name="push_target_geom",
        )

        # 俯视相机（推动任务更关注顶视图）
        wb.add_camera(
            name="top_cam",
            pos=[0.47, 0.0, 1.1],
            quat=[1, 0, 0, 0],
            fovy=50,
        )

    def _reset_scene(self) -> None:
        tc = self.task_cfg

        if self._obj_body_id == -1:
            self._cache_ids()

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)

        # 随机化目标位置
        t_lo, t_hi = tc.target_pos_range[0], tc.target_pos_range[1]
        for _ in range(100):
            t_pos = t_lo + np.random.random(3) * (t_hi - t_lo)
            t_pos[2] = 0.001

            # 随机化物体位置，保证距目标足够远
            o_lo, o_hi = tc.obj_pos_range[0], tc.obj_pos_range[1]
            obj_z = tc.obj_height / 2 if tc.obj_shape == "cylinder" else tc.obj_box_half[2]
            o_pos = o_lo + np.random.random(3) * (o_hi - o_lo)
            o_pos[2] = obj_z

            if np.linalg.norm(o_pos[:2] - t_pos[:2]) >= tc.target_min_dist:
                break

        self._target_pos = t_pos.copy()

        # 更新 target_vis body 位置
        vis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "push_target_vis")
        if vis_id >= 0:
            self.model.body_pos[vis_id] = t_pos

        # 重置物体 qpos
        if self._obj_jnt_qposadr >= 0:
            adr = self._obj_jnt_qposadr
            self.data.qpos[adr:adr+3] = o_pos
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "push_obj_free_joint"
            )
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr:dof_adr+6] = 0.0

        self._phase = PushPhase.APPROACH
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> Dict[str, Any]:
        tac = self._get_tactile_grouped()
        arm_q = self.get_arm_qpos()
        hand_q = self.get_hand_qpos()
        return {
            "camera_rgb":     self._render_camera(),
            "tactile_bottom": tac["bottom"],
            "tactile_middle": tac["middle"],
            "tactile_top":    tac["top"],
            "proprioception": np.concatenate([arm_q, hand_q]).astype(np.float32),
        }

    def _compute_reward(self) -> float:
        tc = self.task_cfg
        reward = tc.r_step_penalty

        ee_pos, _ = self.get_ee_pose()
        obj_pos = self._get_obj_pos()
        dist_obj_target = np.linalg.norm(obj_pos[:2] - self._target_pos[:2])

        # 翻倒惩罚
        if self._is_obj_flipped():
            reward += tc.r_flip_penalty

        if self._phase == PushPhase.APPROACH:
            # 引导末端靠近物体侧面（在物体和目标连线的反方向）
            push_dir = self._target_pos[:2] - obj_pos[:2]
            norm = np.linalg.norm(push_dir)
            if norm > 1e-6:
                push_dir = push_dir / norm
            approach_pt = obj_pos[:2] - push_dir * (tc.obj_radius + 0.04)
            approach_3d = np.array([approach_pt[0], approach_pt[1], obj_pos[2]])
            dist_to_approach = np.linalg.norm(ee_pos - approach_3d)
            reward += tc.r_approach_scale * (1.0 - np.tanh(5 * dist_to_approach))

        elif self._phase == PushPhase.CONTACT:
            tac = self._get_tactile_scalar()
            if tac.max() > 0.05:
                reward += tc.r_contact_bonus
            reward += tc.r_push_scale * 0.5 * (1.0 - np.tanh(3 * dist_obj_target))

        elif self._phase == PushPhase.PUSH:
            reward += tc.r_push_scale * (1.0 - np.tanh(3 * dist_obj_target))
            if self._is_success():
                reward += tc.r_success_bonus

        return float(reward)

    def _is_terminated(self) -> bool:
        return self._is_success()

    def _apply_action(self, action: np.ndarray) -> None:
        super()._apply_action(action)
        self._update_phase()

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        obj_pos = self._get_obj_pos()
        tac = self._get_tactile_scalar()
        info.update({
            "phase": self._phase.name,
            "obj_pos": obj_pos.tolist(),
            "target_pos": self._target_pos.tolist(),
            "dist_obj_target": float(np.linalg.norm(obj_pos[:2] - self._target_pos[:2])),
            "tactile_max": float(tac.max()),
            "is_flipped": self._is_obj_flipped(),
            "is_success": self._is_success(),
        })
        return info

    # ====================== 内部辅助 ======================

    def _update_phase(self) -> None:
        tc = self.task_cfg
        ee_pos, _ = self.get_ee_pose()
        obj_pos = self._get_obj_pos()

        if self._phase == PushPhase.APPROACH:
            dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
            if dist_ee_obj < tc.approach_threshold:
                self._phase = PushPhase.CONTACT

        elif self._phase == PushPhase.CONTACT:
            tac = self._get_tactile_scalar()
            if tac.max() > 0.05:
                self._phase = PushPhase.PUSH

    def _is_success(self) -> bool:
        tc = self.task_cfg
        obj_pos = self._get_obj_pos()
        dist = np.linalg.norm(obj_pos[:2] - self._target_pos[:2])
        return dist < tc.success_dist and not self._is_obj_flipped()

    def _is_obj_flipped(self) -> bool:
        """判断物体是否翻倒（通过高度判断）."""
        tc = self.task_cfg
        obj_pos = self._get_obj_pos()
        expected_z = tc.obj_height / 2 if tc.obj_shape == "cylinder" else tc.obj_box_half[2]
        return obj_pos[2] < tc.success_height_min or obj_pos[2] > expected_z * 3

    def _get_obj_pos(self) -> np.ndarray:
        if self._obj_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._obj_body_id].copy()

    def _cache_ids(self) -> None:
        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "push_obj"
        )
        jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "push_obj_free_joint"
        )
        if jnt_id >= 0:
            self._obj_jnt_qposadr = self.model.jnt_qposadr[jnt_id]

    def _get_tactile_scalar(self) -> np.ndarray:
        if self.reader is None:
            return np.zeros(6)
        try:
            raw = self.reader.read_raw(self.data)
            if not raw:
                return np.zeros(6)
            feats = []
            for finger in _FINGER_NAMES:
                vals = [float(raw[n].max()) for n in FINGER_PHALANX_ORDER[finger] if n in raw]
                feats.append(min(max(vals) / TactileReader.FORCE_MAX_NEWTON, 1.0) if vals else 0.0)
            return np.array(feats, dtype=np.float32)
        except Exception:
            return np.zeros(6)

    def _get_tactile_grouped(self) -> Dict[str, np.ndarray]:
        if self.reader is None:
            return self._empty_tactile()
        try:
            imgs = self.reader.read_image(self.data)
            if not imgs:
                return self._empty_tactile()
            result = {}
            for level, level_idx in [("bottom", 0), ("middle", 1), ("top", 2)]:
                rows = []
                for finger in _FINGER_NAMES:
                    phalanx = FINGER_PHALANX_ORDER[finger][level_idx]
                    exp_h, exp_w = _TACTILE_LEVELS[level]
                    img = imgs.get(phalanx)
                    if img is not None:
                        if img.shape == (exp_w, exp_h):
                            img = img.T
                        rows.append(img.astype(np.uint8))
                    else:
                        rows.append(np.zeros((exp_h, exp_w), dtype=np.uint8))
                result[level] = np.stack(rows, axis=0)
            return result
        except Exception:
            return self._empty_tactile()

    def _empty_tactile(self) -> Dict[str, np.ndarray]:
        return {
            "bottom": np.zeros((5, 10, 7), dtype=np.uint8),
            "middle": np.zeros((5, 8, 5),  dtype=np.uint8),
            "top":    np.zeros((5, 6, 5),  dtype=np.uint8),
        }

    def _render_camera(self) -> np.ndarray:
        if self._renderer is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        self._renderer.update_scene(self.data, camera="top_cam")
        return self._renderer.render()
