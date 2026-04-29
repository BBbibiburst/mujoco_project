"""
重定向任务环境

任务描述：
    机械臂+灵巧手抓取桌面上的长方体物块，将其旋转到指定目标姿态后放回桌面原位。
    纯姿态控制任务，位置不需要大幅移动，重点考察灵巧手的姿态调整能力。

难度设计：
    - 目标姿态随机（绕 Z 轴旋转 0~180°，绕 X/Y 轴小幅随机倾斜）
    - 需要感知当前物体姿态（通过相机图像隐式推断）
    - 触觉反馈用于感知抓握稳定性，防止旋转过程中掉落

阶段：
    REACH → GRASP → REORIENT → PLACE

奖励结构：
    REACH：接近奖励
    GRASP：抓取成功奖励
    REORIENT：姿态误差奖励（四元数距离越小越好）
    PLACE：放置成功且姿态达标奖励

观测/动作空间与 pick_place_env 完全一致。
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import mujoco
import numpy as np
from .base_env import RobotArmEnvBase, RobotConfig
from gymnasium import spaces
from src.sensors.tactile_sensor import TactileReader, FINGER_PHALANX_ORDER


class ReorientPhase(IntEnum):
    REACH    = 0
    GRASP    = 1
    REORIENT = 2
    PLACE    = 3


@dataclass
class ReorientConfig:
    """重定向任务配置."""

    # 物块尺寸（非正方体，便于感知姿态）
    obj_half_size: Tuple[float, float, float] = (0.04, 0.02, 0.015)
    obj_mass: float = 0.12

    # 初始位置范围
    obj_pos_range: np.ndarray = field(
        default_factory=lambda: np.array([[0.35, -0.10, 0.015],
                                          [0.50,  0.10, 0.015]])
    )

    # 目标姿态：绕 Z 轴随机旋转角度范围（度）
    target_yaw_range: Tuple[float, float] = (30.0, 150.0)
    # 绕 X/Y 轴的随机小幅倾斜（度），增加难度
    target_pitch_roll_std: float = 10.0

    # 手部
    hand_open:  np.ndarray = field(default_factory=lambda: np.zeros(6))
    hand_close: np.ndarray = field(default_factory=lambda: np.full(6, 0.008))

    # 阈值
    reach_threshold:   float = 0.05
    grasp_threshold:   float = 0.04
    orient_threshold:  float = 0.15   # 四元数距离（rad，约 8.6°）
    place_height_max:  float = 0.06   # 放置时物块高度不超过此值认为已落地

    # 奖励
    r_reach_scale:    float = 1.5
    r_grasp_bonus:    float = 8.0
    r_orient_scale:   float = 4.0
    r_place_bonus:    float = 80.0
    r_drop_penalty:   float = -5.0
    r_step_penalty:   float = -0.01

    # 成功判定
    success_orient_tol: float = 0.12   # 四元数距离容限


_FINGER_NAMES = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
_TACTILE_LEVELS = {"bottom": (10, 7), "middle": (8, 5), "top": (6, 5)}


class ReorientEnv(RobotArmEnvBase):
    """抓取物块并旋转到目标姿态的重定向任务环境."""

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[ReorientConfig] = None,
    ):
        self.task_cfg = task_config or ReorientConfig()
        super().__init__(robot_config)

        self._phase = ReorientPhase.REACH
        self._obj_init_pos = np.zeros(3)
        self._target_quat = np.array([1., 0., 0., 0.])   # w, x, y, z

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
        hx, hy, hz = tc.obj_half_size

        wb.add_light(
            name="top_light", pos=[0.3, 0.0, 1.8],
            dir=[0.0, 0.0, -1.0],
            diffuse=[0.9, 0.9, 0.9], ambient=[0.35, 0.35, 0.35],
        )

        # 目标物块（非正方体长方体，颜色：蓝色）
        obj = wb.add_body(name="obj_reorient", pos=[0.42, 0.0, hz])
        obj.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hx, hy, hz],
            rgba=[0.15, 0.35, 0.9, 1.0],
            mass=tc.obj_mass,
            friction=[1.0, 0.005, 0.0001],
            condim=4,
            name="obj_reorient_geom",
        )
        obj.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="obj_reorient_free_joint")

        # 目标姿态指示器：半透明轮廓，动态更新
        # 用一个固定 site 标记目标姿态（可视化用）
        target_vis = wb.add_body(name="reorient_target_vis", pos=[0.42, 0.0, hz])
        target_vis.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hx * 1.1, hy * 1.1, hz * 1.1],
            rgba=[0.9, 0.8, 0.1, 0.25],
            contype=0, conaffinity=0,
            name="reorient_target_geom",
        )

        # 俯视 + 侧视双相机（侧视有助于感知倾斜姿态）
        wb.add_camera(
            name="top_cam",
            pos=[0.42, 0.0, 1.2],
            quat=[1, 0, 0, 0],
            fovy=45,
        )

    def _reset_scene(self) -> None:
        tc = self.task_cfg

        if self._obj_body_id == -1:
            self._cache_ids()

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)

        # 随机化物块位置
        lo, hi = tc.obj_pos_range[0], tc.obj_pos_range[1]
        obj_pos = lo + np.random.random(3) * (hi - lo)
        obj_pos[2] = tc.obj_half_size[2]
        self._obj_init_pos = obj_pos.copy()

        # 随机化目标姿态
        yaw_lo, yaw_hi = tc.target_yaw_range
        yaw = np.radians(np.random.uniform(yaw_lo, yaw_hi))
        pitch = np.radians(np.random.normal(0, tc.target_pitch_roll_std))
        roll  = np.radians(np.random.normal(0, tc.target_pitch_roll_std))
        self._target_quat = self._euler_to_quat(roll, pitch, yaw)

        # 更新目标可视化 body 的位置和朝向
        vis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "reorient_target_vis")
        if vis_id >= 0:
            self.model.body_pos[vis_id] = obj_pos
            self.model.body_quat[vis_id] = self._target_quat

        # 重置物块 qpos（初始姿态为单位四元数）
        if self._obj_jnt_qposadr >= 0:
            adr = self._obj_jnt_qposadr
            self.data.qpos[adr:adr+3] = obj_pos
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_reorient_free_joint"
            )
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr:dof_adr+6] = 0.0

        self._phase = ReorientPhase.REACH
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
        obj_quat = self._get_obj_quat()
        orient_err = self._quat_distance(obj_quat, self._target_quat)

        if self._phase == ReorientPhase.REACH:
            above = obj_pos + np.array([0, 0, 0.12])
            dist = np.linalg.norm(ee_pos - above)
            reward += tc.r_reach_scale * (1.0 - np.tanh(5 * dist))

        elif self._phase == ReorientPhase.GRASP:
            if self._is_grasped():
                reward += tc.r_grasp_bonus
            reward -= orient_err * 0.5   # 轻微姿态感知

        elif self._phase == ReorientPhase.REORIENT:
            reward += tc.r_orient_scale * (1.0 - np.tanh(3 * orient_err))
            if not self._is_grasped():
                reward += tc.r_drop_penalty

        elif self._phase == ReorientPhase.PLACE:
            if self._is_success():
                reward += tc.r_place_bonus
            reward += tc.r_orient_scale * (1.0 - np.tanh(3 * orient_err))

        return float(reward)

    def _is_terminated(self) -> bool:
        return self._is_success()

    def _apply_action(self, action: np.ndarray) -> None:
        super()._apply_action(action)
        self._update_phase()

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        obj_quat = self._get_obj_quat()
        orient_err = self._quat_distance(obj_quat, self._target_quat)
        info.update({
            "phase": self._phase.name,
            "obj_pos": self._get_obj_pos().tolist(),
            "obj_quat": obj_quat.tolist(),
            "target_quat": self._target_quat.tolist(),
            "orient_error_rad": float(orient_err),
            "is_grasped": self._is_grasped(),
            "is_success": self._is_success(),
        })
        return info

    # ====================== 内部辅助 ======================

    def _update_phase(self) -> None:
        tc = self.task_cfg
        ee_pos, _ = self.get_ee_pose()
        obj_pos = self._get_obj_pos()
        obj_quat = self._get_obj_quat()
        orient_err = self._quat_distance(obj_quat, self._target_quat)

        if self._phase == ReorientPhase.REACH:
            above = obj_pos + np.array([0, 0, 0.10])
            if np.linalg.norm(ee_pos - above) < tc.reach_threshold:
                self._phase = ReorientPhase.GRASP

        elif self._phase == ReorientPhase.GRASP:
            if self._is_grasped():
                self._phase = ReorientPhase.REORIENT

        elif self._phase == ReorientPhase.REORIENT:
            if orient_err < tc.orient_threshold and self._is_grasped():
                self._phase = ReorientPhase.PLACE

    def _is_success(self) -> bool:
        tc = self.task_cfg
        obj_pos = self._get_obj_pos()
        obj_quat = self._get_obj_quat()
        orient_err = self._quat_distance(obj_quat, self._target_quat)
        # 物块已落回桌面且姿态达标
        on_table = obj_pos[2] <= tc.place_height_max
        return on_table and orient_err < tc.success_orient_tol

    def _is_grasped(self) -> bool:
        tc = self.task_cfg
        ee_pos, _ = self.get_ee_pose()
        obj_pos = self._get_obj_pos()
        dist = np.linalg.norm(ee_pos - obj_pos)
        if dist > tc.grasp_threshold * 2:
            return False
        tac = self._get_tactile_scalar()
        return bool(dist < tc.grasp_threshold and tac.max() > 0.05)

    def _get_obj_pos(self) -> np.ndarray:
        if self._obj_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._obj_body_id].copy()

    def _get_obj_quat(self) -> np.ndarray:
        if self._obj_body_id < 0:
            self._cache_ids()
        return self.data.xquat[self._obj_body_id].copy()

    def _cache_ids(self) -> None:
        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "obj_reorient"
        )
        jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_reorient_free_joint"
        )
        if jnt_id >= 0:
            self._obj_jnt_qposadr = self.model.jnt_qposadr[jnt_id]

    @staticmethod
    def _quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        """计算两个四元数之间的旋转距离（弧度）."""
        dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
        return float(2.0 * np.arccos(dot))

    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ZYX 欧拉角（弧度）转四元数 (w, x, y, z)."""
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)

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
