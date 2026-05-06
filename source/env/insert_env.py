"""
插孔任务环境

任务描述：
    机械臂+灵巧手抓取细长圆柱体（peg），将其垂直插入桌面上的圆形孔洞（hole）中。
    这是精密装配任务，对触觉反馈和末端位姿控制精度要求最高。

难度设计：
    - 插孔误差容限极小（<= insert_tolerance，约 5mm 级）
    - 需要先水平对准，再垂直下压
    - 触觉传感器指示接触状态，有助于感知对准

阶段：
    REACH → GRASP → ALIGN → INSERT

奖励结构：
    REACH：接近奖励
    GRASP：抓取奖励（触觉接触）
    ALIGN：XY 对准奖励（越近越高）
    INSERT：插入成功奖励（最高）

观测/动作空间与 pick_place_env 完全一致。
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional
import mujoco
import numpy as np
from .base_env import RobotArmEnvBase, RobotConfig
from gymnasium import spaces
from source.sensors.tactile_sensor import TactileReader, FINGER_PHALANX_ORDER


class InsertPhase(IntEnum):
    REACH  = 0
    GRASP  = 1
    ALIGN  = 2
    INSERT = 3


@dataclass
class InsertConfig:
    """插孔任务配置."""

    # 圆柱 peg 尺寸
    peg_radius:  float = 0.012
    peg_length:  float = 0.08
    peg_mass:    float = 0.05

    # 孔半径（略大于 peg，留有公差）
    hole_radius: float = 0.015

    # peg 初始随机范围
    peg_pos_range: np.ndarray = field(
        default_factory=lambda: np.array([[0.30, -0.12, 0.04],
                                          [0.42,  0.12, 0.04]])
    )

    # hole 固定位置随机范围
    hole_pos_range: np.ndarray = field(
        default_factory=lambda: np.array([[0.46, -0.12, 0.001],
                                          [0.58,  0.12, 0.001]])
    )

    min_ph_dist: float = 0.15   # peg 和 hole 最小初始距离

    # 手部
    hand_open:  np.ndarray = field(default_factory=lambda: np.zeros(6))
    hand_close: np.ndarray = field(default_factory=lambda: np.full(6, 0.006))

    # 阈值
    reach_threshold:  float = 0.05
    grasp_threshold:  float = 0.04
    align_threshold:  float = 0.015   # 水平对准阈值（m）
    insert_depth:     float = 0.04    # 认为成功插入的最小深度（peg 进入 hole 的深度）
    insert_tolerance: float = 0.012   # 插入时允许的 XY 偏差

    # 奖励
    r_reach_scale:    float = 1.5
    r_grasp_bonus:    float = 8.0
    r_align_scale:    float = 5.0
    r_insert_bonus:   float = 100.0
    r_drop_penalty:   float = -3.0
    r_step_penalty:   float = -0.01


_FINGER_NAMES = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
_TACTILE_LEVELS = {"bottom": (10, 7), "middle": (8, 5), "top": (6, 5)}


class InsertEnv(RobotArmEnvBase):
    """将圆柱 peg 插入圆孔的精密装配任务环境."""

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[InsertConfig] = None,
    ):
        self.task_cfg = task_config or InsertConfig()
        super().__init__(robot_config)

        self._phase = InsertPhase.REACH
        self._peg_pos_init = np.zeros(3)
        self._hole_pos = np.zeros(3)

        self._peg_body_id: int = -1
        self._hole_body_id: int = -1
        self._peg_jnt_qposadr: int = -1

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
            diffuse=[1.0, 1.0, 1.0], ambient=[0.4, 0.4, 0.4],
        )

        # Peg：竖立圆柱，橙色，带自由关节
        peg = wb.add_body(
            name="peg",
            pos=[0.35, 0.0, tc.peg_length / 2],
        )
        peg.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[tc.peg_radius, tc.peg_length / 2],
            rgba=[0.95, 0.5, 0.1, 1.0],
            mass=tc.peg_mass,
            friction=[1.2, 0.005, 0.0001],
            condim=4,
            name="peg_geom",
        )
        peg.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="peg_free_joint")

        # 孔板：固定在桌面，深色平板带圆孔（通过 condim/contype 模拟孔的碰撞）
        # 简化：用一个大平面体 + 孔位置用 site 标记
        hole_plate = wb.add_body(name="hole_plate", pos=[0.52, 0.0, 0.005])
        hole_plate.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.08, 0.08, 0.005],
            rgba=[0.3, 0.3, 0.8, 1.0],
            mass=5.0,
            name="hole_plate_geom",
        )
        # 孔中心 site（用于距离计算）
        hole_plate.add_site(
            name="hole_center",
            pos=[0.0, 0.0, 0.006],   # 板面正上方
            size=[tc.hole_radius],
            rgba=[0.9, 0.9, 0.1, 0.8],
        )

        # 俯视相机
        wb.add_camera(
            name="top_cam",
            pos=[0.45, 0.0, 1.2],
            quat=[1, 0, 0, 0],
            fovy=45,
        )

    def _reset_scene(self) -> None:
        tc = self.task_cfg

        if self._peg_body_id == -1:
            self._cache_ids()

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)

        # 随机化 hole 位置
        h_lo, h_hi = tc.hole_pos_range[0], tc.hole_pos_range[1]
        self._hole_pos = h_lo + np.random.random(3) * (h_hi - h_lo)
        self._hole_pos[2] = 0.001
        # 移动 hole_plate body
        self.model.body_pos[self._hole_body_id] = [
            self._hole_pos[0], self._hole_pos[1], 0.005
        ]

        # 随机化 peg 位置，保证与 hole 有距离
        p_lo, p_hi = tc.peg_pos_range[0], tc.peg_pos_range[1]
        for _ in range(100):
            p_xy = p_lo[:2] + np.random.random(2) * (p_hi[:2] - p_lo[:2])
            if np.linalg.norm(p_xy - self._hole_pos[:2]) >= tc.min_ph_dist:
                break
        p_pos = np.array([p_xy[0], p_xy[1], tc.peg_length / 2])
        self._peg_pos_init = p_pos.copy()

        if self._peg_jnt_qposadr >= 0:
            adr = self._peg_jnt_qposadr
            self.data.qpos[adr:adr+3] = p_pos
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "peg_free_joint")
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr:dof_adr+6] = 0.0

        self._phase = InsertPhase.REACH
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
        peg_pos = self._get_peg_pos()

        if self._phase == InsertPhase.REACH:
            above_peg = peg_pos + np.array([0, 0, 0.12])
            dist = np.linalg.norm(ee_pos - above_peg)
            reward += tc.r_reach_scale * (1.0 - np.tanh(5 * dist))

        elif self._phase == InsertPhase.GRASP:
            if self._is_peg_grasped():
                reward += tc.r_grasp_bonus
            if not self._is_peg_grasped():
                reward += tc.r_drop_penalty * 0.3

        elif self._phase == InsertPhase.ALIGN:
            xy_dist = np.linalg.norm(peg_pos[:2] - self._hole_pos[:2])
            reward += tc.r_align_scale * (1.0 - np.tanh(10 * xy_dist))
            if not self._is_peg_grasped():
                reward += tc.r_drop_penalty

        elif self._phase == InsertPhase.INSERT:
            if self._is_inserted():
                reward += tc.r_insert_bonus
            else:
                xy_dist = np.linalg.norm(peg_pos[:2] - self._hole_pos[:2])
                reward += tc.r_align_scale * 2 * (1.0 - np.tanh(10 * xy_dist))

        return float(reward)

    def _is_terminated(self) -> bool:
        return self._is_inserted()

    def _apply_action(self, action: np.ndarray) -> None:
        super()._apply_action(action)
        self._update_phase()

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        peg_pos = self._get_peg_pos()
        info.update({
            "phase": self._phase.name,
            "peg_pos": peg_pos.tolist(),
            "hole_pos": self._hole_pos.tolist(),
            "xy_dist_to_hole": float(np.linalg.norm(peg_pos[:2] - self._hole_pos[:2])),
            "is_grasped": self._is_peg_grasped(),
            "is_inserted": self._is_inserted(),
        })
        return info

    # ====================== 内部辅助 ======================

    def _update_phase(self) -> None:
        tc = self.task_cfg
        ee_pos, _ = self.get_ee_pose()
        peg_pos = self._get_peg_pos()

        if self._phase == InsertPhase.REACH:
            above_peg = peg_pos + np.array([0, 0, 0.10])
            if np.linalg.norm(ee_pos - above_peg) < tc.reach_threshold:
                self._phase = InsertPhase.GRASP

        elif self._phase == InsertPhase.GRASP:
            if self._is_peg_grasped():
                self._phase = InsertPhase.ALIGN

        elif self._phase == InsertPhase.ALIGN:
            xy_dist = np.linalg.norm(peg_pos[:2] - self._hole_pos[:2])
            if xy_dist < tc.align_threshold:
                self._phase = InsertPhase.INSERT

    def _is_inserted(self) -> bool:
        tc = self.task_cfg
        peg_pos = self._get_peg_pos()
        xy_dist = np.linalg.norm(peg_pos[:2] - self._hole_pos[:2])
        # 当 peg 底端深入 hole 板面以下，且水平对齐
        peg_bottom_z = peg_pos[2] - tc.peg_length / 2
        return xy_dist < tc.insert_tolerance and peg_bottom_z < 0.005

    def _is_peg_grasped(self) -> bool:
        tc = self.task_cfg
        ee_pos, _ = self.get_ee_pose()
        peg_pos = self._get_peg_pos()
        dist = np.linalg.norm(ee_pos - peg_pos)
        if dist > tc.grasp_threshold * 2:
            return False
        tac = self._get_tactile_scalar()
        return bool(dist < tc.grasp_threshold and tac.max() > 0.05)

    def _get_peg_pos(self) -> np.ndarray:
        if self._peg_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._peg_body_id].copy()

    def _cache_ids(self) -> None:
        self._peg_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "peg"
        )
        self._hole_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hole_plate"
        )
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "peg_free_joint")
        if jnt_id >= 0:
            self._peg_jnt_qposadr = self.model.jnt_qposadr[jnt_id]

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
                    img = imgs.get(phalanx, None)
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
