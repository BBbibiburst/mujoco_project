"""
堆叠任务环境

任务描述：
    机械臂+灵巧手抓取红色方块（block_A），将其堆叠放置到绿色方块（block_B）正上方。
    block_B 固定在桌面不动，block_A 初始随机摆放在桌面区域。

难度设计：
    - 需要精确的高度控制（放置高度 = block_B.z + 2×obj_size）
    - 需要水平对齐（XY 误差 < place_threshold）
    - 触觉反馈对防止掉落至关重要

奖励结构：
    REACH 阶段：-dist(ee, above_A) * scale
    GRASP 阶段：一次性抓取奖励 + 触觉接触奖励
    TRANSPORT：-dist(A, above_B) * scale
    PLACE：一次性堆叠成功奖励

观测空间（与 pick_place_env 完全一致）：
    - camera_rgb:      (240, 320, 3)
    - tactile_bottom:  (5, 10, 7)
    - tactile_middle:  (5, 8, 5)
    - tactile_top:     (5, 6, 5)
    - proprioception:  (13,)

"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from .base_env import RobotArmEnvBase, RobotConfig
from gymnasium import spaces
from src.sensors.tactile_sensor import TactileReader, FINGER_PHALANX_ORDER


# ====================== 任务阶段 ======================

class StackPhase(IntEnum):
    REACH     = 0
    GRASP     = 1
    TRANSPORT = 2
    PLACE     = 3


# ====================== 配置 ======================

@dataclass
class StackConfig:
    """堆叠任务配置."""

    # 方块尺寸
    obj_size: float = 0.025
    obj_mass: float = 0.1

    # block_A 初始位置随机范围（被抓方块）
    block_a_pos_range: np.ndarray = field(
        default_factory=lambda: np.array([[0.30, -0.15, 0.025],
                                          [0.45,  0.15, 0.025]])
    )
    # block_B 固定位置随机范围（底座方块）
    block_b_pos_range: np.ndarray = field(
        default_factory=lambda: np.array([[0.45, -0.15, 0.025],
                                          [0.60,  0.15, 0.025]])
    )
    # A、B 最小水平距离（防止初始重叠）
    min_ab_dist: float = 0.12

    # 手部开合
    hand_open:  np.ndarray = field(default_factory=lambda: np.zeros(6))
    hand_close: np.ndarray = field(default_factory=lambda: np.full(6, 0.008))

    # 阶段切换阈值
    reach_threshold:     float = 0.04
    grasp_threshold:     float = 0.03
    transport_threshold: float = 0.05

    # 奖励
    r_reach_scale:     float = 2.0
    r_grasp_bonus:     float = 10.0
    r_transport_scale: float = 3.0
    r_place_bonus:     float = 80.0
    r_drop_penalty:    float = -5.0
    r_step_penalty:    float = -0.01

    # 成功判定
    success_xy_dist:   float = 0.03   # 水平对齐误差
    success_height_tol: float = 0.015  # 高度误差容限


# ====================== 触觉辅助常量（与 pick_place_env 一致） ======================

_FINGER_NAMES = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
_TACTILE_LEVELS = {
    "bottom": (10, 7),
    "middle": (8, 5),
    "top":    (6, 5),
}


# ====================== 堆叠环境 ======================

class StackEnv(RobotArmEnvBase):
    """将 block_A 堆叠到 block_B 上的任务环境."""

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        task_config: Optional[StackConfig] = None,
    ):
        self.task_cfg = task_config or StackConfig()
        super().__init__(robot_config)

        self._phase = StackPhase.REACH
        self._block_a_pos = np.zeros(3)
        self._block_b_pos = np.zeros(3)

        self._block_a_body_id: int = -1
        self._block_b_body_id: int = -1
        self._block_a_jnt_qposadr: int = -1

        self._renderer: Optional[mujoco.Renderer] = None
        self._cam_id: int = -1

    # ====================== 观测/动作空间 ======================

    @property
    def observation_space(self):
        return spaces.Dict({
            "camera_rgb": spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8),
            "tactile_bottom": spaces.Box(low=0, high=255, shape=(5, 10, 7), dtype=np.uint8),
            "tactile_middle": spaces.Box(low=0, high=255, shape=(5, 8, 5),  dtype=np.uint8),
            "tactile_top":    spaces.Box(low=0, high=255, shape=(5, 6, 5),  dtype=np.uint8),
            "proprioception": spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
        })

    # ====================== 抽象方法实现 ======================

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        wb = spec.worldbody
        tc = self.task_cfg

        wb.add_light(
            name="top_light", pos=[0.3, 0.0, 1.8],
            dir=[0.0, 0.0, -1.0],
            diffuse=[0.9, 0.9, 0.9], ambient=[0.3, 0.3, 0.3],
        )

        # block_A：可移动（自由关节），红色
        body_a = wb.add_body(name="block_a", pos=[0.35, 0.0, tc.obj_size])
        body_a.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.obj_size] * 3,
            rgba=[0.9, 0.2, 0.1, 1.0],
            mass=tc.obj_mass,
            friction=[1.0, 0.005, 0.0001],
            condim=4,
            name="block_a_geom",
        )
        body_a.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="block_a_free_joint")

        # block_B：固定底座，绿色（无关节，焊死）
        body_b = wb.add_body(name="block_b", pos=[0.5, 0.0, tc.obj_size])
        body_b.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.obj_size * 1.1, tc.obj_size * 1.1, tc.obj_size],
            rgba=[0.1, 0.8, 0.2, 1.0],
            mass=1.0,   # 质量大，模拟固定
            friction=[1.0, 0.005, 0.0001],
            name="block_b_geom",
        )
        # block_B 通过 weld 固定——此处以超大质量近似固定

        # 目标指示圈（半透明轮廓，显示 block_B 顶部）
        target_vis = wb.add_body(name="stack_target_vis", pos=[0.5, 0.0, tc.obj_size * 2 + 0.001])
        target_vis.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tc.obj_size * 1.2, tc.obj_size * 1.2, 0.001],
            rgba=[0.2, 0.9, 0.9, 0.3],
            contype=0, conaffinity=0,
        )

        # 俯视相机
        wb.add_camera(
            name="top_cam",
            pos=[0.45, 0.0, 1.2],
            quat=[1, 0, 0, 0],
            fovy=50,
        )

    def _reset_scene(self) -> None:
        tc = self.task_cfg

        if self._block_a_body_id == -1:
            self._cache_ids()

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)

        # 随机化 block_B 位置
        b_lo, b_hi = tc.block_b_pos_range[0], tc.block_b_pos_range[1]
        self._block_b_pos = b_lo + np.random.random(3) * (b_hi - b_lo)
        self._block_b_pos[2] = tc.obj_size  # 固定高度

        b_id = self._block_b_body_id
        self.model.body_pos[b_id] = self._block_b_pos

        # 同步 target_vis
        vis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "stack_target_vis")
        if vis_id >= 0:
            vis_pos = self._block_b_pos.copy()
            vis_pos[2] = tc.obj_size * 2 + 0.001
            self.model.body_pos[vis_id] = vis_pos

        # 随机化 block_A，保证与 block_B 有足够距离
        a_lo, a_hi = tc.block_a_pos_range[0], tc.block_a_pos_range[1]
        for _ in range(100):
            a_pos = a_lo + np.random.random(3) * (a_hi - a_lo)
            a_pos[2] = tc.obj_size
            if np.linalg.norm(a_pos[:2] - self._block_b_pos[:2]) >= tc.min_ab_dist:
                break
        self._block_a_pos = a_pos.copy()

        if self._block_a_jnt_qposadr >= 0:
            adr = self._block_a_jnt_qposadr
            self.data.qpos[adr:adr+3] = a_pos
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_a_free_joint")
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr:dof_adr+6] = 0.0

        self._phase = StackPhase.REACH
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> Dict[str, Any]:
        camera_rgb = self._render_camera()
        tactile_grouped = self._get_tactile_grouped()
        arm_q = self.get_arm_qpos()
        hand_q = self.get_hand_qpos()
        proprioception = np.concatenate([arm_q, hand_q]).astype(np.float32)

        return {
            "camera_rgb": camera_rgb,
            "tactile_bottom": tactile_grouped["bottom"],
            "tactile_middle": tactile_grouped["middle"],
            "tactile_top":    tactile_grouped["top"],
            "proprioception": proprioception,
        }

    def _compute_reward(self) -> float:
        tc = self.task_cfg
        reward = tc.r_step_penalty

        ee_pos, _ = self.get_ee_pose()
        a_pos = self._get_block_a_pos()
        # 目标放置位置：block_B 正上方，高度为两倍方块
        stack_target = self._block_b_pos.copy()
        stack_target[2] = tc.obj_size * 2 + tc.obj_size  # block_B 顶面 + half of A

        if self._phase == StackPhase.REACH:
            above_a = a_pos + np.array([0, 0, 0.12])
            dist = np.linalg.norm(ee_pos - above_a)
            reward += tc.r_reach_scale * (1.0 - np.tanh(5 * dist))

        elif self._phase == StackPhase.GRASP:
            if self._is_block_a_grasped():
                reward += tc.r_grasp_bonus
            else:
                reward += tc.r_reach_scale * 0.5

        elif self._phase == StackPhase.TRANSPORT:
            dist = np.linalg.norm(a_pos - stack_target)
            reward += tc.r_transport_scale * (1.0 - np.tanh(5 * dist))
            if not self._is_block_a_grasped():
                reward += tc.r_drop_penalty

        elif self._phase == StackPhase.PLACE:
            if self._is_stacked_success():
                reward += tc.r_place_bonus

        return float(reward)

    def _is_terminated(self) -> bool:
        return self._is_stacked_success()

    def _apply_action(self, action: np.ndarray) -> None:
        super()._apply_action(action)
        self._update_phase()

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        a_pos = self._get_block_a_pos()
        stack_target = self._block_b_pos.copy()
        stack_target[2] = self.task_cfg.obj_size * 3
        info.update({
            "phase": self._phase.name,
            "block_a_pos": a_pos.tolist(),
            "block_b_pos": self._block_b_pos.tolist(),
            "stack_target": stack_target.tolist(),
            "is_grasped": self._is_block_a_grasped(),
            "is_stacked": self._is_stacked_success(),
        })
        return info

    # ====================== 内部辅助 ======================

    def _update_phase(self) -> None:
        tc = self.task_cfg
        ee_pos, _ = self.get_ee_pose()
        a_pos = self._get_block_a_pos()
        stack_target = self._block_b_pos.copy()
        stack_target[2] = tc.obj_size * 3

        if self._phase == StackPhase.REACH:
            above_a = a_pos + np.array([0, 0, 0.10])
            if np.linalg.norm(ee_pos - above_a) < tc.reach_threshold:
                self._phase = StackPhase.GRASP

        elif self._phase == StackPhase.GRASP:
            if self._is_block_a_grasped():
                self._phase = StackPhase.TRANSPORT

        elif self._phase == StackPhase.TRANSPORT:
            horiz_dist = np.linalg.norm(a_pos[:2] - self._block_b_pos[:2])
            if horiz_dist < tc.transport_threshold and a_pos[2] > tc.obj_size * 2:
                self._phase = StackPhase.PLACE

    def _is_stacked_success(self) -> bool:
        tc = self.task_cfg
        a_pos = self._get_block_a_pos()
        xy_dist = np.linalg.norm(a_pos[:2] - self._block_b_pos[:2])
        target_z = tc.obj_size * 3
        z_err = abs(a_pos[2] - target_z)
        return xy_dist < tc.success_xy_dist and z_err < tc.success_height_tol

    def _is_block_a_grasped(self) -> bool:
        tc = self.task_cfg
        ee_pos, _ = self.get_ee_pose()
        a_pos = self._get_block_a_pos()
        dist = np.linalg.norm(ee_pos - a_pos)
        if dist > tc.grasp_threshold * 2:
            return False
        tac = self._get_tactile_scalar()
        return bool(dist < tc.grasp_threshold and tac.max() > 0.05)

    def _get_block_a_pos(self) -> np.ndarray:
        if self._block_a_body_id < 0:
            self._cache_ids()
        return self.data.xpos[self._block_a_body_id].copy()

    def _cache_ids(self) -> None:
        self._block_a_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block_a"
        )
        self._block_b_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block_b"
        )
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_a_free_joint")
        if jnt_id >= 0:
            self._block_a_jnt_qposadr = self.model.jnt_qposadr[jnt_id]

    def _get_tactile_scalar(self) -> np.ndarray:
        if self.reader is None:
            return np.zeros(6)
        try:
            raw = self.reader.read_raw(self.data)
            if not raw:
                return np.zeros(6)
            feats = []
            for finger in _FINGER_NAMES:
                phalanges = FINGER_PHALANX_ORDER[finger]
                vals = [float(raw[n].max()) for n in phalanges if n in raw]
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
                level_imgs = []
                for finger in _FINGER_NAMES:
                    phalanx = FINGER_PHALANX_ORDER[finger][level_idx]
                    exp_h, exp_w = _TACTILE_LEVELS[level]
                    if phalanx in imgs:
                        img = imgs[phalanx]
                        if img.shape == (exp_w, exp_h):
                            img = img.T
                        level_imgs.append(img.astype(np.uint8))
                    else:
                        level_imgs.append(np.zeros((exp_h, exp_w), dtype=np.uint8))
                result[level] = np.stack(level_imgs, axis=0)
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
