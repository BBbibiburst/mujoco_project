"""
仿真可视化工具.

EETrajectoryVisualizer  : 末端执行器轨迹 + 目标位姿可视化
FingertipMidpointVisualizer : thumb 与 finger_3 指尖连线中点可视化
"""

from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from source.env.base_env import RobotArmEnvBase


# ====================== 末端轨迹可视化 ======================

@dataclass
class TrajectoryVisualStyle:
    actual_rgba: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 1.0, 0.8]))
    target_rgba: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.4]))
    actual_size: float = 0.005
    target_size: float = 0.015


class EETrajectoryVisualizer:
    """在 MuJoCo viewer 中绘制末端执行器轨迹（实际位置 + 目标位置 + 历史轨迹）."""

    _MAX_GEOMS   = 1000
    _SAFE_MARGIN = 50

    def __init__(self, style: Optional[TrajectoryVisualStyle] = None, max_history: int = 2000):
        self.style       = style or TrajectoryVisualStyle()
        self.max_history = max_history
        self.actual_pos: Optional[np.ndarray] = None
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.history: list = []

    def update(
        self,
        actual_pos: np.ndarray,
        target_pos: Optional[np.ndarray] = None,
        target_quat: Optional[np.ndarray] = None,
    ) -> None:
        self.actual_pos = actual_pos.copy()
        if target_pos  is not None: self.target_pos  = target_pos.copy()
        if target_quat is not None: self.target_quat = target_quat.copy()
        if self.max_history > 0:
            self.history.append(actual_pos.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def draw(self, viewer) -> None:
        if self.actual_pos is None:
            return
        budget = self._MAX_GEOMS - self._SAFE_MARGIN

        self._draw_history(viewer, budget)
        self._draw_sphere(viewer, self.actual_pos, self.style.actual_size, self.style.actual_rgba, budget)
        if self.target_pos is not None:
            self._draw_sphere(viewer, self.target_pos, self.style.target_size, self.style.target_rgba, budget)
        if self.target_quat is not None and self.target_pos is not None:
            self._draw_axes(viewer, self.target_pos, self.target_quat, budget)

    def reset(self) -> None:
        self.history.clear()
        self.actual_pos  = None
        self.target_pos  = None
        self.target_quat = None

    # ---- 私有绘制方法 ----

    def _draw_history(self, viewer, budget: int) -> None:
        n = len(self.history)
        if n < 2:
            return
        for i, pos in enumerate(self.history[:-1]):
            if viewer.user_scn.ngeom >= budget:
                break
            alpha = 0.1 + 0.3 * (i / n)
            size  = self.style.actual_size * (0.5 + 0.5 * (i / n))
            rgba  = self.style.actual_rgba.copy()
            rgba[3] = alpha
            self._add_sphere(viewer, pos, size, rgba)

    def _draw_sphere(self, viewer, pos, size, rgba, budget) -> None:
        if viewer.user_scn.ngeom < budget:
            self._add_sphere(viewer, pos, size, rgba)

    def _draw_axes(self, viewer, pos, quat, budget) -> None:
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        rot = mat.reshape(3, 3)
        colors = [
            np.array([1, 0, 0, 1], dtype=np.float32),
            np.array([0, 1, 0, 1], dtype=np.float32),
            np.array([0, 0, 1, 1], dtype=np.float32),
        ]
        axis_len = 0.1
        for i in range(3):
            if viewer.user_scn.ngeom >= budget:
                break
            from_pos = pos.astype(np.float64)
            to_pos   = (pos + rot[:, i] * axis_len).astype(np.float64)
            geom_id  = viewer.user_scn.ngeom
            geom     = viewer.user_scn.geoms[geom_id]
            mujoco.mjv_initGeom(
                geom, type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                size=np.array([0.002, 0.002, axis_len], dtype=np.float64),
                pos=from_pos, mat=np.eye(3).flatten().astype(np.float64),
                rgba=colors[i],
            )
            mujoco.mjv_connector(geom, int(mujoco.mjtGeom.mjGEOM_CYLINDER), 0.002, from_pos, to_pos)
            viewer.user_scn.ngeom += 1

    @staticmethod
    def _add_sphere(viewer, pos, size, rgba) -> None:
        geom_id = viewer.user_scn.ngeom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0, 0], pos=pos,
            mat=np.eye(3).flatten(), rgba=rgba,
        )
        viewer.user_scn.ngeom += 1


# ====================== 指尖中点可视化 ======================

@dataclass
class FingertipMidpointStyle:
    midpoint_rgba: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.2, 0.8, 1.0]))
    midpoint_size: float = 0.012


class FingertipMidpointVisualizer:
    """绘制 thumb 与 finger_3 指尖连线中点."""

    def __init__(self, style: Optional[FingertipMidpointStyle] = None):
        self.style    = style or FingertipMidpointStyle()
        self.midpoint: Optional[np.ndarray] = None

    def update(self, env: RobotArmEnvBase) -> None:
        try:
            thumb   = env.get_site_pos("inspirehand_fingertip_thumb")
            finger3 = env.get_site_pos("inspirehand_fingertip_3")
            self.midpoint = (thumb + finger3) / 2.0
        except ValueError:
            self.midpoint = None

    def draw(self, viewer) -> None:
        if self.midpoint is None or viewer.user_scn.ngeom >= 950:
            return
        geom_id = viewer.user_scn.ngeom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.style.midpoint_size, 0, 0],
            pos=self.midpoint,
            mat=np.eye(3).flatten(),
            rgba=self.style.midpoint_rgba,
        )
        viewer.user_scn.ngeom += 1

    def reset(self) -> None:
        self.midpoint = None