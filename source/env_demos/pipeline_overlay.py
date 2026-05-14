"""
Pipeline 模式通用状态叠加显示.

不依赖任何具体任务，只通过 TaskStrategy 基类接口获取信息。
"""

from typing import Optional

import cv2
import mujoco
import numpy as np

from .strategies.base import TaskStrategy


class PipelineStateOverlay:
    """
    通用 Pipeline 状态叠加器.

    支持:
      - MuJoCo viewer 中的阶段指示器（彩色小球）
      - OpenCV 相机画面的状态文字叠加
      - 终端日志（由调用方控制）
    """

    # 阶段颜色循环（当阶段数超过颜色数时循环使用）
    PHASE_COLORS = [
        np.array([0.0, 1.0, 0.0, 0.8]),    # 绿
        np.array([0.0, 0.8, 1.0, 0.8]),    # 青
        np.array([1.0, 0.8, 0.0, 0.8]),    # 橙
        np.array([1.0, 0.4, 0.0, 0.8]),    # 深橙
        np.array([1.0, 0.0, 0.0, 0.8]),    # 红
        np.array([0.8, 0.0, 1.0, 0.8]),    # 紫
        np.array([0.0, 1.0, 0.5, 0.8]),    # 翠绿
        np.array([1.0, 1.0, 0.0, 0.8]),    # 黄
        np.array([0.5, 0.5, 1.0, 0.8]),    # 淡蓝
        np.array([1.0, 0.5, 0.5, 0.8]),    # 粉红
    ]

    def __init__(self, strategy: TaskStrategy):
        self.strategy = strategy
        self._phase_color_map: dict = {}
        self._last_phase_idx: int = -1
        self._phase_enter_step: int = 0
        self._episode_steps: int = 0

    def reset(self) -> None:
        """新回合开始时重置."""
        self._phase_color_map.clear()
        self._last_phase_idx = -1
        self._phase_enter_step = 0
        self._episode_steps = 0

    def update(self, step: int) -> Optional[str]:
        """
        更新状态，检测阶段切换.

        Args:
            step: 当前回合总步数

        Returns:
            阶段切换信息字符串（如果发生切换），否则 None
        """
        self._episode_steps = step
        current_idx = self.strategy.phase_idx

        if current_idx != self._last_phase_idx:
            # 阶段切换
            old_idx = self._last_phase_idx
            old_name = self._get_phase_name(old_idx) if old_idx >= 0 else "start"
            new_name = self._get_phase_name(current_idx)

            duration = step - self._phase_enter_step
            self._phase_enter_step = step
            self._last_phase_idx = current_idx

            # 分配颜色
            if new_name not in self._phase_color_map:
                color_idx = len(self._phase_color_map) % len(self.PHASE_COLORS)
                self._phase_color_map[new_name] = self.PHASE_COLORS[color_idx]

            return (
                f"[{'='*20}]\n"
                f"  → 阶段切换: {old_name} → {new_name}\n"
                f"   上一阶段耗时: {duration} 步\n"
                f"[{'='*20}]"
            )

        return None

    def draw_viewer_indicator(self, viewer, indicator_pos: Optional[np.ndarray] = None) -> None:
        """
        在 MuJoCo viewer 中绘制阶段指示器（彩色小球）.

        Args:
            viewer: MuJoCo viewer 实例
            indicator_pos: 指示器位置，默认场景右上角
        """
        if viewer.user_scn.ngeom >= 950:
            return

        phase_name = self._get_phase_name(self.strategy.phase_idx)
        color = self._phase_color_map.get(
            phase_name,
            np.array([1.0, 1.0, 1.0, 0.5])
        )

        pos = indicator_pos if indicator_pos is not None else np.array([1.8, -1.2, 1.5])

        geom_id = viewer.user_scn.ngeom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=color,
        )
        viewer.user_scn.ngeom += 1

    def draw_camera_overlay(self, camera_bgr: np.ndarray, reward: float = 0.0) -> np.ndarray:
        """
        在相机画面上叠加状态信息.

        Args:
            camera_bgr: BGR 格式相机图像
            reward: 当前步奖励

        Returns:
            叠加后的图像
        """
        phase_name = self._get_phase_name(self.strategy.phase_idx)
        color = self._get_bgr_color(phase_name)

        # 构建状态文本
        lines = [
            f"[PIPELINE]",
            f"EP STEP:{self._episode_steps}",
            f"PHASE: {phase_name.upper()}",
            f"PHASE Step: {self.strategy.phase_step}",
            f"Reward: {reward:+.3f}",
        ]

        # 绘制半透明背景
        overlay = camera_bgr.copy()
        line_h = 22
        total_h = line_h * len(lines) + 10
        cv2.rectangle(overlay, (0, 0), (320, total_h), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.6, camera_bgr, 0.4, 0)

        # 绘制文字
        for i, txt in enumerate(lines):
            y = 20 + i * line_h
            cv2.putText(result, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(result, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1, cv2.LINE_AA)

        return result

    # ---- 私有方法 ----

    def _get_phase_name(self, idx: int) -> str:
        """获取阶段名称."""
        if 0 <= idx < len(self.strategy.phases):
            return self.strategy.phases[idx]
        return "done" if self.strategy.finished else "unknown"

    def _get_bgr_color(self, phase_name: str) -> tuple:
        """获取阶段对应的 BGR 颜色（用于 OpenCV）."""
        color_map = {
            "open_hand": (0, 255, 0),
            "approach": (255, 255, 0),
            "align": (0, 200, 255),
            "descend": (0, 100, 255),
            "grasp": (0, 0, 255),
            "lift": (255, 0, 255),
            "hold": (0, 255, 100),
        }
        # 通用回退：用哈希生成颜色
        if phase_name in color_map:
            return color_map[phase_name]
        h = hash(phase_name) % 256
        return (h, (h * 7) % 256, (h * 13) % 256)