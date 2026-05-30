"""
LearnedDiffusionPolicy - 将训练好的扩散策略模型包装为 TaskStrategy 子类.

与 BlockLiftingStrategy 使用相同的 tick() 接口，但动作由神经网络生成。
"""

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from source.train.inference import MultiStageDiffusionPolicyInference
from source.env_demos.strategies.base import TaskStrategy, PhaseResult, PhaseContext, ActionContext


class LearnedDiffusionPolicy(TaskStrategy):
    """
    模仿学习策略：使用训练好的 MultiStageDiffusionPolicy 生成动作。

    与硬编码策略（BlockLiftingStrategy）的区别：
      - 无阶段概念，每步直接输出动作
      - 内部维护观测队列和动作队列（来自 inference.py）
      - 不依赖环境的 phase 推进逻辑
    """

    def __init__(
        self,
        model_path: str,
        stats_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.model_path = model_path
        self.stats_path = stats_path

        # 加载推理器
        self.inference = MultiStageDiffusionPolicyInference(
            model_path=model_path,
            stats_path=stats_path,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # 覆盖基类的阶段列表（learned 策略无阶段，只保留一个占位）
        self._phases = ["learned"]

        # 记录运行状态
        self.total_steps = 0
        self.action_queue_hits = 0  # 动作队列命中次数（统计用）

    @property
    def phases(self) -> list:
        """Learned 策略无真实阶段，返回占位."""
        return self._phases

    def reset(self) -> None:
        """重置策略状态，清空推理器的观测/动作队列."""
        super().reset()
        self.inference.reset()
        self.total_steps = 0
        self.action_queue_hits = 0

    def execute_phase(self, phase_idx: int, ctx: PhaseContext) -> tuple[PhaseResult, ActionContext]:
        """
        执行 learned 策略的单步。

        虽然基类要求实现 execute_phase，但 learned 策略不遵循阶段模型。
        我们直接在这里完成所有逻辑，返回动作上下文。
        """
        env = ctx.env
        obs = ctx.obs
        step = ctx.step

        # ── 构造 step_info（与 inference.py update_obs 期望的格式一致）──
        step_info = self._build_step_info(obs, env)

        # ── 更新观测队列 ──
        self.inference.update_obs(step_info)

        # ── 获取动作 ──
        raw_action = self.inference.get_action()

        if raw_action is None:
            # 观测不足（前 obs_horizon 步），返回零动作
            action_dim = env.action_space.shape[0]
            raw_action = np.zeros(action_dim, dtype=np.float32)
        else:
            self.action_queue_hits += 1

        self.total_steps += 1

        # ── 将原始动作转换为 ActionContext ──
        # 扩散策略直接输出 (arm_qpos[7], hand_qpos[6]) 的绝对目标值
        # 需要根据 action_mode 转换为 delta 或 joint target
        act_ctx = self._action_to_context(raw_action, env)

        # Learned 策略始终返回 CONTINUE，由环境信号判断终止
        return PhaseResult.CONTINUE, act_ctx

    def _build_step_info(self, obs: Dict[str, Any], env: Any) -> Dict[str, Any]:
        """
        将环境 obs 转换为 inference.update_obs 期望的 step_info 格式。

        关键字段:
            - images: {camera_rgb: base64字符串}
            - arm_qpos: [7]
            - hand_qpos: [6]
            - tactile: {finger_0_bottom: {data, shape, dtype}, ...}
        """
        step_info = {}

        # 1. 图像：numpy [H,W,3] RGB → base64 JPEG
        if "camera_rgb" in obs:
            camera_rgb = obs["camera_rgb"]
            if isinstance(camera_rgb, np.ndarray):
                step_info["images"] = {
                    "camera_rgb": self._encode_image_to_base64(camera_rgb)
                }
            else:
                # 已经是 base64 或其他格式
                step_info["images"] = {"camera_rgb": str(camera_rgb)}
        else:
            # 无图像时构造空占位（模型可能依赖视觉，这里需要确保环境提供）
            step_info["images"] = {"camera_rgb": ""}

        # 2. 本体感知
        step_info["arm_qpos"] = obs.get("arm_qpos", env.get_arm_qpos()).astype(np.float32)
        step_info["hand_qpos"] = obs.get("hand_qpos", env.get_hand_qpos()).astype(np.float32)

        # 3. 触觉：从 obs 中提取，或构造空触觉
        tactile_obs = obs.get("tactile", {})
        if isinstance(tactile_obs, dict) and len(tactile_obs) > 0:
            # 已经是 dict 格式（如环境直接提供）
            step_info["tactile"] = tactile_obs
        else:
            # 构造空触觉（所有传感器为零）
            step_info["tactile"] = self._make_empty_tactile()

        return step_info

    def _encode_image_to_base64(self, img_array: np.ndarray) -> str:
        """numpy RGB uint8 [H,W,3] → base64 JPEG data URI."""
        img = Image.fromarray(img_array.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _make_empty_tactile(self) -> Dict[str, Any]:
        """构造全零触觉数据（用于无触觉观测的情况）."""
        from source.train.dataset import SENSOR_ORDER

        def _sensor_data(shape):
            n = shape[0] * shape[1]
            raw = np.zeros(n, dtype=np.uint8).tobytes()
            b64 = "data:application/octet-stream;base64," + base64.b64encode(raw).decode()
            return {"data": b64, "shape": shape, "dtype": "uint8"}

        tactile = {}
        for key in SENSOR_ORDER:
            shape = [10, 7] if "bottom" in key else ([8, 5] if "middle" in key else [6, 5])
            tactile[key] = _sensor_data(shape)
        return tactile

    def _action_to_context(self, raw_action: np.ndarray, env: Any) -> ActionContext:
        cfg = env.cfg
        act = ActionContext()

        arm_target = raw_action[: env.ARM_DOF]
        hand_target = raw_action[env.ARM_DOF : env.ARM_DOF + env.HAND_DOF]

        if cfg.action_mode == "joint":
            act.hand_target = hand_target.astype(np.float64)   # 只存手部 (6,)
            self._arm_target = arm_target.astype(np.float64)   # 手臂单独存

        elif cfg.action_mode == "ee":
            act.ee_delta_pos = np.zeros(3)
            act.ee_delta_rot = np.zeros(3)
            act.hand_target = hand_target.astype(np.float64)
            self._arm_target = arm_target.astype(np.float64)

        return act
    
    def _build_action(self, act_ctx: ActionContext, env) -> np.ndarray:
        cfg = env.cfg

        if cfg.action_mode == "joint":
            current_arm = env.get_arm_qpos()
            current_hand = env.get_hand_qpos()
            current = np.concatenate([current_arm, current_hand])
            target = current.copy()

            arm_target = getattr(self, "_arm_target", None)
            if arm_target is not None:
                target[: env.ARM_DOF] = arm_target
            if act_ctx.hand_target is not None:
                target[env.ARM_DOF :] = act_ctx.hand_target

            delta = target - current
            scale_hand = cfg.action_scale_hand or cfg.action_scale
            delta[: env.ARM_DOF] /= cfg.action_scale
            delta[env.ARM_DOF :] /= scale_hand

            return np.clip(delta.astype(np.float32), -1.0, 1.0)

        # ee 模式沿用父类逻辑
        return super()._build_action(act_ctx, env)

    def get_status_dict(self) -> dict:
        """返回策略状态信息."""
        base = super().get_status_dict()
        base.update({
            "model_path": self.model_path,
            "stats_path": self.stats_path,
            "device": str(self.inference.device),
            "obs_queue": len(self.inference.visual_deque),
            "action_queue": len(self.inference.action_queue),
            "total_steps": self.total_steps,
            "action_queue_hits": self.action_queue_hits,
            "strategy": "learned_diffusion",
        })
        return base