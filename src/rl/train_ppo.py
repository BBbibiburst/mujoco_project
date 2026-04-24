"""
PPO 训练脚本 — 适配扁平化 Dict 观测空间（SB3 MultiInputPolicy）
修复: VecTransposeImage 误判触觉数据为图像空间的问题
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from src.env.pick_place_env import PickPlaceEnv, RobotConfig, PickPlaceConfig


# ====================== 观测包装器：触觉数据维度扩展（避免被识别为图像） ======================

class TactileShapeWrapper(gym.ObservationWrapper):
    """
    将触觉数据从 (5, H, W) 扩展为 (5, H, W, 1)，避免被 SB3 VecTransposeImage 误判为图像空间。

    SB3 的 is_image_space 对 3D Box 且 shape[-1] 较小时会误判为图像。
    扩展为 4D 后 len(shape)==4，不会被识别为图像，从而避免强制转置。
    """

    TACTILE_KEYS = ["tactile_bottom", "tactile_middle", "tactile_top"]

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Dict), (
            f"TactileShapeWrapper 需要 Dict 观测空间，实际: {type(env.observation_space)}"
        )

        # 复制并修改观测空间：触觉键从 (5, H, W) -> (5, H, W, 1)
        new_spaces = dict(env.observation_space.spaces)
        for key in self.TACTILE_KEYS:
            if key in new_spaces:
                old_shape = new_spaces[key].shape  # e.g. (5, 10, 7)
                new_spaces[key] = spaces.Box(
                    low=0, high=255,
                    shape=(*old_shape, 1),  # e.g. (5, 10, 7, 1)
                    dtype=np.uint8
                )

        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs: dict) -> dict:
        new_obs = dict(obs)
        for key in self.TACTILE_KEYS:
            if key in new_obs:
                # (5, H, W) -> (5, H, W, 1)
                new_obs[key] = np.expand_dims(new_obs[key], axis=-1)
        return new_obs


# ====================== CNN 特征提取器（适配 4D 触觉数据） ======================

class MultiModalFeatureExtractor(BaseFeaturesExtractor):
    """
    多模态 CNN 特征提取器：适配扁平化 Dict 观测空间（无嵌套 Dict）。

    处理键：
        - camera_rgb:      (C, H, W)   已被 VecTransposeImage 转置为 channel-first
        - tactile_bottom:  (5, H, W, 1) 4D，避免被识别为图像
        - tactile_middle:  (5, H, W, 1)
        - tactile_top:     (5, H, W, 1)
        - proprioception:  (13,)

    使用方式：
        policy_kwargs = dict(
            features_extractor_class=MultiModalFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
        )
    """

    TACTILE_KEYS = ["tactile_bottom", "tactile_middle", "tactile_top"]

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # ---- 1. 相机图像处理 ----
        cam_space = observation_space["camera_rgb"]
        # 注意：经过 VecTransposeImage 后，shape 已经是 (C, H, W)
        C, H, W = cam_space.shape

        self.camera_cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),   # -> (32, ~58, ~78)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, ~28, ~38)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, ~26, ~36)
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, C, H, W)
            cam_flat_dim = self.camera_cnn(sample).shape[1]

        total_concat_size = cam_flat_dim

        # ---- 2. 触觉图像处理 ----
        self.tactile_cnns = nn.ModuleDict()
        for key in self.TACTILE_KEYS:
            if key not in observation_space.spaces:
                continue

            tac_shape = observation_space[key].shape  # (5, rows, cols, 1)
            n_fingers, rows, cols, _ = tac_shape

            # 输入: (B, 5, rows, cols) —— squeeze 掉最后一维后
            tac_cnn = nn.Sequential(
                nn.Conv2d(n_fingers, 16, kernel_size=3, padding=1),  # -> (16, rows, cols)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),         # -> (32, rows, cols)
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                sample_tac = torch.zeros(1, n_fingers, rows, cols)
                tac_flat_dim = tac_cnn(sample_tac).shape[1]

            self.tactile_cnns[key] = tac_cnn
            total_concat_size += tac_flat_dim

        # ---- 3. 本体感觉处理 ----
        prop_dim = int(np.prod(observation_space["proprioception"].shape))
        self.proprio_mlp = nn.Sequential(
            nn.Linear(prop_dim, 64),
            nn.ReLU(),
        )
        total_concat_size += 64

        # ---- 4. 融合层 ----
        self.fusion = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.ReLU(),
        )

        # 保存维度信息（调试用）
        self._cam_flat_dim = cam_flat_dim
        self._total_concat_size = total_concat_size
        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        encoded = []

        # --- 相机图像 ---
        # VecTransposeImage 已将 (B, H, W, C) -> (B, C, H, W)
        cam = observations["camera_rgb"].float() / 255.0
        cam_features = self.camera_cnn(cam)
        encoded.append(cam_features)

        # --- 触觉图像 ---
        # 输入: (B, 5, rows, cols, 1) -> squeeze -> (B, 5, rows, cols)
        for key in self.TACTILE_KEYS:
            if key in observations:
                tac = observations[key].float() / 255.0   # (B, 5, rows, cols, 1)
                tac = tac.squeeze(-1)                      # (B, 5, rows, cols)
                tac_features = self.tactile_cnns[key](tac)
                encoded.append(tac_features)

        # --- 本体感觉 ---
        prop = observations["proprioception"].float()
        prop = torch.clamp(prop / np.pi, -1.0, 1.0)
        prop_features = self.proprio_mlp(prop)
        encoded.append(prop_features)

        # --- 融合 ---
        combined = torch.cat(encoded, dim=1)
        return self.fusion(combined)


# ====================== 环境工厂 ======================

def make_env(rank: int = 0, seed: int = 0):
    """创建并包装环境（用于多进程向量环境）."""
    def _init():
        robot_cfg = RobotConfig(
            action_mode="osc_pose",
            controller_type="osc",
            max_episode_steps=200,
            action_scale=0.03,
            action_scale_rot=0.06,
            control_freq=20.0,
            sim_freq=1000.0,
            tactile_backend="simple_avg",
            init_arm_qpos=np.array([0.0, 0.5, 0.0, 1.5, 0.0, -1.0, 0.0]),
            init_hand_qpos=np.zeros(6),
        )

        task_cfg = PickPlaceConfig(
            r_step_penalty=-0.005,
            r_place_bonus=100.0,
            r_grasp_bonus=10.0,
            reach_threshold=0.05,
            grasp_threshold=0.04,
        )

        env = PickPlaceEnv(robot_cfg, task_cfg)

        # 包装器顺序（从内到外）:
        # 1. TactileShapeWrapper: 扩展触觉维度，避免 SB3 图像转置误判
        # 2. Monitor: 记录 episode 统计（最外层）
        env = TactileShapeWrapper(env)
        env = Monitor(env)

        return env
    return _init


# ====================== 主训练流程 ======================

def main():
    N_ENVS = 4
    TOTAL_TIMESTEPS = 5_000_000
    SAVE_DIR = PROJECT_ROOT / "rl_models" / "ppo_pickplace"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== PPO 训练启动 ===")
    print(f"  并行环境数: {N_ENVS}")
    print(f"  总训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"  模型保存路径: {SAVE_DIR}")

    if N_ENVS == 1:
        vec_env = DummyVecEnv([make_env(0, seed=42)])
    else:
        vec_env = SubprocVecEnv([make_env(i, seed=42+i) for i in range(N_ENVS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(SAVE_DIR / "checkpoints"),
        name_prefix="ppo_pickplace",
    )

    policy_kwargs = dict(
        features_extractor_class=MultiModalFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    # 使用 MultiInputPolicy 处理 Dict 观测空间
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(SAVE_DIR / "tensorboard"),
        policy_kwargs=policy_kwargs,
        device="auto",
    )

    print("\n=== 开始训练 ===")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_path = SAVE_DIR / "ppo_pickplace_final.zip"
    model.save(final_path)
    print(f"\n✓ 训练完成！模型已保存到: {final_path}")

    vec_env.close()


if __name__ == "__main__":
    main()