"""
PPO 训练脚本 — 扁平化观测版本（解耦观测空间 shape）
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


# ====================== 环境包装器：扁平化观测（动态读取观测空间） ======================

class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    将 Dict 观测空间扁平化为 Box，适配 SB3 标准策略。
    
    从环境 observation_space 动态读取 shape，不硬编码任何维度。
    支持的结构：
        {
            "camera_rgb": (H, W, C) uint8,
            "tactile": Dict { level: (N_fingers, rows, cols) uint8, ... },
            "proprioception": (D,) float32,
        }
    
    输出: (flattened_dim,) float32 — 所有数据归一化后拼接
    """
    
    # 观测 key 的约定（换任务时保持 key 名一致即可）
    CAM_KEY = "camera_rgb"
    TACTILE_KEY = "tactile"
    PROP_KEY = "proprioception"
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Dict), (
            f"FlattenObservationWrapper 需要 Dict 观测空间，"
            f"实际类型: {type(obs_space)}"
        )
        
        # 动态读取各组件维度
        self._cam_shape = obs_space[self.CAM_KEY].shape  # (H, W, C)
        self._cam_dim = int(np.prod(self._cam_shape))
        
        self._tac_shapes = {}  # {level: shape}
        self._tac_dim = 0
        if self.TACTILE_KEY in obs_space.spaces:
            tac_space = obs_space[self.TACTILE_KEY]
            assert isinstance(tac_space, spaces.Dict), (
                f"tactile 必须是 Dict，实际: {type(tac_space)}"
            )
            for level, space in tac_space.spaces.items():
                self._tac_shapes[level] = space.shape
                self._tac_dim += int(np.prod(space.shape))
        
        self._prop_shape = obs_space[self.PROP_KEY].shape
        self._prop_dim = int(np.prod(self._prop_shape))
        
        self.flattened_dim = self._cam_dim + self._tac_dim + self._prop_dim
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.flattened_dim,), 
            dtype=np.float32
        )
        
        self.last_raw_obs = None
        
        # 打印维度分解（方便调试）
        print(f"[FlattenWrapper] 观测维度: {self.flattened_dim:,}")
        print(f"  camera ({self.CAM_KEY}): {self._cam_dim:,} "
              f"({self._cam_dim/self.flattened_dim*100:.1f}%) "
              f"shape={self._cam_shape}")
        if self._tac_dim > 0:
            tac_detail = ", ".join(f"{k}={v}" for k, v in self._tac_shapes.items())
            print(f"  tactile ({self.TACTILE_KEY}): {self._tac_dim:,} "
                  f"({self._tac_dim/self.flattened_dim*100:.1f}%) "
                  f"[{tac_detail}]")
        else:
            print(f"  tactile: 0 (禁用)")
        print(f"  proprioception ({self.PROP_KEY}): {self._prop_dim} "
              f"({self._prop_dim/self.flattened_dim*100:.1f}%) "
              f"shape={self._prop_shape}")

    def observation(self, obs: dict) -> np.ndarray:
        self.last_raw_obs = obs
        
        parts = []
        
        # 1. 相机图像：uint8 [0,255] → float32 [0,1]
        cam = obs[self.CAM_KEY].astype(np.float32).flatten() / 255.0
        parts.append(cam)
        
        # 2. 触觉：uint8 [0,255] → float32 [0,1]
        if self.TACTILE_KEY in obs and self._tac_dim > 0:
            tac_flat = []
            for level in self._tac_shapes.keys():
                tac_flat.append(
                    obs[self.TACTILE_KEY][level].astype(np.float32).flatten() / 255.0
                )
            parts.append(np.concatenate(tac_flat))
        
        # 3. 本体感觉：已经是 float32，做简单归一化
        prop = obs[self.PROP_KEY].astype(np.float32)
        prop = np.clip(prop / np.pi, -1.0, 1.0)
        parts.append(prop)
        
        return np.concatenate(parts)


# ====================== 可选：CNN 特征提取器（动态 reshape） ======================

class CustomCNN(BaseFeaturesExtractor):
    """
    当观测包含图像时，用 CNN 提取特征。
    从 FlattenObservationWrapper 动态读取 shape，不硬编码分辨率。
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 注意：CustomCNN 必须与 FlattenObservationWrapper 配合使用
        # 这里假设 wrapper 已经初始化，且 observation_space 是扁平化后的 Box
        
        # 需要从环境获取原始 shape 信息，这里通过全局约定或额外参数传递
        # 实际使用时，推荐在 make_env 中把 shape 信息存到 env.metadata 或 wrapper 属性
        # 为简化，这里保留从环境创建时的显式参数方式
        
        raise NotImplementedError(
            "CustomCNN 需要配合 FlattenObservationWrapper 使用，"
            "请通过 policy_kwargs 传入 cam_shape 和 tac_shapes，"
            "或改用下面的 DynamicCNNFeatureExtractor。"
        )


class DynamicCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    动态 CNN 特征提取器：接收原始 Dict 观测，内部处理各模态。
    
    这个提取器直接包装在环境之前（不经过 FlattenObservationWrapper），
    因此可以直接访问原始 Dict 观测空间的 shape 信息。
    
    使用方式：
        policy_kwargs = dict(
            features_extractor_class=DynamicCNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 从 Dict 空间动态读取
        cam_space = observation_space["camera_rgb"]
        self.cam_shape = cam_space.shape  # (H, W, C)
        H, W, C = self.cam_shape
        
        # CNN 处理相机图像: (C, H, W)
        self.camera_cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算 CNN 输出维度
        with torch.no_grad():
            sample = torch.zeros(1, C, H, W)
            cam_features_dim = self.camera_cnn(sample).shape[1]
        
        # 触觉 MLP
        tac_dim = 0
        if "tactile" in observation_space.spaces:
            tac_space = observation_space["tactile"]
            for level, space in tac_space.spaces.items():
                tac_dim += int(np.prod(space.shape))
        
        prop_dim = int(np.prod(observation_space["proprioception"].shape))
        
        self.tac_prop_mlp = nn.Sequential(
            nn.Linear(tac_dim + prop_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(cam_features_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # 处理相机: (B, H, W, C) -> (B, C, H, W)
        cam = observations["camera_rgb"].float() / 255.0
        cam = cam.permute(0, 3, 1, 2)  # NHWC -> NCHW
        cam_features = self.camera_cnn(cam)
        
        # 处理触觉
        tac_parts = []
        if "tactile" in observations:
            for level in sorted(observations["tactile"].keys()):
                tac_parts.append(
                    observations["tactile"][level].float().flatten(start_dim=1) / 255.0
                )
        
        # 处理本体感觉
        prop = observations["proprioception"].float()
        prop = torch.clamp(prop / np.pi, -1.0, 1.0)
        
        # 拼接触觉+本体感觉
        if tac_parts:
            tac_prop = torch.cat(tac_parts + [prop], dim=1)
        else:
            tac_prop = prop
        
        tac_prop_features = self.tac_prop_mlp(tac_prop)
        
        # 融合
        combined = torch.cat([cam_features, tac_prop_features], dim=1)
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
        
        # 包装器顺序：Monitor 最外层，然后是 Flatten
        env = FlattenObservationWrapper(env)
        env = Monitor(env)
        
        return env
    return _init


# ====================== 主训练流程 ======================

def main():
    N_ENVS = 4
    TOTAL_TIMESTEPS = 5_000_000
    SAVE_DIR = PROJECT_ROOT / "models" / "ppo_pickplace"
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
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    
    model = PPO(
        "MlpPolicy",
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