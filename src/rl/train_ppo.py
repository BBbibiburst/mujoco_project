"""
PPO 训练脚本 — 扁平化观测版本（最快跑通）
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


# ====================== 环境包装器：扁平化观测 ======================

class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    将 Dict 观测空间扁平化为 Box，适配 SB3 标准策略。
    
    输入: {
        "camera_rgb": (240, 320, 3) uint8,
        "tactile": {
            "bottom": (5, 7, 10) uint8,
            "middle": (5, 5, 8) uint8,
            "top": (5, 5, 6) uint8,
        },
        "proprioception": (13,) float32,
    }
    
    输出: (flattened_dim,) float32 — 所有数据归一化后拼接
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # 计算扁平化后的维度
        cam_dim = 240 * 320 * 3          # 230,400
        tac_bottom = 5 * 7 * 10          # 350
        tac_middle = 5 * 5 * 8           # 200
        tac_top = 5 * 5 * 6              # 150
        prop_dim = 13
        
        self.flattened_dim = cam_dim + tac_bottom + tac_middle + tac_top + prop_dim
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.flattened_dim,), 
            dtype=np.float32
        )
        
        print(f"[FlattenWrapper] 观测维度: {self.flattened_dim:,}")
        # 打印各组件占比，帮你决定是否要降维
        print(f"  camera: {cam_dim:,} ({cam_dim/self.flattened_dim*100:.1f}%)")
        print(f"  tactile: {tac_bottom+tac_middle+tac_top:,} ({(tac_bottom+tac_middle+tac_top)/self.flattened_dim*100:.1f}%)")
        print(f"  proprioception: {prop_dim} ({prop_dim/self.flattened_dim*100:.1f}%)")

    def observation(self, obs: dict) -> np.ndarray:
        # 1. 相机图像：uint8 [0,255] → float32 [0,1]
        cam = obs["camera_rgb"].astype(np.float32).flatten() / 255.0
        
        # 2. 触觉：uint8 [0,255] → float32 [0,1]
        tac_bottom = obs["tactile"]["bottom"].astype(np.float32).flatten() / 255.0
        tac_middle = obs["tactile"]["middle"].astype(np.float32).flatten() / 255.0
        tac_top = obs["tactile"]["top"].astype(np.float32).flatten() / 255.0
        
        # 3. 本体感觉：已经是 float32，做简单归一化（假设关节范围在合理区间）
        prop = obs["proprioception"].astype(np.float32)
        # 机械臂关节 [-pi, pi]，手部 [0, 0.01] 左右，统一缩放
        prop = np.clip(prop / np.pi, -1.0, 1.0)
        
        return np.concatenate([cam, tac_bottom, tac_middle, tac_top, prop])


# ====================== 可选：CNN 特征提取器（处理高维图像） ======================

class CustomCNN(BaseFeaturesExtractor):
    """
    当观测包含图像时，用 CNN 提取特征比全连接层高效得多。
    这里处理 240x320x3 的相机图像 + 触觉数据。
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 输入是扁平化的，我们需要知道各组件的边界
        # 但为了简单，这里假设输入维度固定（你可以改进为动态解析）
        self.cam_dim = 240 * 320 * 3
        self.tac_dim = 5*7*10 + 5*5*8 + 5*5*6  # 700
        self.prop_dim = 13
        
        # CNN 处理相机图像 (reshape back to 240x320x3)
        self.camera_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # -> 58x79x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> 28x38x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> 26x36x64
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算 CNN 输出维度
        with torch.no_grad():
            sample = torch.zeros(1, 3, 240, 320)
            cam_features_dim = self.camera_cnn(sample).shape[1]
        
        # MLP 处理触觉 + 本体感觉
        self.tac_prop_mlp = nn.Sequential(
            nn.Linear(self.tac_dim + self.prop_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(cam_features_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 切分输入
        cam_flat = observations[:, :self.cam_dim]
        tac_prop = observations[:, self.cam_dim:]
        
        # 重塑并处理相机图像
        cam = cam_flat.reshape(-1, 3, 240, 320)
        cam_features = self.camera_cnn(cam)
        
        # 处理触觉+本体感觉
        tac_prop_features = self.tac_prop_mlp(tac_prop)
        
        # 融合
        combined = torch.cat([cam_features, tac_prop_features], dim=1)
        return self.fusion(combined)


# ====================== 环境工厂 ======================

def make_env(rank: int = 0, seed: int = 0):
    """创建并包装环境（用于多进程向量环境）."""
    def _init():
        robot_cfg = RobotConfig(
            action_mode="osc_pose",      # 12维：xyz位移(3) + rpy旋转(3) + 手部(6)
            controller_type="osc",        # OSC 底层控制器
            max_episode_steps=200,
            action_scale=0.03,            # 位置增量缩放
            action_scale_rot=0.06,        # 旋转增量缩放
            control_freq=20.0,            # 20Hz 控制频率
            sim_freq=1000.0,              # 1kHz 物理仿真
            tactile_backend="simple_avg", # 轻量级触觉（训练快）
            # 初始姿态：稍微抬起来避免地面碰撞
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
        env.reset(seed=seed + rank)
        
        # 包装器顺序很重要：Monitor 必须在最外层（记录奖励），然后是 Flatten
        env = FlattenObservationWrapper(env)
        env = Monitor(env)
        
        return env
    return _init


# ====================== 主训练流程 ======================

def main():
    # 配置
    N_ENVS = 4               # 并行环境数（根据你的 CPU 核心数调整）
    TOTAL_TIMESTEPS = 5_000_000
    SAVE_DIR = PROJECT_ROOT / "models" / "ppo_pickplace"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"=== PPO 训练启动 ===")
    print(f"  并行环境数: {N_ENVS}")
    print(f"  总训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"  模型保存路径: {SAVE_DIR}")
    
    # 创建向量环境
    if N_ENVS == 1:
        vec_env = DummyVecEnv([make_env(0, seed=42)])
    else:
        vec_env = SubprocVecEnv([make_env(i, seed=42+i) for i in range(N_ENVS)])
    
    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(SAVE_DIR / "checkpoints"),
        name_prefix="ppo_pickplace",
    )
    
    # 可选：评估回调（需要另一个独立环境）
    # eval_env = DummyVecEnv([make_env(999, seed=999)])
    # eval_callback = EvalCallback(eval_env, best_model_save_path=str(SAVE_DIR / "best"))
    
    # 初始化 PPO
    # 如果要使用 CustomCNN，取消下面 policy_kwargs 的注释
    policy_kwargs = dict(
        # features_extractor_class=CustomCNN,
        # features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # 策略网络和价值网络结构
    )
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,           # 每个环境收集 2048 步再更新
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # 熵系数，鼓励探索
        verbose=1,
        tensorboard_log=str(SAVE_DIR / "tensorboard"),
        policy_kwargs=policy_kwargs,
        device="auto",          # 自动选择 cuda/cpu
    )
    
    print("\n=== 开始训练 ===")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    
    # 保存最终模型
    final_path = SAVE_DIR / "ppo_pickplace_final.zip"
    model.save(final_path)
    print(f"\n✓ 训练完成！模型已保存到: {final_path}")
    
    vec_env.close()


if __name__ == "__main__":
    main()