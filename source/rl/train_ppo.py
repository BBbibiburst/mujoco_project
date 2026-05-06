"""
通用 PPO 训练脚本 — 支持所有继承自 RobotArmEnvBase 的任务环境.

用法：
    # 从项目根目录执行
    python -m source.rl.train_ppo --task pick_place
    python -m source.rl.train_ppo --task stack
    python -m source.rl.train_ppo --task insert
    python -m source.rl.train_ppo --task reorient
    python -m source.rl.train_ppo --task push

完整参数示例：
    python -m source.rl.train_ppo \\
        --task insert \\
        --n-envs 8 \\
        --total-steps 10_000_000 \\
        --action-mode joint \\
        --controller osc \\
        --n-steps 2048 \\
        --batch-size 256 \\
        --eval-episodes 10 \\
        --resume rl_models/ppo_insert/checkpoints/ppo_insert_1000000_steps.zip

修复：
    VecTransposeImage 误判触觉数据为图像空间的问题（通过 TactileShapeWrapper 解决）

特性：
    - 统一的任务注册表，一套代码训练全部5个任务
    - 可选 EvalCallback（自动评估并保存最优模型）
    - 断点续训（--resume）
    - 自动按任务名和时间戳管理存储路径
    - 多进程（SubprocVecEnv）/ 单进程（DummyVecEnv）自动切换
"""

import sys
import importlib
import argparse
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from source.env.base_env import RobotConfig


# ====================== 任务注册表 ======================
# 与 demo.py 对齐，方便后续统一维护

TASK_REGISTRY = {
    "pick_place": {
        "module":    "source.env.pick_place_env",
        "env_class": "PickPlaceEnv",
        "cfg_class": "PickPlaceConfig",
        "display_name": "抓取放置",
        "default_cfg_kwargs": {
            "r_step_penalty": -0.005,
            "r_place_bonus":  100.0,
            "r_grasp_bonus":  10.0,
            "reach_threshold": 0.05,
            "grasp_threshold": 0.04,
        },
        # 推荐超参（可被命令行覆盖）
        "recommended_hp": {
            "max_episode_steps": 200,
            "n_steps":           2048,
            "batch_size":        64,
            "total_steps":       5_000_000,
        },
    },
    "stack": {
        "module":    "source.env.stack_env",
        "env_class": "StackEnv",
        "cfg_class": "StackConfig",
        "display_name": "堆叠",
        "default_cfg_kwargs": {},
        "recommended_hp": {
            "max_episode_steps": 300,
            "n_steps":           2048,
            "batch_size":        128,
            "total_steps":       8_000_000,
        },
    },
    "insert": {
        "module":    "source.env.insert_env",
        "env_class": "InsertEnv",
        "cfg_class": "InsertConfig",
        "display_name": "插孔（精密装配）",
        "default_cfg_kwargs": {},
        # 插孔任务最难，建议更长训练和更大批次
        "recommended_hp": {
            "max_episode_steps": 400,
            "n_steps":           4096,
            "batch_size":        256,
            "total_steps":       15_000_000,
        },
    },
    "reorient": {
        "module":    "source.env.reorient_env",
        "env_class": "ReorientEnv",
        "cfg_class": "ReorientConfig",
        "display_name": "重定向（姿态控制）",
        "default_cfg_kwargs": {},
        "recommended_hp": {
            "max_episode_steps": 300,
            "n_steps":           2048,
            "batch_size":        128,
            "total_steps":       10_000_000,
        },
    },
    "push": {
        "module":    "source.env.push_env",
        "env_class": "PushEnv",
        "cfg_class": "PushConfig",
        "display_name": "推动",
        "default_cfg_kwargs": {},
        # 推动任务相对简单，训练步数可以少
        "recommended_hp": {
            "max_episode_steps": 200,
            "n_steps":           2048,
            "batch_size":        64,
            "total_steps":       3_000_000,
        },
    },
}


# ====================== 观测包装器：触觉维度扩展 ======================

class TactileShapeWrapper(gym.ObservationWrapper):
    """
    将触觉数据从 (5, H, W) 扩展为 (5, H, W, 1)，避免被 SB3 VecTransposeImage 误判为图像。

    SB3 的 is_image_space 对 3D Box 且 shape[-1] 较小时会误判为图像，从而强制
    对其做 (H, W, C) → (C, H, W) 转置，破坏触觉数据的语义。
    扩展为 4D（len(shape)==4）后，SB3 不再识别为图像，从而跳过转置。
    """

    TACTILE_KEYS = ["tactile_bottom", "tactile_middle", "tactile_top"]

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), (
            f"TactileShapeWrapper 需要 Dict 观测空间，实际: {type(env.observation_space)}"
        )
        new_spaces = dict(env.observation_space.spaces)
        for key in self.TACTILE_KEYS:
            if key in new_spaces:
                old_shape = new_spaces[key].shape          # (5, H, W)
                new_spaces[key] = spaces.Box(
                    low=0, high=255,
                    shape=(*old_shape, 1),                  # (5, H, W, 1)
                    dtype=np.uint8,
                )
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs: dict) -> dict:
        new_obs = dict(obs)
        for key in self.TACTILE_KEYS:
            if key in new_obs:
                new_obs[key] = np.expand_dims(new_obs[key], axis=-1)  # (...) → (..., 1)
        return new_obs


# ====================== 多模态 CNN 特征提取器 ======================

class MultiModalFeatureExtractor(BaseFeaturesExtractor):
    """
    多模态特征提取器，处理扁平化 Dict 观测空间。

    各分支：
        camera_rgb      (C, H, W)       → 3层 Conv2D + Flatten
        tactile_bottom  (5, H, W, 1)    → squeeze → 2层 Conv2D + Flatten
        tactile_middle  (5, H, W, 1)    → 同上
        tactile_top     (5, H, W, 1)    → 同上
        proprioception  (13,)           → 2层 Linear

    所有分支输出拼接后过一个全连接融合层，输出 features_dim 维特征。

    policy_kwargs 用法：
        policy_kwargs = dict(
            features_extractor_class=MultiModalFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
        )
    """

    TACTILE_KEYS = ["tactile_bottom", "tactile_middle", "tactile_top"]

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # ---- 1. 相机 CNN ----
        cam_space = observation_space["camera_rgb"]
        C, H, W = cam_space.shape   # VecTransposeImage 已将其变为 channel-first

        self.camera_cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cam_flat_dim = self.camera_cnn(torch.zeros(1, C, H, W)).shape[1]
        total_concat = cam_flat_dim

        # ---- 2. 触觉 CNN ----
        self.tactile_cnns = nn.ModuleDict()
        for key in self.TACTILE_KEYS:
            if key not in observation_space.spaces:
                continue
            n_fingers, rows, cols, _ = observation_space[key].shape   # (5, H, W, 1)
            cnn = nn.Sequential(
                nn.Conv2d(n_fingers, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                flat_dim = cnn(torch.zeros(1, n_fingers, rows, cols)).shape[1]
            self.tactile_cnns[key] = cnn
            total_concat += flat_dim

        # ---- 3. 本体感觉 MLP ----
        prop_dim = int(np.prod(observation_space["proprioception"].shape))
        self.proprio_mlp = nn.Sequential(
            nn.Linear(prop_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        total_concat += 64

        # ---- 4. 融合层 ----
        self.fusion = nn.Sequential(
            nn.Linear(total_concat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        encoded = []

        # 相机：VecTransposeImage 已保证 channel-first
        cam = observations["camera_rgb"].float() / 255.0
        encoded.append(self.camera_cnn(cam))

        # 触觉：squeeze 最后一维后送入 CNN
        for key in self.TACTILE_KEYS:
            if key in observations:
                tac = observations[key].float() / 255.0   # (B, 5, H, W, 1)
                tac = tac.squeeze(-1)                      # (B, 5, H, W)
                encoded.append(self.tactile_cnns[key](tac))

        # 本体感觉：归一化到 [-1, 1]
        prop = torch.clamp(observations["proprioception"].float() / np.pi, -1.0, 1.0)
        encoded.append(self.proprio_mlp(prop))

        return self.fusion(torch.cat(encoded, dim=1))


# ====================== 环境工厂 ======================

def make_env(
    task_name: str,
    robot_cfg: RobotConfig,
    task_cfg_kwargs: dict,
    rank: int = 0,
    seed: int = 0,
):
    """
    工厂函数：创建并包装单个任务环境，供 DummyVecEnv / SubprocVecEnv 使用.

    包装顺序（从内到外）：
        TaskEnv
          └─ TactileShapeWrapper   （触觉维度扩展，避免被误识别为图像）
               └─ Monitor          （记录 episode reward/length，供 EvalCallback 使用）
    """
    reg = TASK_REGISTRY[task_name]

    def _init():
        mod = importlib.import_module(reg["module"])
        EnvClass = getattr(mod, reg["env_class"])
        CfgClass = getattr(mod, reg["cfg_class"])

        # 合并默认配置和用户自定义配置
        merged_kwargs = {**reg["default_cfg_kwargs"], **task_cfg_kwargs}
        task_cfg = CfgClass(**merged_kwargs)

        env = EnvClass(robot_config=robot_cfg, task_config=task_cfg)
        env = TactileShapeWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# ====================== 主训练函数 ======================

def train(args: argparse.Namespace) -> None:
    task_name = args.task
    reg = TASK_REGISTRY[task_name]
    rec = reg["recommended_hp"]

    # ---------- 超参（命令行 > 推荐值）----------
    max_episode_steps = args.max_episode_steps or rec["max_episode_steps"]
    n_steps           = args.n_steps           or rec["n_steps"]
    batch_size        = args.batch_size        or rec["batch_size"]
    total_steps       = args.total_steps       or rec["total_steps"]
    n_envs            = args.n_envs

    # ---------- 保存路径 ----------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"ppo_{task_name}_{timestamp}" if not args.run_name else args.run_name
    save_dir  = PROJECT_ROOT / "rl_models" / task_name / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir  = save_dir / "checkpoints"
    eval_dir  = save_dir / "best_model"
    tb_dir    = save_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  任务: {reg['display_name']} ({task_name})")
    print(f"  运行: {run_name}")
    print(f"  并行环境数: {n_envs}")
    print(f"  总训练步数: {total_steps:,}")
    print(f"  n_steps={n_steps}, batch_size={batch_size}")
    print(f"  max_episode_steps={max_episode_steps}")
    print(f"  action_mode={args.action_mode}, controller={args.controller}")
    print(f"  保存路径: {save_dir}")
    print("=" * 70)

    # ---------- RobotConfig ----------
    robot_cfg = RobotConfig(
        action_mode=args.action_mode,
        controller_type=args.controller,
        max_episode_steps=max_episode_steps,
        action_scale=args.action_scale,
        action_scale_rot=args.action_scale_rot,
        control_freq=args.control_freq,
        sim_freq=args.sim_freq,
        tactile_backend=args.tactile_backend,
        init_arm_qpos=np.array([0.0, 0.5, 0.0, 1.5, 0.0, -1.0, 0.0]),
        init_hand_qpos=np.zeros(6),
    )

    # ---------- 向量化训练环境 ----------
    env_fns = [
        make_env(task_name, robot_cfg, {}, rank=i, seed=args.seed + i)
        for i in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns)

    # ---------- 向量化评估环境（独立，不影响训练）----------
    eval_env = None
    if args.eval_episodes > 0:
        eval_env_fns = [
            make_env(task_name, robot_cfg, {}, rank=0, seed=args.seed + 9999)
        ]
        eval_env = DummyVecEnv(eval_env_fns)

    # ---------- 回调 ----------
    callbacks = []

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=f"ppo_{task_name}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_cb)

    if eval_env is not None and args.eval_episodes > 0:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(eval_dir),
            log_path=str(eval_dir),
            eval_freq=max(args.eval_freq // n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callbacks.append(eval_cb)

    callback = CallbackList(callbacks)

    # ---------- 策略网络参数 ----------
    policy_kwargs = dict(
        features_extractor_class=MultiModalFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    # ---------- 创建或恢复 PPO 模型 ----------
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"恢复路径不存在: {resume_path}")
        print(f"\n  [恢复训练] 加载模型: {resume_path}")
        model = PPO.load(
            str(resume_path),
            env=vec_env,
            device=args.device,
            # 注意：恢复时以下超参以加载模型中的为准，
            # 若需修改请手动指定 custom_objects
        )
        model.set_env(vec_env)
        remaining_steps = total_steps - model.num_timesteps
        if remaining_steps <= 0:
            print(f"  已达到目标步数 {total_steps:,}，无需继续训练。")
            vec_env.close()
            return
        print(f"  已训练步数: {model.num_timesteps:,}，剩余: {remaining_steps:,}")
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=None,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(tb_dir),
            policy_kwargs=policy_kwargs,
            device=args.device,
            seed=args.seed,
        )
        remaining_steps = total_steps

    # ---------- 打印模型结构摘要 ----------
    print(f"\n  策略网络参数量: {sum(p.numel() for p in model.policy.parameters()):,}")

    # ---------- 训练 ----------
    print(f"\n{'='*70}")
    print(f"  开始训练 — 任务: {reg['display_name']}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"{'='*70}\n")

    model.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        reset_num_timesteps=not bool(args.resume),
        progress_bar=True,
    )

    # ---------- 保存最终模型 ----------
    final_path = save_dir / f"ppo_{task_name}_final.zip"
    model.save(str(final_path))
    print(f"\n✓ 训练完成！最终模型: {final_path}")

    # ---------- 清理 ----------
    vec_env.close()
    if eval_env is not None:
        eval_env.close()


# ====================== 参数解析 ======================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通用 PPO 训练脚本（支持5个灵巧手任务）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 任务 ----
    parser.add_argument(
        "--task",
        choices=list(TASK_REGISTRY.keys()),
        default="pick_place",
        help="训练任务名称",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="运行名称（用于保存路径，默认自动生成含时间戳的名称）",
    )

    # ---- 机器人与控制器 ----
    parser.add_argument("--action-mode",  choices=["joint", "ee"], default="joint")
    parser.add_argument("--controller",   choices=["osc", "ik"], default="osc")
    parser.add_argument("--action-scale",     type=float, default=0.03)
    parser.add_argument("--action-scale-rot", type=float, default=0.06)
    parser.add_argument("--control-freq",     type=float, default=20.0)
    parser.add_argument("--sim-freq",         type=float, default=1000.0)
    parser.add_argument("--tactile-backend",  choices=["simple", "physics", "simple_avg", "physics_avg"],
                        default="simple_avg")

    # ---- 训练超参 ----
    parser.add_argument("--n-envs",      type=int,   default=32,
                        help="并行环境数（1 → DummyVecEnv，>1 → SubprocVecEnv）")
    parser.add_argument("--total-steps", type=int,   default=None,
                        help="总训练步数（None → 使用任务推荐值）")
    parser.add_argument("--n-steps",     type=int,   default=None,
                        help="每次 rollout 的步数（None → 任务推荐值）")
    parser.add_argument("--batch-size",  type=int,   default=None,
                        help="SGD mini-batch 大小（None → 任务推荐值）")
    parser.add_argument("--n-epochs",    type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--gae-lambda",  type=float, default=0.95)
    parser.add_argument("--clip-range",  type=float, default=0.2)
    parser.add_argument("--ent-coef",    type=float, default=0.01)
    parser.add_argument("--max-episode-steps", type=int, default=None,
                        help="单回合最大步数（None → 任务推荐值）")

    # ---- 网络结构 ----
    parser.add_argument("--features-dim", type=int, default=512,
                        help="多模态特征提取器输出维度")

    # ---- 评估 ----
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="评估回合数（0 → 禁用 EvalCallback）")
    parser.add_argument("--eval-freq",     type=int, default=50_000,
                        help="评估频率（总环境步数）")

    # ---- 保存 ----
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                        help="checkpoint 保存频率（总环境步数）")

    # ---- 续训 ----
    parser.add_argument("--resume", type=str, default=None,
                        help="从已有 .zip 文件恢复训练（路径）")

    # ---- 其他 ----
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    parser.add_argument("--seed",   type=int, default=42)

    return parser.parse_args()


# ====================== 入口 ======================

if __name__ == "__main__":
    args = parse_args()
    train(args)