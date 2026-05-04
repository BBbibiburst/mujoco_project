"""
train.py —— Robosuite 1.5 灵巧手 PickPlace 训练
完全使用 robosuite 内置奖励与功能，不添加额外包装
"""

import os
import argparse
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


# ─────────────────────────────────────────────
# 环境工厂
# ─────────────────────────────────────────────

HAND_MAP = {
    "inspire": "InspireHands",
    "shadow":  "ShadowHands",
    "allegro": "AllegroHands",
    "default": "default",
}

def make_env(args, seed: int = 0, render: bool = False, reward_shaping: bool = True):
    """
    返回一个 callable，供 DummyVecEnv 使用。
    完全使用 robosuite 原生配置，不套任何自定义 wrapper。
    """
    def _init():
        # robosuite 1.5 灵巧手需要复合控制器
        # 手臂默认 OSC_POSE，灵巧手部分默认 JOINT_POSITION
        controller_cfg = load_composite_controller_config(
            controller=None,      # None = 使用机器人默认复合配置
            robot=args.robot,
        )

        env = suite.make(
            env_name="PickPlace",
            robots=args.robot,
            gripper_types=HAND_MAP.get(args.hand, "InspireHands"),
            controller_configs=controller_cfg,

            # ── 观测 ──────────────────────────────
            use_camera_obs=False,
            use_object_obs=True,      # 包含物体位姿状态

            # ── 渲染 ──────────────────────────────
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="frontview",

            # ── 奖励（直接用内置分阶段密集奖励）──
            reward_shaping=reward_shaping,
            reward_scale=1.0,

            # ── 任务 ──────────────────────────────
            horizon=args.horizon,
            ignore_done=False,
            single_object_mode=2,     # 2 = 固定单物体，最易入门
            object_type="milk",

            # ── 控制 ──────────────────────────────
            control_freq=20,
        )

        # GymWrapper：robosuite → gymnasium 接口，自动拼接所有 obs key
        env = GymWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


# ─────────────────────────────────────────────
# 模型构建
# ─────────────────────────────────────────────

def build_model(args, env):
    policy_kwargs = dict(net_arch=[512, 512, 256])

    if args.algo == "SAC":
        return SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
        )

    elif args.algo == "TD3":
        n_actions = env.action_space.shape[0]
        noise = NormalActionNoise(np.zeros(n_actions), 0.1 * np.ones(n_actions))
        return TD3(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            action_noise=noise,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
        )

    elif args.algo == "PPO":
        return PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
        )

    raise ValueError(f"未知算法: {args.algo}")


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────

def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练环境
    train_env = DummyVecEnv([
        make_env(args, seed=args.seed + i, reward_shaping=True)
        for i in range(args.n_envs)
    ])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 评估环境（关闭奖励归一化，保持原始奖励尺度）
    eval_env = DummyVecEnv([make_env(args, seed=args.seed + 999, reward_shaping=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # 回调
    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.save_dir, "best"),
            log_path=os.path.join(args.log_dir, "eval"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=os.path.join(args.save_dir, "ckpt"),
            name_prefix=f"{args.algo}_{args.robot}",
        ),
    ])

    # 模型
    if args.load_model:
        cls = {"SAC": SAC, "TD3": TD3, "PPO": PPO}[args.algo]
        model = cls.load(args.load_model, env=train_env, device=args.device)
        print(f"[INFO] 续训模型: {args.load_model}")
    else:
        model = build_model(args, train_env)

    print(f"\n{'='*50}")
    print(f"  机器人  : {args.robot}")
    print(f"  灵巧手  : {HAND_MAP.get(args.hand)}")
    print(f"  算法    : {args.algo}")
    print(f"  总步数  : {args.total_steps:,}")
    print(f"  动作维度: {train_env.action_space.shape[0]}")
    print(f"  观测维度: {train_env.observation_space.shape[0]}")
    print(f"{'='*50}\n")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        reset_num_timesteps=not bool(args.load_model),
        progress_bar=True,
    )

    final = os.path.join(args.save_dir, f"final_{args.algo}_{args.robot}")
    model.save(final)
    train_env.save(final + "_vecnorm.pkl")
    print(f"[INFO] 已保存: {final}")


# ─────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────

def evaluate(args):
    cls = {"SAC": SAC, "TD3": TD3, "PPO": PPO}[args.algo]

    env = DummyVecEnv([make_env(args, seed=0, render=True, reward_shaping=False)])

    # 如果训练时保存了 VecNormalize 统计，加载以保持一致
    vecnorm_path = args.load_model + "_vecnorm.pkl"
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"[INFO] 已加载归一化统计: {vecnorm_path}")

    model = cls.load(args.load_model, env=env, device=args.device)

    for ep in range(args.eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
        print(f"Episode {ep+1:2d} | 累计奖励: {total_reward:.3f}")

    env.close()


# ─────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",         default="train", choices=["train", "eval"])
    p.add_argument("--robot",        default="GR1ArmsOnly")
    p.add_argument("--hand",         default="inspire",
                   choices=["inspire", "shadow", "allegro", "default"])
    p.add_argument("--algo",         default="SAC", choices=["SAC", "TD3", "PPO"])
    p.add_argument("--horizon",      type=int, default=500)
    p.add_argument("--n_envs",       type=int, default=1)
    p.add_argument("--total_steps",  type=int, default=2_000_000)
    p.add_argument("--eval_freq",    type=int, default=20_000)
    p.add_argument("--save_freq",    type=int, default=50_000)
    p.add_argument("--eval_episodes",type=int, default=10)
    p.add_argument("--log_dir",      default="logs/")
    p.add_argument("--save_dir",     default="models/")
    p.add_argument("--load_model",   default="")
    p.add_argument("--device",       default="auto")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        assert args.load_model, "eval 模式需要 --load_model"
        evaluate(args)