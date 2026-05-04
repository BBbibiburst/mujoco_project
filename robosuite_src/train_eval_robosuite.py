"""
train.py —— Robosuite 1.5 Kinova3FlippedGripper（InspireRightHand）PickPlace 并行训练
"""

import os
import warnings
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

import robosuite as suite
from robosuite.robots import register_robot_class
from robosuite.models.robots import Kinova3
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


# ─────────────────────────────────────────────
# 自定义机器人注册
# ─────────────────────────────────────────────

@register_robot_class("FixedBaseRobot")
class Kinova3FlippedGripper(Kinova3):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand"}

    @property
    def gripper_mount_quat_offset(self):
        euler_deg = [180, 0, -90]
        r = R.from_euler('xyz', euler_deg, degrees=True)
        w, x, y, z = r.as_quat()
        return {"right": [w, x, y, z]}


# ─────────────────────────────────────────────
# 环境工厂
# ─────────────────────────────────────────────

ROBOT_NAME = "Kinova3FlippedGripper"

def make_env(args, seed: int = 0, render: bool = False, reward_shaping: bool = True):
    def _init():
        # SubprocVecEnv 在子进程中运行，需要在 _init 内部重新注册机器人
        # 因为子进程不继承主进程的注册状态
        from scipy.spatial.transform import Rotation as _R
        from robosuite.robots import register_robot_class as _reg
        from robosuite.models.robots import Kinova3 as _Kinova3

        @_reg("FixedBaseRobot")
        class Kinova3FlippedGripper(_Kinova3):
            @property
            def default_gripper(self):
                return {"right": "InspireRightHand"}
            @property
            def gripper_mount_quat_offset(self):
                r = _R.from_euler('xyz', [180, 0, -90], degrees=True)
                w, x, y, z = r.as_quat()
                return {"right": [w, x, y, z]}

        controller_cfg = load_composite_controller_config(
            controller=None,
            robot="Kinova3",
        )

        env = suite.make(
            env_name="PickPlace",
            robots=ROBOT_NAME,
            controller_configs=controller_cfg,
            use_camera_obs=False,
            use_object_obs=True,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="frontview",
            reward_shaping=reward_shaping,
            reward_scale=1.0,
            horizon=args.horizon,
            ignore_done=False,
            single_object_mode=2,
            object_type="milk",
            control_freq=20,
        )

        env = GymWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def make_vec_env(args, n_envs: int, seed: int = 0,
                 render: bool = False, reward_shaping: bool = True,
                 use_subproc: bool = True):
    """
    use_subproc=True  → SubprocVecEnv（真并行，推荐 n_envs >= 4）
    use_subproc=False → DummyVecEnv（单进程，调试用）
    """
    fns = [make_env(args, seed=seed + i, render=render, reward_shaping=reward_shaping)
           for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        return SubprocVecEnv(fns, start_method="fork")   # Linux 用 fork 最快
    else:
        return DummyVecEnv(fns)


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
            learning_starts=max(10_000, args.n_envs * 1000),  # 并行时适当扩大预热
            batch_size=256 * max(1, args.n_envs // 4),        # 并行时适当扩大 batch
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
            learning_starts=max(10_000, args.n_envs * 1000),
            batch_size=256 * max(1, args.n_envs // 4),
            action_noise=noise,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
        )

    elif args.algo == "PPO":
        # PPO on-policy：n_steps 是每个环境采集的步数
        # 有效 batch = n_steps * n_envs，并行收益最直接
        return PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64 * max(1, args.n_envs),   # minibatch 随并行数扩大
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

    use_subproc = args.n_envs > 1  # 单环境没必要开子进程

    # 训练环境
    train_vec = make_vec_env(args, n_envs=args.n_envs, seed=args.seed,
                             reward_shaping=True, use_subproc=use_subproc)
    train_env = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 评估环境（固定单进程，避免渲染冲突）
    eval_vec = make_vec_env(args, n_envs=1, seed=args.seed + 999,
                            reward_shaping=True, use_subproc=False)
    eval_env = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, training=False)

    # 让 eval_env 的归一化统计与 train_env 同步
    eval_env.obs_rms = train_env.obs_rms

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
            name_prefix=f"{args.algo}_{ROBOT_NAME}",
        ),
    ])

    if args.load_model:
        cls = {"SAC": SAC, "TD3": TD3, "PPO": PPO}[args.algo]
        model = cls.load(args.load_model, env=train_env, device=args.device)
        print(f"[INFO] 续训模型: {args.load_model}")
    else:
        model = build_model(args, train_env)

    print(f"\n{'='*50}")
    print(f"  机器人  : {ROBOT_NAME}")
    print(f"  手爪    : InspireRightHand (内置)")
    print(f"  算法    : {args.algo}")
    print(f"  并行环境: {args.n_envs} ({'SubprocVecEnv' if use_subproc else 'DummyVecEnv'})")
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

    final = os.path.join(args.save_dir, f"final_{args.algo}_{ROBOT_NAME}")
    model.save(final)
    train_env.save(final + "_vecnorm.pkl")
    print(f"[INFO] 已保存: {final}")
    train_env.close()


# ─────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────

def evaluate(args):
    cls = {"SAC": SAC, "TD3": TD3, "PPO": PPO}[args.algo]

    env = make_vec_env(args, n_envs=1, seed=0, render=True,
                       reward_shaping=False, use_subproc=False)

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
    p.add_argument("--mode",          default="train", choices=["train", "eval"])
    p.add_argument("--algo",          default="SAC",   choices=["SAC", "TD3", "PPO"])
    p.add_argument("--horizon",       type=int, default=500)
    p.add_argument("--n_envs",        type=int, default=8,   # ← 默认8并行
                   help="并行环境数，建议设为 CPU 核心数的一半")
    p.add_argument("--total_steps",   type=int, default=2)
    p.add_argument("--eval_freq",     type=int, default=20_000)
    p.add_argument("--save_freq",     type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_dir",       default="logs/")
    p.add_argument("--save_dir",      default="models/kinova3_inspire/")
    p.add_argument("--load_model",    default="")
    p.add_argument("--device",        default="auto")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    # SubprocVecEnv 在 Windows/macOS 上需要此保护
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)  # Linux 默认，可省略

    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        assert args.load_model, "eval 模式需要 --load_model"
        evaluate(args)