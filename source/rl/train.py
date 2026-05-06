"""
train.py —— Robosuite 1.5 PickPlace 并行训练
支持:
  --robot default   → Kinova3FlippedGripper + InspireRightHand（自定义）
  --robot standard  → Panda + 默认夹爪
  --task  milk      → PickPlaceMilk（单物体，较易）
  --task  full      → PickPlace 全4物体（较难）
  两个参数完全独立，可任意组合。
断点重训: --resume（自动找最新 checkpoint）或 --load_model 指定路径

使用示例:
# 从头训练 Panda 在 PickPlaceMilk 上，使用 16 个并行环境，总步数 10M，保存频率 50k，评估频率 20k
python -m source.rl.train \
    --mode train \
    --algo PPO \
    --robot standard \
    --task milk \
    --n_envs 16 \
    --total_steps 10000000 \
    --eval_freq 20000 \
    --save_freq 50000 \
    --base_dir runs/
# 续训（自动找最新 checkpoint）
python -m source.rl.train \
    --mode train \
    --algo PPO \
    --robot standard \
    --task milk \
    --n_envs 16 \
    --total_steps 20000000 \
    --eval_freq 20000 \
    --save_freq 50000 \
    --base_dir runs/ \
    --resume
# 续训（手动指定 checkpoint）
python -m source.rl.train \
    --mode train \
    --algo PPO \
    --robot standard \
    --task milk \
    --n_envs 16 \
    --total_steps 20000000 \
    --eval_freq 20000 \
    --save_freq 50000 \
    --base_dir runs/ \
    --load_model runs/PPO_Panda_milk/20250505_153000/ckpt/PPO_Panda_milk_50000_steps
# 评估（手动指定模型）
python -m source.rl.train \
    --mode eval \
    --algo PPO \
    --robot standard \
    --task milk \
    --load_model runs/PPO_Panda_milk/20250505_153000/best/PPO_Panda_milk_150000_steps
"""

import os
import glob
import warnings
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


# ─────────────────────────────────────────────
# 任务配置表
# ─────────────────────────────────────────────

TASK_CONFIGS = {
    "milk": {
        # PickPlaceMilk 已内置 single_object_mode=2 + object_type="milk"
        # 直接用该类名，无需额外 kwargs
        "env_name":    "PickPlaceMilk",
        "extra_kwargs": {},
    },
    "full": {
        # 标准 PickPlace，4 个物体全部出现
        "env_name":    "PickPlace",
        "extra_kwargs": {"single_object_mode": 0},
    },
}


# ─────────────────────────────────────────────
# 机器人配置表
# ─────────────────────────────────────────────

ROBOT_CONFIGS = {
    "default": {
        "robot_name":   "Kinova3FlippedGripper",   # 注册后使用此名
        "gripper_type": "default",                  # 由 default_gripper 决定
        "controller":   None,                       # None → robosuite 默认
        "base_robot":   "Kinova3",                  # 用于加载 controller_config
        "custom":       True,                       # 需要动态注册
    },
    "standard": {
        "robot_name":   "Panda",
        "gripper_type": "default",
        "controller":   None,
        "base_robot":   "Panda",
        "custom":       False,
    },
}


# ─────────────────────────────────────────────
# 自定义机器人注册（主进程 & 子进程均需调用）
# ─────────────────────────────────────────────

def register_custom_robots():
    """注册所有自定义机器人，可安全重复调用。"""
    from robosuite.robots import register_robot_class
    from robosuite.models.robots import Kinova3

    @register_robot_class("FixedBaseRobot")
    class Kinova3FlippedGripper(Kinova3):
        @property
        def default_gripper(self):
            return {"right": "InspireRightHand"}

        @property
        def gripper_mount_quat_offset(self):
            r = R.from_euler('xyz', [180, 0, -90], degrees=True)
            w, x, y, z = r.as_quat()
            return {"right": [w, x, y, z]}


# 主进程立即注册
register_custom_robots()


# ─────────────────────────────────────────────
# 环境工厂
# ─────────────────────────────────────────────

def make_env(args, seed: int = 0, render: bool = False, reward_shaping: bool = True):
    robot_cfg = ROBOT_CONFIGS[args.robot]
    task_cfg  = TASK_CONFIGS[args.task]

    def _init():
        # 子进程不继承主进程注册状态，需重新注册
        if robot_cfg["custom"]:
            register_custom_robots()

        controller_cfg = load_composite_controller_config(
            controller=robot_cfg["controller"],
            robot=robot_cfg["base_robot"],
        )

        env = suite.make(
            env_name=task_cfg["env_name"],
            robots=robot_cfg["robot_name"],
            gripper_types=robot_cfg["gripper_type"],
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
            control_freq=20,
            **task_cfg["extra_kwargs"],
        )

        env = GymWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def make_vec_env(args, n_envs: int, seed: int = 0,
                 render: bool = False, reward_shaping: bool = True,
                 use_subproc: bool = True):
    fns = [make_env(args, seed=seed + i, render=render, reward_shaping=reward_shaping)
           for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(fns, start_method="fork")
    return DummyVecEnv(fns)


# ─────────────────────────────────────────────
# 断点重训工具
# ─────────────────────────────────────────────

def find_latest_checkpoint(search_root: str, algo: str, robot_task: str) -> str | None:
    """
    在 search_root 下递归搜索最新的 checkpoint。
    文件名格式: {algo}_{robot}_{task}_{steps}_steps.zip
    返回不含 .zip 的路径，或 None。
    """
    pattern = os.path.join(search_root, "**", f"{algo}_{robot_task}_*_steps.zip")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None

    def _extract_steps(path):
        base = os.path.basename(path)
        parts = base.replace(".zip", "").split("_")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0

    latest = max(files, key=_extract_steps)
    return latest.replace(".zip", "")


class SyncVecNormalizeCallback(BaseCallback):
    """训练时实时把 train_env 的 obs_rms 同步给 eval_env。"""
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env  = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        return True


# ─────────────────────────────────────────────
# 模型构建
# ─────────────────────────────────────────────

def build_model(args, env, log_dir: str):
    policy_kwargs = dict(net_arch=[512, 512, 256])

    if args.algo == "SAC":
        return SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=50_000,
            batch_size=256 * max(1, args.n_envs // 4),
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device=args.device,
        )

    elif args.algo == "TD3":
        n_actions = env.action_space.shape[0]
        noise = NormalActionNoise(np.zeros(n_actions), 0.1 * np.ones(n_actions))
        return TD3(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=50_000,
            batch_size=256 * max(1, args.n_envs // 4),
            action_noise=noise,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device=args.device,
        )

    elif args.algo == "PPO":
        return PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64 * max(1, args.n_envs),
            n_epochs=10,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device=args.device,
        )

    raise ValueError(f"未知算法: {args.algo}")


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────

def train(args):
    cfg        = ROBOT_CONFIGS[args.robot]
    algo_name  = args.algo
    robot_name = cfg["robot_name"]
    task_name  = args.task
    run_prefix = f"{algo_name}_{robot_name}_{task_name}"

    # ── 目录：base/{algo}_{robot}_{task}/{timestamp}  ──────────────
    # 续训（--resume 或 --load_model）时：从路径中解析出已有的 run_dir，
    # 保持同一目录，不新建时间戳子目录。
    # 新训时：用当前时间创建新目录。
    resuming = bool(args.load_model) or args.resume

    if resuming and args.load_model:
        # load_model 形如 .../models/PPO_Panda_milk/20250505_153000/ckpt/PPO_Panda_milk_50000_steps
        # 向上两级就是 run_dir
        run_dir = os.path.dirname(os.path.dirname(args.load_model))
        print(f"[RESUME] 复用已有目录: {run_dir}")
    else:
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir   = os.path.join(args.base_dir, run_prefix, timestamp)
        if resuming:
            # --resume 但没有 load_model，目录由 find_latest_checkpoint 后确定，先用时间戳占位
            pass
        print(f"[INFO] 运行目录: {run_dir}")

    ckpt_dir = os.path.join(run_dir, "ckpt")
    best_dir = os.path.join(run_dir, "best")
    log_dir  = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    use_subproc = args.n_envs > 1

    # ── 训练环境 ──────────────────────────────
    train_vec = make_vec_env(args, n_envs=args.n_envs, seed=args.seed,
                             reward_shaping=True, use_subproc=use_subproc)
    train_env = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── 评估环境 ──────────────────────────────
    eval_vec = make_vec_env(args, n_envs=1, seed=args.seed + 999,
                            reward_shaping=True, use_subproc=False)
    eval_env = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = train_env.obs_rms   # 初始对齐，后续由 callback 持续同步

    # ── 断点重训逻辑 ──────────────────────────
    resume_model_path  = None
    resume_vecnorm_path = None
    reset_timesteps    = True

    if args.load_model:
        resume_model_path   = args.load_model
        resume_vecnorm_path = args.load_model + "_vecnorm.pkl"
        reset_timesteps     = False
        print(f"[RESUME] 手动指定模型: {resume_model_path}")

    elif args.resume:
        # 在 save_dir/{run_prefix}/ 下搜索所有时间戳子目录里的 checkpoint
        search_root = os.path.join(args.base_dir, run_prefix)
        latest = find_latest_checkpoint(search_root, algo_name, f"{robot_name}_{task_name}")
        if latest:
            resume_model_path   = latest
            # vecnorm 与 checkpoint 放在同一目录
            resume_vecnorm_path = latest + "_vecnorm.pkl"
            # 复用 checkpoint 所在的 run_dir
            run_dir  = os.path.dirname(os.path.dirname(latest))
            ckpt_dir = os.path.join(run_dir, "ckpt")
            best_dir = os.path.join(run_dir, "best")
            log_dir  = os.path.join(run_dir, "logs")
            reset_timesteps = False
            print(f"[RESUME] 自动找到最新 checkpoint: {latest}.zip")
            print(f"[RESUME] 复用目录: {run_dir}")
        else:
            print("[RESUME] 未找到任何 checkpoint，从头开始训练。")

    # ── 加载归一化统计 ─────────────────────────
    if resume_vecnorm_path and os.path.exists(resume_vecnorm_path):
        saved_vecnorm     = VecNormalize.load(resume_vecnorm_path, train_vec)
        train_env.obs_rms = saved_vecnorm.obs_rms
        train_env.ret_rms = saved_vecnorm.ret_rms
        eval_env.obs_rms  = saved_vecnorm.obs_rms
        print(f"[RESUME] 归一化统计已恢复: {resume_vecnorm_path}")

    # ── 构建或加载模型 ─────────────────────────
    cls = {"SAC": SAC, "TD3": TD3, "PPO": PPO}[algo_name]
    if resume_model_path:
        model = cls.load(resume_model_path, env=train_env, device=args.device)
        print(f"[RESUME] 模型权重已加载，reset_timesteps={reset_timesteps}")
    else:
        model = build_model(args, train_env, log_dir)

    # ── Callbacks ─────────────────────────────
    callbacks = CallbackList([
        SyncVecNormalizeCallback(train_env, eval_env),
        EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=ckpt_dir,
            name_prefix=run_prefix,
            save_vecnormalize=True,
        ),
    ])

    print(f"\n{'='*50}")
    print(f"  运行目录: {run_dir}")
    print(f"  机器人  : {robot_name}")
    print(f"  任务    : {task_name}  ({TASK_CONFIGS[args.task]['env_name']})")
    print(f"  算法    : {algo_name}")
    print(f"  并行环境: {args.n_envs} ({'SubprocVecEnv' if use_subproc else 'DummyVecEnv'})")
    print(f"  总步数  : {args.total_steps:,}")
    print(f"  续训    : {'是' if not reset_timesteps else '否'}")
    print(f"  动作维度: {train_env.action_space.shape[0]}")
    print(f"  观测维度: {train_env.observation_space.shape[0]}")
    print(f"{'='*50}\n")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        reset_num_timesteps=reset_timesteps,
        progress_bar=True,
    )

    # ── 保存最终模型 ───────────────────────────
    final_prefix = os.path.join(run_dir, f"final_{run_prefix}")
    model.save(final_prefix)
    train_env.save(final_prefix + "_vecnorm.pkl")
    print(f"[INFO] 已保存最终模型: {final_prefix}.zip")

    train_env.close()
    eval_env.close()


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
        env.training   = False
        env.norm_reward = False
        print(f"[INFO] 已加载归一化统计: {vecnorm_path}")

    model = cls.load(args.load_model, env=env, device=args.device)

    for ep in range(args.eval_episodes):
        obs  = env.reset()
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
    p.add_argument("--mode",          default="train",    choices=["train", "eval"])
    p.add_argument("--algo",          default="PPO",      choices=["SAC", "TD3", "PPO"])
    p.add_argument("--robot",         default="default",  choices=["default", "standard"],
                   help="default=Kinova3+InspireRightHand  standard=Panda+默认夹爪")
    p.add_argument("--task",          default="milk",     choices=["milk", "full"],
                   help="milk=PickPlaceMilk(单物体)  full=PickPlace(全4物体)")
    p.add_argument("--horizon",       type=int, default=1000)
    p.add_argument("--n_envs",        type=int, default=32)
    p.add_argument("--total_steps",   type=int, default=20_000_000)
    p.add_argument("--eval_freq",     type=int, default=20_000)
    p.add_argument("--save_freq",     type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--base_dir",      default="runs/",
                   help="所有训练产物的根目录，子目录结构自动生成: base_dir/{algo}_{robot}_{task}/{timestamp}/")
    p.add_argument("--load_model",    default="",
                   help="指定模型路径（不含 .zip）手动续训或评估")
    p.add_argument("--resume",        action="store_true",
                   help="自动找最新 checkpoint 续训，无需手动指定路径")
    p.add_argument("--device",        default="auto")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        assert args.load_model, "eval 模式需要 --load_model"
        evaluate(args)