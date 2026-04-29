"""
通用 PPO 模型评估与可视化脚本.

支持所有继承自 RobotArmEnvBase 的任务环境，通过 --task 参数切换任务。

用法：
    python -m src.rl.eval_ppo --task pick_place --model rl_models/pick_place/.../ppo_pick_place_final.zip
    python -m src.rl.eval_ppo --task insert     --model rl_models/insert/.../best_model/best_model.zip
    python -m src.rl.eval_ppo --task push       --model path/to/model.zip --episodes 20 --no-render

适配：
    - MultiInputPolicy + Dict Observation
    - TactileShapeWrapper（触觉维度扩展）
    - 所有5个任务的任务特有 info 字段
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import cv2
import mujoco
from stable_baselines3 import PPO

# 复用 train_ppo 中的公共组件（任务注册表、TactileShapeWrapper、make_env）
from src.rl.train_ppo import TASK_REGISTRY, make_env
from src.env.demo import render_tactile_heatmap, _format_info_line
from src.env.base_env import RobotConfig


# ====================== 辅助函数 ======================

def unwrap_env(env):
    """剥开 Monitor / TactileShapeWrapper 等包装器，拿到最底层的任务 env。"""
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env
    return raw


def show_tactile(obs: dict, win_w: int = 1000, win_h: int = 480):
    """将三层触觉热图渲染到 OpenCV 窗口（调用 demo.py 的通用实现）."""
    heatmap = render_tactile_heatmap(obs)
    cv2.namedWindow("Tactile (Top/Mid/Bot)", cv2.WINDOW_NORMAL)
    cv2.imshow("Tactile (Top/Mid/Bot)", heatmap)
    cv2.resizeWindow("Tactile (Top/Mid/Bot)", win_w, win_h)
    cv2.waitKey(1)


def show_camera(obs: dict, info: dict, info_display: dict):
    """将相机图像叠加任务状态文字后显示。"""
    if "camera_rgb" not in obs:
        return
    cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
    info_str = _format_info_line(info, info_display)
    cv2.putText(cam_bgr, info_str, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.imshow("Camera", cam_bgr)
    cv2.resizeWindow("Camera", 640, 480)
    cv2.waitKey(1)


# ====================== 单回合评估 ======================

def run_episode(
    model: PPO,
    env,
    raw_env,
    info_display: dict,
    ep_idx: int,
    n_episodes: int,
    render: bool,
    show_tactile_win: bool,
    show_camera_win: bool,
):
    """执行一个完整的评估回合，返回 (ep_reward, ep_steps, success)。"""
    obs, info = env.reset()
    terminated = truncated = False
    ep_reward = 0.0
    ep_steps = 0

    print(f"\n--- Episode {ep_idx + 1}/{n_episodes} ---")

    def _step():
        nonlocal obs, terminated, truncated, ep_reward, ep_steps, info
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        ep_steps += 1
        if show_tactile_win:
            show_tactile(obs)
        if show_camera_win:
            show_camera(obs, info, info_display)
        return terminated, truncated

    if render:
        with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
            while viewer.is_running() and not (terminated or truncated):
                terminated, truncated = _step()
                viewer.sync()
    else:
        while not (terminated or truncated):
            terminated, truncated = _step()

    success = terminated and not truncated
    status = "✓ 成功" if success else "✗ 超时"
    info_line = _format_info_line(info, info_display)
    print(f"  {status} | 奖励: {ep_reward:.2f} | 步数: {ep_steps} | {info_line}")

    return ep_reward, ep_steps, success


# ====================== 主评估函数 ======================

def evaluate(args: argparse.Namespace) -> None:
    task_name = args.task
    reg = TASK_REGISTRY[task_name]
    info_display = reg.get("info_display", {"phase": "阶段"})
    rec = reg["recommended_hp"]

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print("=" * 65)
    print(f"  [Eval] 任务: {reg['display_name']}")
    print(f"  模型: {model_path}")
    print(f"  回合数: {args.episodes}")
    print(f"  渲染: {'是' if not args.no_render else '否'}")
    print("=" * 65)

    # ---------- 构建评估环境（与训练时完全一致） ----------
    robot_cfg = RobotConfig(
        action_mode=args.action_mode,
        controller_type=args.controller,
        max_episode_steps=args.max_episode_steps or rec["max_episode_steps"],
        action_scale=args.action_scale,
        action_scale_rot=args.action_scale_rot,
        control_freq=args.control_freq,
        sim_freq=args.sim_freq,
        tactile_backend=args.tactile_backend,
    )

    env = make_env(
        task_name=task_name,
        robot_cfg=robot_cfg,
        task_cfg_kwargs={},
        rank=0,
        seed=args.seed,
    )()

    raw_env = unwrap_env(env)

    # ---------- 加载模型 ----------
    model = PPO.load(str(model_path), env=env, device=args.device)
    print(f"\n✓ 已加载模型: {model_path}")
    print(f"  训练步数: {model.num_timesteps:,}")

    # ---------- 评估循环 ----------
    all_rewards, all_steps, successes = [], [], 0
    render = not args.no_render
    show_tac = render and not args.no_tactile
    show_cam = render and not args.no_camera

    for ep in range(args.episodes):
        reward, steps, success = run_episode(
            model=model,
            env=env,
            raw_env=raw_env,
            info_display=info_display,
            ep_idx=ep,
            n_episodes=args.episodes,
            render=render,
            show_tactile_win=show_tac,
            show_camera_win=show_cam,
        )
        all_rewards.append(reward)
        all_steps.append(steps)
        if success:
            successes += 1

    # ---------- 汇总统计 ----------
    print(f"\n{'='*65}")
    print(f"  任务: {reg['display_name']}  |  模型: {model_path.name}")
    print(f"  回合数: {args.episodes}")
    print(f"  成功率:   {successes}/{args.episodes} = {successes/args.episodes*100:.1f}%")
    print(f"  平均奖励: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  平均步数: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
    print(f"  最高奖励: {max(all_rewards):.2f}  最低奖励: {min(all_rewards):.2f}")
    print(f"{'='*65}")

    cv2.destroyAllWindows()
    env.close()


# ====================== 参数解析 ======================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通用 PPO 评估脚本（支持5个灵巧手任务）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 必填 ----
    parser.add_argument(
        "--model", type=str, required=True,
        help="模型 .zip 文件路径",
    )

    # ---- 任务 ----
    parser.add_argument(
        "--task",
        choices=list(TASK_REGISTRY.keys()),
        default="pick_place",
        help="评估任务名称（必须与训练时一致）",
    )

    # ---- 评估设置 ----
    parser.add_argument("--episodes", type=int, default=5, help="评估回合数")
    parser.add_argument("--no-render",  action="store_true", help="关闭 MuJoCo 仿真窗口")
    parser.add_argument("--no-tactile", action="store_true", help="关闭触觉热图窗口")
    parser.add_argument("--no-camera",  action="store_true", help="关闭相机图像窗口")

    # ---- 机器人与控制器（需与训练时一致）----
    parser.add_argument("--action-mode",  choices=["joint", "ee"],
                        default="joint")
    parser.add_argument("--controller",   choices=["osc", "ik"], default="osc")
    parser.add_argument("--action-scale",     type=float, default=0.03)
    parser.add_argument("--action-scale-rot", type=float, default=0.06)
    parser.add_argument("--control-freq",     type=float, default=20.0)
    parser.add_argument("--sim-freq",         type=float, default=1000.0)
    parser.add_argument("--tactile-backend",  choices=["simple", "physics", "simple_avg", "physics_avg"],
                        default="simple_avg")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                        help="单回合最大步数（None → 使用任务推荐值）")

    # ---- 其他 ----
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed",   type=int, default=999)

    return parser.parse_args()


# ====================== 入口 ======================

if __name__ == "__main__":
    evaluate(parse_args())