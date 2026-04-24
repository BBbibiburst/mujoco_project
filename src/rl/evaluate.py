"""
PPO 模型评估与可视化脚本
适配：
- MultiInputPolicy
- Dict Observation
- TactileShapeWrapper
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mujoco
from stable_baselines3 import PPO

from src.rl.train_ppo import make_env
from src.env.demo import render_tactile_heatmap


def unwrap_env(env):
    """解包 Monitor 等包装器，获取最底层环境。"""
    raw_env = env
    while hasattr(raw_env, "env"):
        raw_env = raw_env.env
    return raw_env


def show_tactile(obs):
    """显示三个触觉传感器热图。"""
    tactile_keys = [
        "tactile_bottom",
        "tactile_middle",
        "tactile_top",
    ]

    for key in tactile_keys:
        if key not in obs:
            continue

        tactile = obs[key]

        # (5, H, W, 1) -> (5, H, W)
        if tactile.ndim == 4 and tactile.shape[-1] == 1:
            tactile = tactile.squeeze(-1)

        heatmap = render_tactile_heatmap(tactile)
        cv2.imshow(key, heatmap)

    cv2.waitKey(1)


def evaluate(
    model_path: str,
    n_episodes: int = 5,
    render: bool = True,
):
    """加载训练好的 PPO 模型并评估。"""

    # 创建环境
    env = make_env(rank=0, seed=999)()

    # 加载模型
    model = PPO.load(model_path, env=env)
    print(f"✓ 已加载模型: {model_path}")

    raw_env = unwrap_env(env)

    for ep in range(n_episodes):
        obs, info = env.reset()

        terminated = False
        truncated = False

        ep_reward = 0.0
        ep_steps = 0

        print(f"\n--- Episode {ep + 1}/{n_episodes} ---")

        if render:
            with mujoco.viewer.launch_passive(
                raw_env.model,
                raw_env.data,
            ) as viewer:

                while (
                    viewer.is_running()
                    and not (terminated or truncated)
                ):
                    action, _ = model.predict(
                        obs,
                        deterministic=True,
                    )

                    obs, reward, terminated, truncated, info = env.step(action)

                    ep_reward += float(reward)
                    ep_steps += 1

                    show_tactile(obs)
                    viewer.sync()
        else:
            while not (terminated or truncated):
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = env.step(action)

                ep_reward += float(reward)
                ep_steps += 1

        success = terminated and not truncated
        status = "✓ 成功" if success else "✗ 超时"

        print(
            f"  {status} | "
            f"奖励: {ep_reward:.2f} | "
            f"步数: {ep_steps} | "
            f"阶段: {info.get('phase', 'N/A')}"
        )

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="评估 PPO PickPlace 模型"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径 (.zip)",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="评估回合数",
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="关闭 Mujoco 可视化",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render,
    )