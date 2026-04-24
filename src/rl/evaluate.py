"""
模型评估与可视化脚本
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import mujoco
from stable_baselines3 import PPO
from src.rl.train_ppo import FlattenObservationWrapper, make_env
from src.env.demo import render_tactile_heatmap


def evaluate(model_path: str, n_episodes: int = 5, render: bool = True):
    """加载训练好的模型并运行评估."""
    
    # 创建环境
    env = make_env(0, seed=999)()
    
    # 加载模型
    model = PPO.load(model_path, env=env)
    print(f"✓ 已加载模型: {model_path}")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        
        print(f"\n--- Episode {ep+1}/{n_episodes} ---")
        
        if render:
            # 解包环境获取底层 PickPlaceEnv（用于 viewer.sync）
            raw_env = env.env  # 解包 Monitor
            while hasattr(raw_env, 'env'):
                raw_env = raw_env.env
            
            with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
                while not done and viewer.is_running():
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    ep_steps += 1
                    
                    # ✅ FIX: 从 FlattenObservationWrapper 缓存获取原始 Dict 观测
                    # 避免重复调用 _get_obs() 导致的性能浪费和时序问题
                    flatten_wrapper = env.env  # Monitor -> FlattenWrapper
                    while hasattr(flatten_wrapper, 'env') and not isinstance(flatten_wrapper, FlattenObservationWrapper):
                        flatten_wrapper = flatten_wrapper.env
                    
                    # ✅ FIX: 用 is not None 判断，避免 falsy 误判
                    if isinstance(flatten_wrapper, FlattenObservationWrapper) and flatten_wrapper.last_raw_obs is not None:
                        raw_obs = flatten_wrapper.last_raw_obs
                        if 'tactile' in raw_obs:
                            heatmap = render_tactile_heatmap(raw_obs['tactile'])
                            cv2.imshow("Tactile", heatmap)
                            cv2.waitKey(1)
                    
                    viewer.sync()
        else:
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1
        
        status = "✓ 成功" if terminated else "✗ 超时"
        print(f"  {status} | 奖励: {ep_reward:.2f} | 步数: {ep_steps} | 阶段: {info.get('phase', 'N/A')}")
    
    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    
    evaluate(args.model, args.episodes, render=not args.no_render)