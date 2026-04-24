"""
抓取放置环境运行演示（视觉-触觉-本体感觉版本）.

运行方式：
    # 从项目根目录执行
    python -m src.env.demo

功能：
    1. 随机策略回合演示（可视化仿真窗口 + 触觉热力图）
    2. 脚本化策略演示（视觉验证）
    3. 观测空间验证与调试
"""

import sys
import time
from pathlib import Path

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
import cv2

from src.env.pick_place_env import PickPlaceEnv, RobotConfig, PickPlaceConfig


# ====================== 可视化工具 ======================

def render_tactile_heatmap(tactile_dict: dict, sub_h: int = 120, sub_w: int = 160) -> np.ndarray:
    """
    将分组触觉图像渲染为热力图网格（与 grasp_task_env.py 一致）.
    
    Args:
        tactile_dict: {"bottom": (5,10,7), "middle": (5,8,5), "top": (5,6,5)}
        sub_h, sub_w: 每块皮肤的显示尺寸
    
    Returns:
        np.ndarray: 拼接后的热力图 (3*sub_h, 5*sub_w, 3)
    """
    from src.sensors.tactile_sensor import FINGER_PHALANX_ORDER
    
    # ✅ FIX: 显式指定手指顺序，与 pick_place_env.py 保持一致
    finger_keys = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
    phalanx_levels = ["top", "middle", "bottom"]  # 显示顺序：指尖在上，指根在下
    
    # 建立 (name -> image) 映射
    name_to_img = {}
    for level, imgs in tactile_dict.items():
        for idx, finger in enumerate(finger_keys):
            phalanx_name = FINGER_PHALANX_ORDER[finger][
                {"top": 2, "middle": 1, "bottom": 0}[level]
            ]
            name_to_img[phalanx_name] = imgs[idx]
    
    # 按网格生成热力图
    grid_rows = []
    for level in phalanx_levels:
        row_frames = []
        for finger in finger_keys:
            phalanx_name = FINGER_PHALANX_ORDER[finger][
                {"top": 2, "middle": 1, "bottom": 0}[level]
            ]
            img = name_to_img.get(phalanx_name, np.zeros((7, 10)))
            
            # 增强对比度
            enhanced = np.clip(img.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
            resized = cv2.resize(enhanced, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
            
            # 标题文字
            parts = phalanx_name.split('_')
            if parts[0] == "thumb":
                short_name = f"T_{parts[1][:3].capitalize()}"
            else:
                short_name = f"F{parts[1]}_{parts[2][:3].capitalize()}"
            
            cv2.rectangle(heatmap, (0, 0), (sub_w, 25), (0, 0, 0), -1)
            cv2.putText(heatmap, short_name, (5, 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            row_frames.append(heatmap)
        
        grid_rows.append(np.hstack(row_frames))
    
    return np.vstack(grid_rows)


# ====================== 演示模式1：随机策略 ======================

def demo_random_policy(n_episodes: int = 3, render: bool = True,
                       action_mode: str = "osc_pose", controller_type: str = "osc"):
    """运行随机策略若干回合并打印统计信息."""
    print("=" * 60)
    print(f"  [Demo] 随机策略演示 | action_mode={action_mode}, controller={controller_type}")
    print("=" * 60)

    robot_cfg = RobotConfig(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=200,
        action_scale=0.03,
        action_scale_rot=0.06,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    task_cfg = PickPlaceConfig(
        r_step_penalty=-0.005,
        r_place_bonus=100.0,
    )

    env = PickPlaceEnv(robot_cfg, task_cfg)

    if render:
        obs, info = env.reset(seed=42)
        print(f"\n[Episode 1] 已重置。")
        print(f"  obs keys: {list(obs.keys())}")
        
        # 安全访问各观测 key
        if 'camera_rgb' in obs:
            print(f"  camera_rgb shape: {obs['camera_rgb'].shape}")
        if 'tactile' in obs:
            if 'bottom' in obs['tactile']:
                print(f"  tactile bottom shape: {obs['tactile']['bottom'].shape}")
            else:
                print(f"  tactile keys: {list(obs['tactile'].keys())}")
        if 'proprioception' in obs:
            print(f"  proprioception shape: {obs['proprioception'].shape}")
        
        print(f"  action_dim: {env.action_space.shape[0]}")

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            episode = 0
            step = 0
            ep_reward = 0.0

            while viewer.is_running() and episode < n_episodes:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step += 1

                # 显示触觉热力图
                tactile_heatmap = render_tactile_heatmap(obs['tactile'])
                cv2.imshow("Tactile Heatmap (Top / Mid / Bot)", tactile_heatmap)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # 可选：显示相机图像
                cv2.imshow("Camera", cv2.cvtColor(obs['camera_rgb'], cv2.COLOR_RGB2BGR))

                viewer.sync()

                if terminated or truncated:
                    status = "✓ 成功" if terminated else "✗ 超时"
                    print(
                        f"[Episode {episode+1}] {status} | "
                        f"steps={step}, reward={ep_reward:.2f}, "
                        f"phase={info['phase']}, "
                        f"dist={info['dist_obj_target']:.3f}m"
                    )
                    episode += 1
                    step = 0
                    ep_reward = 0.0

                    if episode < n_episodes:
                        obs, info = env.reset()
    else:
        # 无渲染模式
        total_rewards = []
        total_steps_list = []
        successes = 0

        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            ep_reward = 0.0
            ep_steps = 0
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated

            total_rewards.append(ep_reward)
            total_steps_list.append(ep_steps)
            if terminated:
                successes += 1
            print(
                f"  Episode {ep+1:3d}: reward={ep_reward:7.2f}, "
                f"steps={ep_steps:4d}, {'SUCCESS' if terminated else 'timeout'}"
            )

        print(f"\n  平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"  平均步数: {np.mean(total_steps_list):.1f}")
        print(f"  成功率:   {successes/n_episodes*100:.1f}%")

    cv2.destroyAllWindows()
    env.close()


# ====================== 演示模式2：脚本化策略 ======================

def demo_scripted_policy(render: bool = True,
                         action_mode: str = "osc_pose", controller_type: str = "osc"):
    """脚本化策略演示（视觉验证）."""
    print("=" * 60)
    print(f"  [Demo] 脚本化策略演示 | action_mode={action_mode}, controller={controller_type}")
    print("=" * 60)

    robot_cfg = RobotConfig(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=500,
        action_scale=0.02,
        action_scale_rot=0.04,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    env = PickPlaceEnv(robot_cfg)
    obs, info = env.reset(seed=0)

    print(f"  物体位置: {info['obj_pos']}")
    print(f"  目标位置: {info['target_pos']}")
    print(f"  动作维度: {env.action_space.shape[0]}")

    def get_obj_pos():
        return env._get_obj_pos()

    def get_ee_pos():
        pos, _ = env.get_ee_pose()
        return pos

    # ✅ FIX: viewer 作为参数传入 move_to，避免闭包隐式依赖
    def move_to(target_pos, hand_target, viewer, tol=0.03, max_steps=100):
        """控制末端移动到目标位置."""
        for _ in range(max_steps):
            ee_pos = get_ee_pos()
            delta = target_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist < tol:
                return True

            direction = delta / (dist + 1e-6)
            arm_pos_action = np.clip(direction * 2.0, -1, 1)
            hand_action = hand_target * np.ones(6)

            if env.cfg.action_mode == "osc_pose":
                arm_rot_action = np.zeros(3)
                action = np.concatenate([arm_pos_action, arm_rot_action, hand_action])
            elif env.cfg.action_mode == "osc_pos":
                action = np.concatenate([arm_pos_action, hand_action])
            elif env.cfg.action_mode == "joint_pd":
                raise NotImplementedError("joint_pd 模式不支持脚本化策略")
            else:
                raise ValueError(f"Unknown action_mode: {env.cfg.action_mode}")

            env.step(action)
            if render and viewer is not None:
                viewer.sync()
        return False

    if render:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            _scripted_steps(env, get_obj_pos, get_ee_pos, move_to, viewer)
    else:
        class FakeViewer:
            def sync(self): pass
        viewer = FakeViewer()
        _scripted_steps(env, get_obj_pos, get_ee_pos, move_to, viewer)

    cv2.destroyAllWindows()
    env.close()


def _scripted_steps(env, get_obj_pos, get_ee_pos, move_to, viewer):
    """脚本化策略的具体执行步骤."""
    print("\n  阶段 1: 移到物体正上方...")
    obj = get_obj_pos()
    above_obj = obj + np.array([0, 0, 0.15])
    success = move_to(above_obj, hand_target=0.0, viewer=viewer, tol=0.04)
    print(f"    {'到达' if success else '超时'}")

    print("  阶段 2: 下降到物体...")
    obj = get_obj_pos()
    near_obj = obj + np.array([0, 0, 0.02])
    success = move_to(near_obj, hand_target=0.0, viewer=viewer, tol=0.03)
    print(f"    {'到达' if success else '超时'}")

    print("  阶段 3: 闭合手指 (50步)...")
    for step in range(50):
        if env.cfg.action_mode == "osc_pose":
            action = np.concatenate([np.zeros(6), np.ones(6)])
        elif env.cfg.action_mode == "osc_pos":
            action = np.concatenate([np.zeros(3), np.ones(6)])
        else:
            raise NotImplementedError("joint_pd 模式不支持脚本化策略")

        env.step(action)
        if hasattr(viewer, 'sync'):
            viewer.sync()
    print("    手指已闭合")

    print("  阶段 4: 抬起物体...")
    ee = get_ee_pos()
    lift_target = ee + np.array([0, 0, 0.15])
    success = move_to(lift_target, hand_target=1.0, viewer=viewer, tol=0.04)
    obj_final = get_obj_pos()
    print(f"    {'成功' if success else '超时'} | 物体高度: {obj_final[2]:.3f}m")

    print(f"\n  最终 Episode info: {env._get_info()}")


# ====================== 演示模式3：观测空间验证 ======================

def demo_verify_observation_space():
    """验证观测空间各组件的形状和范围."""
    print("=" * 60)
    print("  [Demo] 观测空间验证")
    print("=" * 60)

    robot_cfg = RobotConfig(
        action_mode="osc_pose",
        controller_type="osc",
        max_episode_steps=100,
        tactile_backend="simple_avg",
    )
    env = PickPlaceEnv(robot_cfg)
    obs, info = env.reset(seed=0)

    print("\n--- 观测空间结构 ---")
    for key, val in obs.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for sub_key, sub_val in val.items():
                print(f"    {sub_key}: shape={sub_val.shape}, dtype={sub_val.dtype}, "
                      f"min={sub_val.min():.1f}, max={sub_val.max():.1f}")
        else:
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
                  f"min={val.min():.1f}, max={val.max():.1f}")

    print("\n--- 触觉传感器分辨率验证 ---")
    env._verify_tactile_shapes()

    # 验证相机图像可视化
    cv2.imshow("Camera RGB", cv2.cvtColor(obs['camera_rgb'], cv2.COLOR_RGB2BGR))
    tactile_heatmap = render_tactile_heatmap(obs['tactile'])
    cv2.imshow("Tactile Heatmap", tactile_heatmap)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    env.close()


# ====================== 演示模式4：对比测试 ======================

def demo_compare_configs():
    """对比不同 action_mode 和 controller_type 的组合."""
    configs = [
        ("osc_pose", "osc", "OSC 6D位姿控制"),
        ("osc_pos", "osc", "OSC 3D位置控制"),
        ("osc_pos", "ik", "IK 3D位置控制"),
        ("joint_pd", "osc", "OSC 关节PD"),
    ]

    for action_mode, controller_type, desc in configs:
        print(f"\n{'='*60}")
        print(f"  测试配置: {desc}")
        print(f"  action_mode={action_mode}, controller={controller_type}")
        print(f"{'='*60}")

        try:
            demo_random_policy(
                n_episodes=2,
                render=False,
                action_mode=action_mode,
                controller_type=controller_type,
            )
        except Exception as e:
            print(f"  [错误] {e}")


# ====================== 入口 ======================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PickPlace 环境演示（视觉-触觉版本）")
    parser.add_argument("--mode", choices=["random", "scripted", "verify", "benchmark", "compare"],
                        default="random", help="演示模式")
    parser.add_argument("--no-render", action="store_true", help="禁用可视化")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--action-mode", choices=["osc_pose", "osc_pos", "joint_pd"],
                        default="osc_pose", help="动作空间模式")
    parser.add_argument("--controller", choices=["osc", "ik"],
                        default="osc", help="底层控制器类型")
    args = parser.parse_args()

    render = not args.no_render

    if args.mode == "random":
        demo_random_policy(
            n_episodes=args.episodes,
            render=render,
            action_mode=args.action_mode,
            controller_type=args.controller,
        )
    elif args.mode == "scripted":
        demo_scripted_policy(
            render=render,
            action_mode=args.action_mode,
            controller_type=args.controller,
        )
    elif args.mode == "verify":
        demo_verify_observation_space()
    elif args.mode == "benchmark":
        print("基准测试（无渲染，100回合随机策略）...")
        t0 = time.time()
        demo_random_policy(
            n_episodes=100,
            render=False,
            action_mode=args.action_mode,
            controller_type=args.controller,
        )
        elapsed = time.time() - t0
        print(f"耗时: {elapsed:.1f}s | 平均每回合: {elapsed/100*1000:.0f}ms")
    elif args.mode == "compare":
        demo_compare_configs()