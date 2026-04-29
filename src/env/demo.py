"""
通用任务环境演示脚本
支持所有继承自 RobotArmEnvBase 的任务环境，通过 --task 参数切换任务。

运行方式：
# 从项目根目录执行
python -m src.env.demo --task pick_place
python -m src.env.demo --task stack
python -m src.env.demo --task insert
python -m src.env.demo --task reorient
python -m src.env.demo --task push

完整参数示例：
python -m src.env.demo \\
  --task stack \\
  --mode random \\
  --episodes 5 \\
  --action-mode joint \\
  --controller osc \\
  --no-render

功能：
1. random    : 随机策略回合演示（仿真窗口 + 触觉热力图 + 相机画面 + 末端轨迹可视化）
2. verify    : 观测空间形状与数值范围验证
3. benchmark : 无渲染高速基准测试（N 回合）
"""

import sys
import time
from pathlib import Path
from typing import Optional, Type, Tuple
from dataclasses import dataclass, field
import mujoco
import numpy as np
import cv2
from src.env.base_env import RobotArmEnvBase, RobotConfig
from src.sensors.tactile_sensor import FINGER_PHALANX_ORDER


# ====================== 任务注册表 ======================
TASK_REGISTRY: dict = {
    "pick_place": {
        "module": "src.env.pick_place_env",
        "env_class": "PickPlaceEnv",
        "cfg_class": "PickPlaceConfig",
        "display_name": "Pick and Place",
        "default_cfg_kwargs": {
            "r_step_penalty": -0.005,
            "r_place_bonus": 100.0,
            "r_grasp_bonus": 10.0,
        },
        "info_display": {
            "phase": "Phase",
            "dist_obj_target": "Obj-Target Dist(m)",
            "is_grasped": "Grasped",
        },
    },
    "stack": {
        "module": "src.env.stack_env",
        "env_class": "StackEnv",
        "cfg_class": "StackConfig",
        "display_name": "Stack Blocks",
        "default_cfg_kwargs": {},
        "info_display": {
            "phase": "Phase",
            "is_grasped": "Grasped",
            "is_stacked": "Stacked",
        },
    },
    "insert": {
        "module": "src.env.insert_env",
        "env_class": "InsertEnv",
        "cfg_class": "InsertConfig",
        "display_name": "Insert Peg",
        "default_cfg_kwargs": {},
        "info_display": {
            "phase": "Phase",
            "xy_dist_to_hole": "XY Dist to Hole(m)",
            "is_grasped": "Grasped",
            "is_inserted": "Inserted",
        },
    },
    "reorient": {
        "module": "src.env.reorient_env",
        "env_class": "ReorientEnv",
        "cfg_class": "ReorientConfig",
        "display_name": "Reorient Object",
        "default_cfg_kwargs": {},
        "info_display": {
            "phase": "Phase",
            "orient_error_rad": "Orient Error(rad)",
            "is_grasped": "Grasped",
            "is_success": "Success",
        },
    },
    "push": {
        "module": "src.env.push_env",
        "env_class": "PushEnv",
        "cfg_class": "PushConfig",
        "display_name": "Push Object",
        "default_cfg_kwargs": {},
        "info_display": {
            "phase": "Phase",
            "dist_obj_target": "Obj-Target Dist(m)",
            "tactile_max": "Tactile Max(Norm)",
            "is_success": "Success",
        },
    },
}


def _load_task(task_name: str, robot_cfg: RobotConfig) -> RobotArmEnvBase:
    """动态加载任务环境，避免顶层全量导入."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"未知任务: '{task_name}'。"
            f"可用任务: {list(TASK_REGISTRY.keys())}"
        )
    reg = TASK_REGISTRY[task_name]
    import importlib
    mod = importlib.import_module(reg["module"])
    EnvClass = getattr(mod, reg["env_class"])
    CfgClass = getattr(mod, reg["cfg_class"])
    task_cfg = CfgClass(**reg["default_cfg_kwargs"])
    return EnvClass(robot_config=robot_cfg, task_config=task_cfg)


# ====================== 可视化样式配置 ======================

@dataclass
class TrajectoryVisualStyle:
    """末端执行器轨迹可视化样式配置."""
    actual_rgba: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 1.0, 0.8]))  # 青色
    target_rgba: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.4]))  # 红色半透明
    actual_size: float = 0.005   # 5mm
    target_size: float = 0.015   # 15mm


# ====================== 轨迹可视化工具类 ======================

class EETrajectoryVisualizer:
    """
    末端执行器轨迹调试几何体绘制工具.

    利用 MuJoCo 的 user_scn 接口在仿真 Viewer 中绘制自定义几何体，
    同时显示实际位置（青色小球）和目标位置（红色大球）。
    """

    def __init__(self, style: TrajectoryVisualStyle, max_history: int = 50):
        self.style = style
        self.max_history = max_history
        self.actual_pos: Optional[np.ndarray] = None
        self.target_pos: Optional[np.ndarray] = None
        self.history: list = []

    def update(self, actual_pos: np.ndarray, target_pos: Optional[np.ndarray] = None):
        self.actual_pos = actual_pos.copy()
        if target_pos is not None:
            self.target_pos = target_pos.copy()
        if self.max_history > 0:
            self.history.append(actual_pos.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def draw(self, viewer) -> None:
        if self.actual_pos is None:
            return

        max_geoms = 1000
        safety_margin = 50

        # 1. 历史轨迹（衰减小点）
        if self.max_history > 0 and len(self.history) > 1:
            for i, hist_pos in enumerate(self.history[:-1]):
                if viewer.user_scn.ngeom >= max_geoms - safety_margin:
                    break
                alpha = 0.1 + 0.3 * (i / len(self.history))
                size = self.style.actual_size * (0.5 + 0.5 * (i / len(self.history)))
                rgba = self.style.actual_rgba.copy()
                rgba[3] = alpha
                geom_id = viewer.user_scn.ngeom
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geom_id],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[size, 0, 0],
                    pos=hist_pos,
                    mat=np.eye(3).flatten(),
                    rgba=rgba,
                )
                viewer.user_scn.ngeom += 1

        # 2. 当前实际位置（青色实心球）
        if viewer.user_scn.ngeom < max_geoms - safety_margin:
            geom_id = viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_id],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.style.actual_size, 0, 0],
                pos=self.actual_pos,
                mat=np.eye(3).flatten(),
                rgba=self.style.actual_rgba,
            )
            viewer.user_scn.ngeom += 1

        # 3. 目标位置（红色大球）
        if self.target_pos is not None and viewer.user_scn.ngeom < max_geoms - safety_margin:
            geom_id = viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_id],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.style.target_size, 0, 0],
                pos=self.target_pos,
                mat=np.eye(3).flatten(),
                rgba=self.style.target_rgba,
            )
            viewer.user_scn.ngeom += 1

    def reset(self):
        self.history.clear()
        self.actual_pos = None
        self.target_pos = None


# ====================== 通用可视化工具 ======================

def render_tactile_heatmap(obs: dict, sub_h: int = 160, sub_w: int = 200) -> np.ndarray:
    """
    将扁平化触觉图像渲染为热力图网格。
    布局：行=指节层（top/middle/bottom），列=手指（5根）
    返回 shape: (3*sub_h, 5*sub_w, 3)
    """
    finger_keys = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
    level_order = ["top", "middle", "bottom"]
    level_to_key = {
        "top":    "tactile_top",
        "middle": "tactile_middle",
        "bottom": "tactile_bottom",
    }
    level_to_phalanx_idx = {"top": 2, "middle": 1, "bottom": 0}
    grid_rows = []
    for level in level_order:
        tac_key = level_to_key[level]
        if tac_key not in obs:
            continue
        imgs = obs[tac_key]  # (5, H, W) 或 (5, H, W, 1)
        if imgs.ndim == 4:
            imgs = imgs[..., 0]
        row_frames = []
        for finger_idx, finger in enumerate(finger_keys):
            img = imgs[finger_idx]
            enhanced = np.clip(img.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
            resized  = cv2.resize(enhanced, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
            heatmap  = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
            phalanx_name = FINGER_PHALANX_ORDER[finger][level_to_phalanx_idx[level]]
            parts = phalanx_name.split('_')
            if parts[0] == "thumb":
                short_name = f"T_{parts[1][:3].capitalize()}"
            else:
                short_name = f"F{parts[1]}_{parts[2][:3].capitalize()}"
            cv2.rectangle(heatmap, (0, 0), (sub_w, 22), (0, 0, 0), -1)
            cv2.putText(heatmap, short_name, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            row_frames.append(heatmap)
        grid_rows.append(np.hstack(row_frames))
    if not grid_rows:
        return np.zeros((sub_h * 3, sub_w * 5, 3), dtype=np.uint8)
    return np.vstack(grid_rows)


def _format_info_line(info: dict, display_keys: dict) -> str:
    """将 info 字典的指定字段格式化为单行字符串."""
    parts = []
    for k, label in display_keys.items():
        v = info.get(k, "N/A")
        if isinstance(v, float):
            parts.append(f"{label}={v:.3f}")
        elif isinstance(v, bool):
            parts.append(f"{label}={'✓' if v else '✗'}")
        else:
            parts.append(f"{label}={v}")
    return " | ".join(parts)


def _get_ee_position(env: RobotArmEnvBase) -> Optional[np.ndarray]:
    """获取末端执行器位置."""
    return env.get_ee_pose()[0]


# ====================== 演示模式1：随机策略 ======================

def demo_random_policy(
    task_name: str = "pick_place",
    n_episodes: int = 3,
    render: bool = True,
    action_mode: str = "joint",
    controller_type: str = "osc",
    show_ee_traj: bool = True,
):
    """随机策略演示：仿真窗口 + 触觉热力图 + 任务状态信息 + 末端轨迹可视化."""
    reg = TASK_REGISTRY[task_name]
    print("=" * 65)
    print(f" [Demo] 随机策略 | 任务={reg['display_name']}")
    print(f" action_mode={action_mode}, controller={controller_type}")
    print(f" 末端轨迹可视化: {'开启' if show_ee_traj else '关闭'}")
    print("=" * 65)

    robot_cfg = RobotConfig(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=200,
        action_scale=0.03,
        action_scale_rot=0.06,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)
    info_display = reg["info_display"]

    if render:
        obs, info = env.reset(seed=42)
        print(f"\n[初始化] obs keys: {list(obs.keys())}")
        for k, v in obs.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        print(f"  action_dim: {env.action_space.shape[0]}")

        traj_vis = None
        if show_ee_traj:
            traj_vis = EETrajectoryVisualizer(TrajectoryVisualStyle(), max_history=30)

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            episode   = 0
            step      = 0
            ep_reward = 0.0

            while viewer.is_running() and episode < n_episodes:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step      += 1

                # 轨迹可视化
                if show_ee_traj and traj_vis is not None:
                    viewer.user_scn.ngeom = 0
                    actual_pos = _get_ee_position(env)
                    if actual_pos is not None:
                        traj_vis.update(actual_pos)
                        traj_vis.draw(viewer)

                # 触觉热力图
                heatmap = render_tactile_heatmap(obs)
                cv2.namedWindow("Tactile (Top/Mid/Bot)", cv2.WINDOW_NORMAL)
                cv2.imshow("Tactile (Top/Mid/Bot)", heatmap)
                cv2.resizeWindow("Tactile (Top/Mid/Bot)", 1000, 480)

                # 相机画面
                cam_bgr  = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
                info_str = _format_info_line(info, info_display)
                cv2.putText(cam_bgr, info_str, (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
                cv2.imshow("Camera", cam_bgr)
                cv2.resizeWindow("Camera", 640, 480)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                viewer.sync()

                if terminated or truncated:
                    status    = "✓ 成功" if terminated else "✗ 超时"
                    info_line = _format_info_line(info, info_display)
                    print(
                        f"[Episode {episode+1}] {status} | "
                        f"steps={step}, reward={ep_reward:.2f} | {info_line}"
                    )
                    episode  += 1
                    step      = 0
                    ep_reward = 0.0
                    if traj_vis is not None:
                        traj_vis.reset()
                    if episode < n_episodes:
                        obs, info = env.reset()

        cv2.destroyAllWindows()
        env.close()

    else:
        # 无渲染模式
        total_rewards, total_steps, successes = [], [], 0
        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            ep_reward = 0.0
            ep_steps  = 0
            done      = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps  += 1
                done = terminated or truncated
            total_rewards.append(ep_reward)
            total_steps.append(ep_steps)
            if terminated:
                successes += 1
            info_line = _format_info_line(info, info_display)
            print(
                f"  Ep {ep+1:3d}: reward={ep_reward:7.2f}, "
                f"steps={ep_steps:4d}, {'SUCCESS' if terminated else 'timeout'} | {info_line}"
            )
        print(f"\n  平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"  平均步数: {np.mean(total_steps):.1f}")
        print(f"  成功率:   {successes / n_episodes * 100:.1f}%")
        env.close()


# ====================== 演示模式2：观测空间验证 ======================

def demo_verify_observation_space(task_name: str = "pick_place"):
    """验证所有观测分量的形状与数值范围."""
    reg = TASK_REGISTRY[task_name]
    print("=" * 65)
    print(f" [Demo] 观测空间验证 | 任务={reg['display_name']}")
    print("=" * 65)

    robot_cfg = RobotConfig(
        action_mode="joint",
        controller_type="osc",
        max_episode_steps=100,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)
    obs, info = env.reset(seed=0)

    print("\n--- 观测空间结构 ---")
    for key, val in obs.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
              f"min={val.min():.2f}, max={val.max():.2f}")

    print("\n--- 动作空间 ---")
    print(f"  shape={env.action_space.shape}, "
          f"low={env.action_space.low[0]:.1f}, high={env.action_space.high[0]:.1f}")

    print("\n--- 初始任务状态 ---")
    for k, label in reg["info_display"].items():
        print(f"  {label}: {info.get(k, 'N/A')}")

    cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Camera RGB", cv2.WINDOW_NORMAL)
    cv2.imshow("Camera RGB", cam_bgr)
    cv2.resizeWindow("Camera RGB", 640, 480)

    heatmap = render_tactile_heatmap(obs)
    cv2.namedWindow("Tactile Heatmap", cv2.WINDOW_NORMAL)
    cv2.imshow("Tactile Heatmap", heatmap)
    cv2.resizeWindow("Tactile Heatmap", 1000, 480)

    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    env.close()


# ====================== 演示模式3：基准测试 ======================

def demo_benchmark(
    task_name: str = "pick_place",
    n_episodes: int = 100,
    action_mode: str = "joint",
    controller_type: str = "osc",
):
    """无渲染高速基准测试."""
    reg = TASK_REGISTRY[task_name]
    print(f"[Benchmark] 任务={reg['display_name']}, n_episodes={n_episodes}")

    robot_cfg = RobotConfig(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=200,
        action_scale=0.03,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)

    t0          = time.time()
    total_steps = 0
    successes   = 0

    for ep in range(n_episodes):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            total_steps += 1
            done = terminated or truncated
            if terminated:
                successes += 1

    elapsed = time.time() - t0
    env.close()

    print(f"  总步数:  {total_steps} | 总时间: {elapsed:.1f}s")
    print(f"  步频:    {total_steps / elapsed:.0f} steps/s")
    print(f"  回合频:  {n_episodes / elapsed:.1f} eps/s")
    print(f"  成功率:  {successes / n_episodes * 100:.1f}%")

# ====================== 入口 ======================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="通用任务环境演示（支持5个灵巧手任务）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_REGISTRY.keys()),
        default="pick_place",
        help=(
            "要演示的任务名称：\n"
            + "\n".join(f"  {k}: {v['display_name']}" for k, v in TASK_REGISTRY.items())
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["random", "verify", "benchmark"],
        default="random",
        help=(
            "演示模式：\n"
            "  random    随机策略 + 可视化\n"
            "  verify    观测空间验证\n"
            "  benchmark 无渲染高速基准测试\n"
        ),
    )
    parser.add_argument("--no-render",  action="store_true", help="禁用可视化（random 模式）")
    parser.add_argument("--no-traj",    action="store_true", help="禁用末端执行器轨迹可视化")
    parser.add_argument("--episodes",   type=int, default=3, help="演示回合数")
    parser.add_argument(
        "--action-mode", 
        choices=["joint", "ee"],
        default="joint",
    )
    parser.add_argument(
        "--controller",
        choices=["osc", "ik"],
        default="osc",
    )
    args = parser.parse_args()

    render    = not args.no_render
    show_traj = not args.no_traj

    print(f"\n{'='*65}")
    print(f"  任务:       {TASK_REGISTRY[args.task]['display_name']}")
    print(f"  模式:       {args.mode}")
    print(f"  渲染:       {'是' if render else '否'}")
    print(f"  轨迹可视化: {'是' if show_traj else '否'}")
    print(f"{'='*65}\n")

    if args.mode == "random":
        demo_random_policy(
            task_name=args.task,
            n_episodes=args.episodes,
            render=render,
            action_mode=args.action_mode,
            controller_type=args.controller,
            show_ee_traj=show_traj,
        )
    elif args.mode == "verify":
        demo_verify_observation_space(task_name=args.task)
    elif args.mode == "benchmark":
        demo_benchmark(
            task_name=args.task,
            n_episodes=args.episodes if args.episodes != 3 else 100,
            action_mode=args.action_mode,
            controller_type=args.controller,
        )