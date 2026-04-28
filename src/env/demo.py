"""
通用任务环境演示脚本（视觉-触觉-本体感觉版本）
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
  --action-mode osc_pose \\
  --controller osc \\
  --no-render

功能：
1. random    : 随机策略回合演示（仿真窗口 + 触觉热力图 + 相机画面 + 末端轨迹可视化）
2. verify    : 观测空间形状与数值范围验证
3. benchmark : 无渲染高速基准测试（N 回合）
4. sinusoid  : 灵巧手正弦运动演示（+ 末端轨迹可视化）
"""

import sys
import time
from pathlib import Path
from typing import Optional, Type, Tuple
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
import cv2
from src.env.base_env import RobotArmEnvBase, RobotConfig


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
    
    设计策略：
        - 缓冲区安全：检查 ngeom 防止溢出
        - 双轨迹显示：同时绘制实际位置和历史轨迹点
        - 自动衰减：历史轨迹点透明度随时间衰减（可选）
    """
    
    def __init__(self, style: TrajectoryVisualStyle, max_history: int = 50):
        """
        初始化可视化工具.
        
        Args:
            style: 可视化样式配置
            max_history: 历史轨迹点最大保留数量（0=不保留历史）
        """
        self.style = style
        self.max_history = max_history
        self.actual_pos: Optional[np.ndarray] = None
        self.target_pos: Optional[np.ndarray] = None
        self.history: list = []  # 历史实际位置列表
    
    def update(self, actual_pos: np.ndarray, target_pos: Optional[np.ndarray] = None):
        """
        更新当前末端位置.
        
        Args:
            actual_pos: 实际末端位置 (3,)
            target_pos: 目标末端位置 (3,)，可选
        """
        self.actual_pos = actual_pos.copy()
        if target_pos is not None:
            self.target_pos = target_pos.copy()
        
        # 记录历史轨迹
        if self.max_history > 0:
            self.history.append(actual_pos.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
    
    def draw(self, viewer: mujoco.viewer) -> None:
        """
        将轨迹渲染到 Viewer 场景中.
        
        注意：必须在每帧循环开始时重置 viewer.user_scn.ngeom，
              否则几何体会累积导致画面混乱。
        
        Args:
            viewer: MuJoCo Viewer 实例
        """
        if self.actual_pos is None:
            return
        
        # 安全检查：防止超出几何体缓冲区上限
        max_geoms = 1000
        safety_margin = 50
        
        # 1. 绘制历史轨迹（衰减的小点）
        if self.max_history > 0 and len(self.history) > 1:
            for i, hist_pos in enumerate(self.history[:-1]):  # 排除当前点
                if viewer.user_scn.ngeom >= max_geoms - safety_margin:
                    break
                    
                # 透明度随历史衰减
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
                    rgba=rgba
                )
                viewer.user_scn.ngeom += 1
        
        # 2. 绘制当前实际位置（青色实心球）
        if viewer.user_scn.ngeom < max_geoms - safety_margin:
            geom_id = viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_id],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.style.actual_size, 0, 0],
                pos=self.actual_pos,
                mat=np.eye(3).flatten(),
                rgba=self.style.actual_rgba
            )
            viewer.user_scn.ngeom += 1
        
        # 3. 绘制目标位置（红色大球）
        if self.target_pos is not None and viewer.user_scn.ngeom < max_geoms - safety_margin:
            geom_id = viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_id],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.style.target_size, 0, 0],
                pos=self.target_pos,
                mat=np.eye(3).flatten(),
                rgba=self.style.target_rgba
            )
            viewer.user_scn.ngeom += 1
    
    def reset(self):
        """重置历史轨迹."""
        self.history.clear()
        self.actual_pos = None
        self.target_pos = None


# ====================== 通用可视化工具 ======================

def render_tactile_heatmap(obs: dict, sub_h: int = 160, sub_w: int = 200) -> np.ndarray:
    """
    将扁平化触觉图像渲染为热力图网格。
    支持任意包含 tactile_bottom / tactile_middle / tactile_top 键的观测字典。
    布局：行=指节层（top/middle/bottom），列=手指（5根）
    返回 shape: (3*sub_h, 5*sub_w, 3)
    """
    from src.sensors.tactile_sensor import FINGER_PHALANX_ORDER
    finger_keys = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
    level_order = ["top", "middle", "bottom"]
    level_to_key = {
        "top": "tactile_top",
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
            img = imgs[finger_idx]  # (H, W)
            enhanced = np.clip(img.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
            resized = cv2.resize(enhanced, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
            phalanx_name = FINGER_PHALANX_ORDER[finger][level_to_phalanx_idx[level]]
            parts = phalanx_name.split('_')
            if parts[0] == "thumb":
                short_name = f"T_{parts[1][:3].capitalize()}"
            else:
                short_name = f"F{parts[1]}_{parts[2][:3].capitalize()}"
            cv2.rectangle(heatmap, (0, 0), (sub_w, 22), (0, 0, 0), -1)
            cv2.putText(heatmap, short_name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
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
    """
    获取末端执行器位置.
    
    尝试从环境获取末端执行器位置，支持多种环境实现。
    
    Returns:
        3D位置向量，如果无法获取则返回None
    """
    # 尝试从环境直接获取
    if hasattr(env, 'ee_site_id') and hasattr(env, 'data'):
        return env.data.site_xpos[env.ee_site_id].copy()
    
    # 尝试通过观测获取（如果环境在obs中包含末端位置）
    # 这是一个fallback，实际取决于你的RobotArmEnvBase实现
    return None


# ====================== 演示模式1：随机策略（集成轨迹可视化） ======================

def demo_random_policy(
    task_name: str = "pick_place",
    n_episodes: int = 3,
    render: bool = True,
    action_mode: str = "osc_pose",
    controller_type: str = "osc",
    show_ee_traj: bool = True,
):
    """
    随机策略演示：仿真窗口 + 触觉热力图 + 任务状态信息 + 末端轨迹可视化.
    
    Args:
        show_ee_traj: 是否显示末端执行器轨迹小球
    """
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
            print(f" {k}: shape={v.shape}, dtype={v.dtype}")
        print(f" action_dim: {env.action_space.shape[0]}")

        # 初始化轨迹可视化工具
        traj_vis = None
        if show_ee_traj:
            style = TrajectoryVisualStyle()
            traj_vis = EETrajectoryVisualizer(style, max_history=30)

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            episode = 0
            step = 0
            ep_reward = 0.0
            
            while viewer.is_running() and episode < n_episodes:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step += 1

                # --- 轨迹可视化 ---
                if show_ee_traj and traj_vis is not None:
                    # 重置几何体计数（必须在每帧开始时）
                    viewer.user_scn.ngeom = 0
                    
                    # 获取实际末端位置
                    actual_pos = _get_ee_position(env)
                    if actual_pos is not None:
                        # 对于随机策略，没有明确的目标位置，只显示实际轨迹
                        traj_vis.update(actual_pos, target_pos=None)
                        traj_vis.draw(viewer)

                # --- OpenCV 可视化 ---
                heatmap = render_tactile_heatmap(obs)
                cv2.namedWindow("Tactile (Top/Mid/Bot)", cv2.WINDOW_NORMAL)
                cv2.imshow("Tactile (Top/Mid/Bot)", heatmap)
                cv2.resizeWindow("Tactile (Top/Mid/Bot)", 1000, 480)

                cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
                cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
                info_str = _format_info_line(info, info_display)
                cv2.putText(cam_bgr, info_str, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("Camera", cam_bgr)
                cv2.resizeWindow("Camera", 640, 480)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                viewer.sync()

                if terminated or truncated:
                    status = "✓ 成功" if terminated else "✗ 超时"
                    info_line = _format_info_line(info, info_display)
                    print(
                        f"[Episode {episode+1}] {status} | "
                        f"steps={step}, reward={ep_reward:.2f} | {info_line}"
                    )
                    episode += 1
                    step = 0
                    ep_reward = 0.0
                    
                    # 重置轨迹历史
                    if traj_vis is not None:
                        traj_vis.reset()
                    
                    if episode < n_episodes:
                        obs, info = env.reset()
    else:
        # 无渲染模式
        total_rewards, total_steps, successes = [], [], 0
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
            total_steps.append(ep_steps)
            if terminated:
                successes += 1
            info_line = _format_info_line(info, info_display)
            print(
                f" Ep {ep+1:3d}: reward={ep_reward:7.2f}, "
                f"steps={ep_steps:4d}, {'SUCCESS' if terminated else 'timeout'} | {info_line}"
            )
        print(f"\n 平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f" 平均步数: {np.mean(total_steps):.1f}")
        print(f" 成功率: {successes/n_episodes*100:.1f}%")
        cv2.destroyAllWindows()
        env.close()


# ====================== 演示模式3：观测空间验证 ======================
def demo_verify_observation_space(task_name: str = "pick_place"):
    """验证所有观测分量的形状与数值范围."""
    reg = TASK_REGISTRY[task_name]
    print("=" * 65)
    print(f" [Demo] 观测空间验证 | 任务={reg['display_name']}")
    print("=" * 65)

    robot_cfg = RobotConfig(
        action_mode="osc_pose",
        controller_type="osc",
        max_episode_steps=100,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)
    obs, info = env.reset(seed=0)

    print("\n--- 观测空间结构 ---")
    for key, val in obs.items():
        print(f" {key}: shape={val.shape}, dtype={val.dtype}, "
              f"min={val.min():.2f}, max={val.max():.2f}")

    print("\n--- 动作空间 ---")
    print(f" shape={env.action_space.shape}, "
          f"low={env.action_space.low[0]:.1f}, high={env.action_space.high[0]:.1f}")

    print("\n--- 初始任务状态 ---")
    for k, label in reg["info_display"].items():
        print(f" {label}: {info.get(k, 'N/A')}")

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


# ====================== 演示模式4：基准测试 ======================
def demo_benchmark(
    task_name: str = "pick_place",
    n_episodes: int = 100,
    action_mode: str = "osc_pose",
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

    t0 = time.time()
    total_steps = 0
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            total_steps += 1
            done = terminated or truncated
            if terminated:
                successes += 1

    elapsed = time.time() - t0
    env.close()

    print(f" 总步数: {total_steps} | 总时间: {elapsed:.1f}s")
    print(f" 步频: {total_steps/elapsed:.0f} steps/s")
    print(f" 回合频: {n_episodes/elapsed:.1f} eps/s")
    print(f" 成功率: {successes/n_episodes*100:.1f}%")


# ====================== 演示模式5：灵巧手正弦运动（集成轨迹可视化） ======================

def demo_sinusoid(
    task_name: str = "pick_place",
    freq: float = 0.5,
    amplitude: float = 0.5,
    n_episodes: int = 1,
    render: bool = True,
    show_ee_traj: bool = True,
):
    """
    灵巧手关节正弦运动演示（集成末端轨迹可视化）.
    
    - 机械臂本体 (前6轴) 保持静止
    - 灵巧手关节 (最后6/7轴) 执行正弦摆动
    - 显示末端执行器实际位置轨迹（青色）和历史轨迹
    """
    reg = TASK_REGISTRY[task_name]
    print("=" * 65)
    print(f" [Demo] 灵巧手正弦运动 | 任务={reg['display_name']}")
    print(f" 频率={freq}Hz, 幅度={amplitude}rad")
    print(f" 末端轨迹可视化: {'开启' if show_ee_traj else '关闭'}")
    print("=" * 65)

    robot_cfg = RobotConfig(
        action_mode="joint_pd",
        controller_type="osc",
        max_episode_steps=1000,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    
    env = _load_task(task_name, robot_cfg)
    action_dim = env.action_space.shape[0]
    
    ARM_DOF = 6
    HAND_DOF = action_dim - ARM_DOF

    # 初始化轨迹可视化
    traj_vis = None
    if show_ee_traj:
        style = TrajectoryVisualStyle(
            actual_rgba=np.array([0.0, 1.0, 1.0, 0.9]),  # 更亮的青色
            target_rgba=np.array([1.0, 0.0, 0.0, 0.4]),
            actual_size=0.008,
            target_size=0.012
        )
        traj_vis = EETrajectoryVisualizer(style, max_history=50)

    if render:
        obs, info = env.reset(seed=42)
        
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            start_time = time.time()
            
            while viewer.is_running():
                current_elapsed = time.time() - start_time
                
                # 1. 动作生成
                arm_action = np.zeros(ARM_DOF)
                phase_offset = np.linspace(0, np.pi, HAND_DOF)
                hand_action = amplitude * np.sin(2 * np.pi * freq * current_elapsed + phase_offset)
                action = np.concatenate([arm_action, hand_action])
                
                # 2. 环境步进
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 3. 轨迹可视化
                if show_ee_traj and traj_vis is not None:
                    viewer.user_scn.ngeom = 0
                    
                    actual_pos = _get_ee_position(env)
                    if actual_pos is not None:
                        traj_vis.update(actual_pos, target_pos=None)
                        traj_vis.draw(viewer)
                
                # 4. 渲染同步
                viewer.sync()

                # 5. OpenCV 辅助可视化
                heatmap = render_tactile_heatmap(obs)
                cv2.imshow("Tactile (Top/Mid/Bot)", heatmap)
                
                if 'camera_rgb' in obs:
                    cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
                    cv2.imshow("Camera", cam_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if terminated or truncated:
                    obs, info = env.reset()
                    if traj_vis is not None:
                        traj_vis.reset()
                    start_time = time.time()

    cv2.destroyAllWindows()
    env.close()


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
        help="要演示的任务名称：" + "\n" +
             "\n".join(f" {k}: {v['display_name']}" for k, v in TASK_REGISTRY.items()),
    )
    parser.add_argument(
        "--mode",
        choices=["random", "verify", "benchmark", "sinusoid"],
        default="random",
        help="演示模式",
    )
    parser.add_argument("--no-render", action="store_true", help="禁用可视化（加速运行）")
    parser.add_argument("--episodes", type=int, default=3, help="演示回合数")
    parser.add_argument("--freq", type=float, default=0.5, help="正弦波频率 (仅 sinusoid 模式有效)")
    parser.add_argument("--amp", type=float, default=0.5, help="正弦波幅度 (仅 sinusoid 模式有效)")
    parser.add_argument(
        "--no-traj", action="store_true",
        help="禁用末端执行器轨迹可视化小球"
    )
    parser.add_argument(
        "--action-mode",
        choices=["osc_pose", "osc_pos", "joint_pd"],
        default="osc_pose",
    )
    parser.add_argument(
        "--controller",
        choices=["osc", "ik"],
        default="osc",
    )
    args = parser.parse_args()

    render = not args.no_render
    show_traj = not args.no_traj
    
    print(f"\n{'='*65}")
    print(f" 任务: {TASK_REGISTRY[args.task]['display_name']}")
    print(f" 模式: {args.mode}")
    print(f" 渲染: {'是' if render else '否'}")
    print(f" 轨迹可视化: {'是' if show_traj else '否'}")
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
    elif args.mode == "sinusoid":
        demo_sinusoid(
            task_name=args.task,
            freq=args.freq,
            amplitude=args.amp,
            render=render,
            show_ee_traj=show_traj,
        )