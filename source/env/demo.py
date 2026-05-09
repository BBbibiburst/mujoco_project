"""
通用任务环境演示脚本
支持所有继承自 RobotArmEnvBase 的任务环境，通过 --task 参数切换任务。

运行方式：
# 从项目根目录执行
python -m source.env.demo --task pick_place

完整参数示例：
python -m source.env.demo \
  --task stack \
  --mode random \
  --episodes 5 \
  --action-mode joint \
  --controller osc \
  --no-render

python -m source.env.demo \
  --task block_lifting \
  --mode keyboard

功能：
1. random    : 随机策略回合演示（仿真窗口 + 触觉热力图 + 相机画面 + 末端轨迹可视化）
2. verify    : 观测空间形状与数值范围验证
3. benchmark : 无渲染高速基准测试（N 回合）
4. keyboard  : 键盘逐关节控制（Tkinter 控制面板 + 奖励/终止/截断实时显示）
"""

import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Type, Tuple
from dataclasses import dataclass, field
import mujoco
import numpy as np
import cv2
from source.env.base_env import RobotArmEnvBase, RobotConfig
from source.sensors.tactile_sensor import FINGER_PHALANX_ORDER

# ====================== 任务注册表 ======================
TASK_REGISTRY: dict = {
    "pick_and_place": {
        "module": "source.env.pick_and_place_env",
        "env_class": "PickAndPlaceEnv",
        "cfg_class": "PickAndPlaceConfig",
        "display_name": "Pick and Place",
        "default_cfg_kwargs": {},
    },
    "block_stacking": {
        "module": "source.env.block_stacking_env",
        "env_class": "BlockStackingEnv",
        "cfg_class": "BlockStackingConfig",
        "display_name": "Block Stacking",
        "default_cfg_kwargs": {},
    },
    "block_lifting": {
        "module": "source.env.block_lifting_env",
        "env_class": "BlockLiftingEnv",
        "cfg_class": "BlockLiftingConfig",
        "display_name": "Block Lifting",
        "default_cfg_kwargs": {},
    },
    "nut_assembly": {
        "module": "source.env.nut_assembly_env",
        "env_class": "NutAssemblyEnv",
        "cfg_class": "NutAssemblyConfig",
        "display_name": "Nut Assembly",
        "default_cfg_kwargs": {},
    },
    "door_opening": {
        "module": "source.env.door_opening_env",
        "env_class": "DoorOpeningEnv",
        "cfg_class": "DoorOpeningConfig",
        "display_name": "Door Opening",
        "default_cfg_kwargs": {},
    },
}


def _load_task(task_name: str, robot_cfg: RobotConfig) -> RobotArmEnvBase:
    """动态加载任务环境，避免顶层全量导入."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"未知任务: '{task_name}'。" f"可用任务: {list(TASK_REGISTRY.keys())}"
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

    actual_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 1.0, 0.8])
    )  # 青色
    target_rgba: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.4])
    )  # 红色半透明
    actual_size: float = 0.005  # 5mm
    target_size: float = 0.015  # 15mm


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

    def update(self, actual_pos: np.ndarray, target_pos: Optional[np.ndarray] = None, target_quat: Optional[np.ndarray] = None):
        self.actual_pos = actual_pos.copy()
        if target_pos is not None:
            self.target_pos = target_pos.copy()
        if target_quat is not None:
            self.target_quat = target_quat.copy() # 保存目标四元数
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
        if (
            self.target_pos is not None
            and viewer.user_scn.ngeom < max_geoms - safety_margin
        ):
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
        # === 新增：绘制 EE 目标朝向坐标轴 ===
        # 将四元数转换为 3x3 旋转矩阵
        target_mat = np.zeros(9)
        # 假设你传入的 target_quat 存储在类中（稍后修改 update 方法）
        if hasattr(self, 'target_quat') and self.target_quat is not None:
            mujoco.mju_quat2Mat(target_mat, self.target_quat)
            
            # 绘制三色轴：R=X, G=Y, B=Z
            axis_len = 0.1  # 箭头长度 10cm
            axis_width = 0.002 # 箭头粗细
            
            for i in range(3): # 0:X, 1:Y, 2:Z
                color = np.array([0.0, 0.0, 0.0, 1.0])
                color[i] = 1.0 # 对应轴设为满色
                
                geom_id = viewer.user_scn.ngeom
                if geom_id < max_geoms:
                    # 使用 mjGEOM_ARROW 绘制箭头
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geom_id],
                        type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[axis_width, axis_width, axis_len],
                        pos=self.target_pos,
                        mat=target_mat, # 箭头的方向由矩阵决定
                        rgba=color
                    )
                    # 注意：mjGEOM_ARROW 默认沿 Z 轴方向。为了画 X/Y 轴，
                    # 我们需要对矩阵做局部变换，或者简单地使用 mjv_makeConnector 
                    # 这里为了代码简洁，直接在 target_mat 基础上偏移方向
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
            img = imgs[finger_idx]
            enhanced = np.clip(img.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
            resized = cv2.resize(
                enhanced, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST
            )
            heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
            phalanx_name = FINGER_PHALANX_ORDER[finger][level_to_phalanx_idx[level]]
            parts = phalanx_name.split("_")
            if parts[0] == "thumb":
                short_name = f"T_{parts[1][:3].capitalize()}"
            else:
                short_name = f"F{parts[1]}_{parts[2][:3].capitalize()}"
            cv2.rectangle(heatmap, (0, 0), (sub_w, 22), (0, 0, 0), -1)
            cv2.putText(
                heatmap,
                short_name,
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
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
    task_name: str = "pick_and_place",
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
            episode = 0
            step = 0

            while viewer.is_running() and episode < n_episodes:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1

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
                cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
                cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
                cv2.imshow("Camera", cam_bgr)
                cv2.resizeWindow("Camera", 640, 480)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                viewer.sync()

                if terminated or truncated:
                    status = "✓ 终止" if terminated else "✗ 超时"
                    print(
                        f"[Episode {episode+1}] {status} | "
                        f"steps={step}"
                    )
                    episode += 1
                    step = 0
                    if traj_vis is not None:
                        traj_vis.reset()
                    if episode < n_episodes:
                        obs, info = env.reset()

        cv2.destroyAllWindows()
        env.close()

    else:
        # 无渲染模式
        total_steps, successes = [], 0
        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            ep_steps = 0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_steps += 1
                done = terminated or truncated
            total_steps.append(ep_steps)
            if terminated:
                successes += 1
            print(
                f"  Ep {ep+1:3d}: "
                f"steps={ep_steps:4d}, {'TERMINATED' if terminated else 'timeout'}"
            )
        print(f"\n  平均步数: {np.mean(total_steps):.1f}")
        print(f"  终止率:   {successes / n_episodes * 100:.1f}%")
        env.close()


# ====================== 演示模式2：观测空间验证 ======================


def demo_verify_observation_space(task_name: str = "pick_and_place"):
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
        print(
            f"  {key}: shape={val.shape}, dtype={val.dtype}, "
            f"min={val.min():.2f}, max={val.max():.2f}"
        )

    print("\n--- 动作空间 ---")
    print(
        f"  shape={env.action_space.shape}, "
        f"low={env.action_space.low[0]:.1f}, high={env.action_space.high[0]:.1f}"
    )

    print("\n--- 初始任务状态 ---")

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
    task_name: str = "pick_and_place",
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
        action_scale_rot=0.06,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)

    t0 = time.time()
    total_steps = 0
    terminations = 0

    for ep in range(n_episodes):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            total_steps += 1
            done = terminated or truncated
            if terminated:
                terminations += 1

    elapsed = time.time() - t0
    env.close()

    print(f"  总步数:  {total_steps} | 总时间: {elapsed:.1f}s")
    print(f"  步频:    {total_steps / elapsed:.0f} steps/s")
    print(f"  回合频:  {n_episodes / elapsed:.1f} eps/s")
    print(f"  终止率:  {terminations / n_episodes * 100:.1f}%")


# ====================== 演示模式4：键盘逐关节控制 ======================

# -------- 控制面板常量 --------
_ARM_JOINT_NAMES = [
    "J1 (Shoulder Yaw)",
    "J2 (Shoulder Pitch)",
    "J3 (Shoulder Roll)",
    "J4 (Elbow)",
    "J5 (Forearm)",
    "J6 (Wrist Pitch)",
    "J7 (Wrist Roll)",
]

_HAND_JOINT_NAMES = [
    "F0 (Actuator 0)",
    "F1 (Actuator 1)",
    "F2 (Actuator 2)",
    "F3 (Actuator 3)",
    "F4 (Actuator 4)",
    "F5 (Actuator 5)",
]
# ee 模式下末端自由度名称（索引 0-5）
_EE_DOF_NAMES = [
    "EE X  (pos, m)",
    "EE Y  (pos, m)",
    "EE Z  (pos, m)",
    "EE Rx (rot, rad)",
    "EE Ry (rot, rad)",
    "EE Rz (rot, rad)",
]

# 按键绑定（键盘模式）
_KEY_INCREASE = "Up"
_KEY_DECREASE = "Down"
_KEY_PREV_JNT = "Left"
_KEY_NEXT_JNT = "Right"
_KEY_RESET = "r"
_KEY_OPEN_HAND = "o"
_KEY_CLOSE_HAND = "c"
_KEY_QUIT = "q"


def _quat_to_euler_deg(quat: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] → 欧拉角 ZYX (roll, pitch, yaw)，单位度."""
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(np.array([roll, pitch, yaw]))


class KeyboardControlPanel:
    """
    Tkinter 键盘控制面板（joint / ee 双模式）.

    joint 模式：显示 7 个臂关节目标值 + 6 个手部目标值（共 13 列）
    ee    模式：显示 EE 位置(xyz) + 欧拉角(rpy) + 6 个手部目标值（共 12 列）
    """

    HISTORY_LEN = 200
    GRAPH_W = 460
    GRAPH_H = 90

    def __init__(
        self,
        cmd_queue: queue.Queue,
        arm_dof: int = 7,
        hand_dof: int = 6,
        action_mode: str = "joint",  # "joint" | "ee"
    ):
        self.cmd_queue = cmd_queue
        self.arm_dof = arm_dof
        self.hand_dof = hand_dof
        self.action_mode = action_mode

        # joint 模式：总自由度 = arm_dof + hand_dof
        # ee    模式：总自由度 = 6 (ee) + hand_dof
        self.ee_dof = 6
        self.ctrl_dof = (arm_dof if action_mode == "joint" else self.ee_dof) + hand_dof

        self._lock = threading.Lock()
        self._sel_idx = 0
        # _display_vals: joint模式=13维目标, ee模式=[pos(3)+rpy_deg(3)+hand(6)]
        self._display_vals = np.zeros(self.ctrl_dof)
        self._reward = 0.0
        self._ep_reward = 0.0
        self._step = 0
        self._terminated = False
        self._episode = 1
        self._reward_hist: list = []
        self._info_extra = {}

        # 步长（主线程也可直接读写）
        self._pos_step = 0.01  # ee 位置步长 (m)
        self._rot_step = 0.05  # ee 旋转步长 (rad)
        self._arm_step = 0.05  # joint 臂步长 (rad)
        self._hand_step = 0.001  # 手部推杆步长 (m)

        self._root = None
        self._ready = threading.Event()

    # -------- 公开接口（主线程调用） --------

    def update_state(
        self,
        sel_idx: int,
        display_vals: np.ndarray,  # joint: 13维目标; ee: pos(3)+rpy_deg(3)+hand(6)
        reward: float,
        ep_reward: float,
        step: int,
        terminated: bool,
        episode: int,
        info_extra: dict,
    ):
        with self._lock:
            self._sel_idx = sel_idx
            self._display_vals = display_vals.copy()
            self._reward = reward
            self._ep_reward = ep_reward
            self._step = step
            self._terminated = terminated
            self._episode = episode
            self._reward_hist.append(reward)
            if len(self._reward_hist) > self.HISTORY_LEN:
                self._reward_hist.pop(0)
            self._info_extra = dict(info_extra)

    def wait_ready(self, timeout: float = 10.0):
        self._ready.wait(timeout)

    def is_alive(self) -> bool:
        return self._root is not None and self._root.winfo_exists()

    # -------- Tkinter 主循环 --------

    def run(self):
        import tkinter as tk

        root = tk.Tk()
        self._root = root
        mode_label = "Joint Space" if self.action_mode == "joint" else "EE Space"
        root.title(f"Robot Arm Control [{mode_label}]")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)

        BG = "#1e1e2e"
        BG2 = "#2a2a3e"
        FG = "#cdd6f4"
        ACC = "#89b4fa"
        ARM_COL = "#a6e3a1"
        EE_P_COL = "#89dceb"  # ee 位置（青色）
        EE_R_COL = "#cba6f7"  # ee 旋转（紫色）
        HAND_COL = "#fab387"
        SEL_COL = "#f38ba8"
        WARN_COL = "#f9e2af"
        OK_COL = "#a6e3a1"

        FM = ("Consolas", 11)
        FS = ("Consolas", 9)
        FT = ("Consolas", 14, "bold")

        # ---- 标题 ----
        tk.Label(
            root,
            text=f"Robot Arm + Dexterous Hand [{mode_label} Control]",
            font=FT,
            bg=BG,
            fg=ACC,
        ).pack(pady=(10, 2))

        # ---- 按键说明 ----
        if self.action_mode == "joint":
            hint = "  ←/→ Select Joint    ↑/↓ Adjust    R Reset    O Open    C Close    Q Quit  "
        else:
            hint = "  ←/→ Select DOF    ↑/↓ Adjust    R Reset    O Open    C Close    Q Quit  "
            tk.Label(root, text=hint, font=FS, bg=BG2, fg=FG).pack(
                fill="x", padx=8, pady=(0, 4)
            )

        # ---- 步长设置 ----
        sf = tk.Frame(root, bg=BG)
        sf.pack(fill="x", padx=8, pady=2)

        if self.action_mode == "joint":
            tk.Label(sf, text="Arm Step (rad):", font=FS, bg=BG, fg=FG).pack(side="left")
            self._arm_step_var = tk.StringVar(value=str(self._arm_step))
            e1 = tk.Entry(
                sf,
                textvariable=self._arm_step_var,
                width=6,
                font=FS,
                bg=BG2,
                fg=FG,
                insertbackground=FG,
            )
            e1.pack(side="left", padx=(2, 12))
            e1.bind("<Return>", lambda *_: self._sync_steps())
        else:
            tk.Label(sf, text="Position Step (m):", font=FS, bg=BG, fg=FG).pack(side="left")
            self._pos_step_var = tk.StringVar(value=str(self._pos_step))
            e1 = tk.Entry(
                sf,
                textvariable=self._pos_step_var,
                width=6,
                font=FS,
                bg=BG2,
                fg=FG,
                insertbackground=FG,
            )
            e1.pack(side="left", padx=(2, 12))
            e1.bind("<Return>", lambda *_: self._sync_steps())

            tk.Label(sf, text="Rotation Step (rad):", font=FS, bg=BG, fg=FG).pack(side="left")
            self._rot_step_var = tk.StringVar(value=str(self._rot_step))
            e2 = tk.Entry(
                sf,
                textvariable=self._rot_step_var,
                width=6,
                font=FS,
                bg=BG2,
                fg=FG,
                insertbackground=FG,
            )
            e2.pack(side="left", padx=(2, 12))
            e2.bind("<Return>", lambda *_: self._sync_steps())

        tk.Label(sf, text="Hand Step (m):", font=FS, bg=BG, fg=FG).pack(side="left")
        self._hand_step_var = tk.StringVar(value=str(self._hand_step))
        e_hand = tk.Entry(
            sf,
            textvariable=self._hand_step_var,
            width=6,
            font=FS,
            bg=BG2,
            fg=FG,
            insertbackground=FG,
        )
        e_hand.pack(side="left", padx=(2, 0))
        e_hand.bind("<Return>", lambda *_: self._sync_steps())

        # ---- 自由度列表 ----
        jf = tk.Frame(root, bg=BG)
        jf.pack(fill="x", padx=8, pady=4)

        self._ctrl_labels = []  # 左列（臂/ee）
        self._hand_labels = []  # 右列（手部）

        # 左列
        lc = tk.Frame(jf, bg=BG2, padx=6, pady=4)
        lc.pack(side="left", fill="y", padx=(0, 4))

        if self.action_mode == "joint":
            tk.Label(
                lc, text="── Arm 7 DOF [target rad] ──", font=FS, bg=BG2, fg=ARM_COL
            ).pack()
            for i in range(self.arm_dof):
                lbl = tk.Label(
                    lc,
                    text=f"[{i}] {_ARM_JOINT_NAMES[i]}: 0.000",
                    font=FM,
                    bg=BG2,
                    fg=ARM_COL,
                    anchor="w",
                    width=30,
                )
                lbl.pack(fill="x")
                self._ctrl_labels.append((lbl, ARM_COL))
        else:
            tk.Label(
                lc, text="── End-Effector Pose [target values] ──", font=FS, bg=BG2, fg=EE_P_COL
            ).pack()
            # 位置 xyz
            for i in range(3):
                lbl = tk.Label(
                    lc,
                    text=f"[{i}] {_EE_DOF_NAMES[i]}: 0.000",
                    font=FM,
                    bg=BG2,
                    fg=EE_P_COL,
                    anchor="w",
                    width=30,
                )
                lbl.pack(fill="x")
                self._ctrl_labels.append((lbl, EE_P_COL))
            # 旋转 rpy（以度显示）
            rot_names = ["EE Roll  (deg)", "EE Pitch (deg)", "EE Yaw   (deg)"]
            for i in range(3):
                lbl = tk.Label(
                    lc,
                    text=f"[{3+i}] {rot_names[i]}: 0.00°",
                    font=FM,
                    bg=BG2,
                    fg=EE_R_COL,
                    anchor="w",
                    width=30,
                )
                lbl.pack(fill="x")
                self._ctrl_labels.append((lbl, EE_R_COL))

        # 右列（手部，两种模式一样）
        rc = tk.Frame(jf, bg=BG2, padx=6, pady=4)
        rc.pack(side="left", fill="y")
        ee_offset = self.arm_dof if self.action_mode == "joint" else self.ee_dof
        tk.Label(
            rc, text="── Hand 6 DOF [target m] ──", font=FS, bg=BG2, fg=HAND_COL
        ).pack()
        for i in range(self.hand_dof):
            idx = ee_offset + i
            lbl = tk.Label(
                rc,
                text=f"[{idx}] {_HAND_JOINT_NAMES[i]}: 0.00000",
                font=FM,
                bg=BG2,
                fg=HAND_COL,
                anchor="w",
                width=28,
            )
            lbl.pack(fill="x")
            self._hand_labels.append(lbl)

        # ---- 奖励/状态区 ----
        rf = tk.Frame(root, bg=BG2, padx=8, pady=6)
        rf.pack(fill="x", padx=8, pady=(4, 0))

        self._rwd_var    = tk.StringVar(value="Reward:  0.0000")
        self._ep_rwd_var = tk.StringVar(value="Cumulative:  0.0000")
        self._step_var   = tk.StringVar(value="Step:  0")
        self._ep_var     = tk.StringVar(value="Episode:  1")
        self._status_var = tk.StringVar(value="Status:  Running")
        self._extra_var = tk.StringVar(value="")

        for var, col in [
            (self._rwd_var, FG),
            (self._ep_rwd_var, FG),
            (self._step_var, FG),
            (self._ep_var, ACC),
            (self._status_var, FG),
        ]:
            tk.Label(rf, textvariable=var, font=FM, bg=BG2, fg=col, anchor="w").pack(
                fill="x"
            )
        tk.Label(
            rf,
            textvariable=self._extra_var,
            font=FS,
            bg=BG2,
            fg=WARN_COL,
            anchor="w",
            wraplength=460,
        ).pack(fill="x")

        # ---- 奖励折线图 ----
        tk.Label(root, text="Reward History", font=FS, bg=BG, fg=FG).pack(pady=(6, 0))
        self._canvas = tk.Canvas(
            root,
            width=self.GRAPH_W,
            height=self.GRAPH_H,
            bg="#11111b",
            highlightthickness=0,
        )
        self._canvas.pack(padx=8, pady=(0, 8))

        self._COLORS = dict(
            BG=BG,
            BG2=BG2,
            FG=FG,
            ACC=ACC,
            ARM=ARM_COL,
            EEP=EE_P_COL,
            EER=EE_R_COL,
            HAND=HAND_COL,
            SEL=SEL_COL,
            WARN=WARN_COL,
            OK=OK_COL,
        )

        root.bind("<KeyPress>", self._on_key)
        root.focus_set()
        self._ready.set()
        self._refresh()
        root.mainloop()
        self._root = None

    def _sync_steps(self):
        try:
            self._hand_step = float(self._hand_step_var.get())
        except ValueError:
            pass
        if self.action_mode == "joint":
            try:
                self._arm_step = float(self._arm_step_var.get())
            except ValueError:
                pass
        else:
            try:
                self._pos_step = float(self._pos_step_var.get())
            except ValueError:
                pass
            try:
                self._rot_step = float(self._rot_step_var.get())
            except ValueError:
                pass

    def _on_key(self, event):
        key = event.keysym
        if key == _KEY_NEXT_JNT:
            self.cmd_queue.put(("sel", +1))
        elif key == _KEY_PREV_JNT:
            self.cmd_queue.put(("sel", -1))
        elif key == _KEY_INCREASE:
            self.cmd_queue.put(("delta", +1.0))
        elif key == _KEY_DECREASE:
            self.cmd_queue.put(("delta", -1.0))
        elif key.lower() == _KEY_RESET:
            self.cmd_queue.put(("reset", None))
        elif key.lower() == _KEY_OPEN_HAND:
            self.cmd_queue.put(("open_hand", None))
        elif key.lower() == _KEY_CLOSE_HAND:
            self.cmd_queue.put(("close_hand", None))
        elif key.lower() == _KEY_QUIT:
            self.cmd_queue.put(("quit", None))

    def _refresh(self):
        with self._lock:
            sel = self._sel_idx
            vals = self._display_vals.copy()
            reward = self._reward
            ep_rwd = self._ep_reward
            step = self._step
            term = self._terminated
            ep = self._episode
            hist = list(self._reward_hist)
            extra = dict(self._info_extra)

        C = self._COLORS
        ee_offset = self.arm_dof if self.action_mode == "joint" else self.ee_dof

        # ---- 左列标签（臂/ee） ----
        for i, (lbl, base_col) in enumerate(self._ctrl_labels):
            val = vals[i] if i < len(vals) else 0.0
            is_sel = i == sel
            if self.action_mode == "joint":
                text = f"[{i}] {_ARM_JOINT_NAMES[i]}: {val:+.3f}"
            elif i < 3:
                text = f"[{i}] {_EE_DOF_NAMES[i]}: {val:+.4f} m"
            else:
                text = f"[{i}] {'Roll/Pitch/Yaw'.split('/')[i-3]}: {val:+.2f}°"
            lbl.config(
                text=text,
                fg=C["SEL"] if is_sel else base_col,
                font=("Consolas", 11, "bold") if is_sel else ("Consolas", 11),
                bg="#3e1e2e" if is_sel else C["BG2"],
            )

        # ---- 右列标签（手部） ----
        for i, lbl in enumerate(self._hand_labels):
            idx = ee_offset + i
            val = vals[idx] if idx < len(vals) else 0.0
            is_sel = idx == sel
            lbl.config(
                text=f"[{idx}] {_HAND_JOINT_NAMES[i]}: {val:+.5f}",
                fg=C["SEL"] if is_sel else C["HAND"],
                font=("Consolas", 11, "bold") if is_sel else ("Consolas", 11),
                bg="#3e1e2e" if is_sel else C["BG2"],
            )

        # ---- 奖励/状态 ----
        self._rwd_var.set(f"Reward:  {'▲' if reward >= 0 else '▼'} {reward:+.4f}")
        self._ep_rwd_var.set(f"Cumulative:  {ep_rwd:+.4f}")
        self._step_var.set(f"Step:  {step}  (no timeout)")
        self._ep_var.set(f"Episode:  {ep}")
        if term:
            self._status_var.set("Status:  ✅ Success (TERMINATED)")
        else:
            self._status_var.set("Status:  ▶ Running...")

        _skip = {"episode_steps", "episode_reward", "episode_count"}
        parts = []
        for k, v in extra.items():
            if k in _skip:
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            elif isinstance(v, bool):
                parts.append(f"{k}={'✓' if v else '✗'}")
            elif isinstance(v, np.ndarray):
                parts.append(f"{k}=[{', '.join(f'{x:.2f}' for x in v.flat)}]")
            else:
                parts.append(f"{k}={v}")
        self._extra_var.set("  ".join(parts))

        # ---- 折线图 ----
        cv = self._canvas
        cv.delete("all")
        W, H, pad = self.GRAPH_W, self.GRAPH_H, 6
        cv.create_line(pad, H // 2, W - pad, H // 2, fill="#313244", width=1)
        if len(hist) >= 2:
            mn, mx = min(hist), max(hist)
            span = max(mx - mn, 1e-6)

            def _y(v):
                return H - pad - (v - mn) / span * (H - 2 * pad)

            pts = []
            for i, v in enumerate(hist):
                pts.extend([pad + i / (len(hist) - 1) * (W - 2 * pad), _y(v)])
            cv.create_line(*pts, fill="#89b4fa", width=1, smooth=True)
            lv = hist[-1]
            lx = W - pad
            ly = _y(lv)
            cv.create_oval(lx - 3, ly - 3, lx + 3, ly + 3, fill="#f38ba8", outline="")
            cv.create_text(
                lx - 4,
                ly - 10,
                text=f"{lv:+.3f}",
                fill="#f38ba8",
                font=("Consolas", 8),
                anchor="e",
            )
            cv.create_text(
                pad + 2,
                H - pad,
                text=f"min:{mn:.3f}",
                fill="#6c7086",
                font=("Consolas", 8),
                anchor="sw",
            )
            cv.create_text(
                pad + 2,
                pad,
                text=f"max:{mx:.3f}",
                fill="#6c7086",
                font=("Consolas", 8),
                anchor="nw",
            )

        self._root.after(80, self._refresh)


def demo_keyboard_control(
    task_name: str = "pick_and_place",
    action_mode: str = "joint",
    controller_type: str = "osc",
    arm_step: float = 0.05,
    hand_step: float = 0.001,
    pos_step: float = 0.01,
    rot_step: float = 0.05,
):
    """
    键盘控制模式（joint / ee 双模式，禁用超时，仅手动 R 重置）.

    joint 模式：
      ←/→  切换关节（0-6 臂，7-12 手）
      ↑/↓  当前关节 ±arm_step rad（手部 ±hand_step m）

    ee 模式：
      ←/→  切换自由度（0-2 位置xyz，3-5 旋转rpy，6-11 手部）
      ↑/↓  位置 ±pos_step m，旋转 ±rot_step rad，手部 ±hand_step m
      旋转增量在世界坐标系下施加（左乘四元数）

    通用：
      R    重置回合（唯一触发重置的方式，无超时）
      O/C  张手/握手
      Q    退出
    """
    reg = TASK_REGISTRY[task_name]
    is_ee = action_mode == "ee"

    print("=" * 65)
    print(f" [Demo] 键盘控制 | 任务={reg['display_name']}  模式={action_mode}")
    print(f" controller={controller_type}  超时=禁用（手动R重置）")
    if is_ee:
        print(f" 位置步长={pos_step}m  旋转步长={rot_step}rad  手步长={hand_step}m")
    else:
        print(f" 臂步长={arm_step}rad  手步长={hand_step}m")
    print("=" * 65)

    # ---- 建环境 ----
    # max_episode_steps 设极大值禁用超时；action_scale=1.0 直接传差值/绝对增量
    robot_cfg = RobotConfig(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=999_999,  # 实质上禁用超时截断
        action_scale=1.0,
        action_scale_rot=1.0,
        action_scale_hand=1.0,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)
    HAND_MAX, HAND_MIN = 0.0095, 0.0

    # ---- 命令队列 + 面板 ----
    cmd_q: queue.Queue = queue.Queue()
    panel = KeyboardControlPanel(
        cmd_q, arm_dof=env.ARM_DOF, hand_dof=env.HAND_DOF, action_mode=action_mode
    )
    panel._arm_step = arm_step
    panel._hand_step = hand_step
    panel._pos_step = pos_step
    panel._rot_step = rot_step
    threading.Thread(target=panel.run, daemon=True).start()
    panel.wait_ready(timeout=10.0)

    traj_vis = EETrajectoryVisualizer(TrajectoryVisualStyle(), max_history=40)
    obs, info = env.reset(seed=42)

    # =====================================================================
    # 累积目标初始化
    # =====================================================================
    ee_target_pos, ee_target_quat = None, None
    joint_target = None
    hand_target = None
    if is_ee:
        # ee 模式：维护绝对末端位姿目标（pos + quat）+ 手部目标
        ee_target_pos, ee_target_quat = env.get_ee_pose()
        hand_target = env.get_hand_qpos().copy()
        # ctrl_dof = 6 + hand_dof = 12，面板用 [pos(3), rpy_deg(3), hand(6)]
    else:
        # joint 模式：维护绝对关节角目标（arm + hand）
        joint_target = np.concatenate([env.get_arm_qpos(), env.get_hand_qpos()])

    sel_idx = 0
    episode = 1
    step = 0
    ep_reward = 0.0
    reward = 0.0
    terminated = False
    running = True

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135

        while viewer.is_running() and running:

            # ================================================================
            # 处理命令队列
            # ================================================================
            pending_reset = False

            while not cmd_q.empty():
                try:
                    cmd, val = cmd_q.get_nowait()
                except queue.Empty:
                    break

                if cmd == "quit":
                    running = False
                    break

                elif cmd == "reset":
                    pending_reset = True

                elif cmd == "sel":
                    n_ctrl = env.HAND_DOF + (env.ARM_DOF if not is_ee else 6)
                    sel_idx = (sel_idx + val) % n_ctrl

                elif cmd == "open_hand":
                    if is_ee:
                        hand_target[:] = HAND_MIN
                    else:
                        joint_target[env.ARM_DOF :] = HAND_MIN

                elif cmd == "close_hand":
                    if is_ee:
                        hand_target[:] = HAND_MAX
                    else:
                        joint_target[env.ARM_DOF :] = HAND_MAX

                elif cmd == "delta":
                    if is_ee:
                        # ------ ee 模式增量 ------
                        if sel_idx < 3:
                            # 位置 xyz
                            ee_target_pos[sel_idx] += val * panel._pos_step
                        elif sel_idx < 6:
                            # 旋转（世界坐标系下绕对应轴旋转）
                            axis_idx = sel_idx - 3  # 0=X, 1=Y, 2=Z
                            axis = np.zeros(3)
                            axis[axis_idx] = 1.0
                            angle = val * panel._rot_step
                            # 轴角 → 四元数增量
                            dq = np.zeros(4)
                            mujoco.mju_axisAngle2Quat(dq, axis, angle)
                            # 世界系左乘：target_quat = dq ⊗ current_target_quat
                            new_quat = np.zeros(4)
                            mujoco.mju_mulQuat(new_quat, dq, ee_target_quat)
                            # 归一化防止数值漂移
                            norm = np.linalg.norm(new_quat)
                            if norm > 1e-8:
                                ee_target_quat[:] = new_quat / norm
                        else:
                            # 手部推杆
                            hi = sel_idx - 6
                            hand_target[hi] = np.clip(
                                hand_target[hi] + val * panel._hand_step,
                                HAND_MIN,
                                HAND_MAX,
                            )
                    else:
                        # ------ joint 模式增量 ------
                        if sel_idx < env.ARM_DOF:
                            joint_target[sel_idx] += val * panel._arm_step
                        else:
                            joint_target[sel_idx] = np.clip(
                                joint_target[sel_idx] + val * panel._hand_step,
                                HAND_MIN,
                                HAND_MAX,
                            )

            if not running:
                break

            # ================================================================
            # 手动重置（只在 pending_reset 或 terminated 时触发）
            # ================================================================
            if pending_reset or terminated:
                if terminated:
                    print(
                        f"[回合 {episode}] ✅ 任务成功！  累积奖励={ep_reward:.4f}  步数={step}"
                    )
                else:
                    print(
                        f"[回合 {episode}] 🔄 手动重置    累积奖励={ep_reward:.4f}  步数={step}"
                    )

                obs, info = env.reset()
                traj_vis.reset()

                # reset 后重新从真实状态初始化目标
                ee_target_pos, ee_target_quat, joint_target = None, None, None
                if is_ee:
                    ee_target_pos, ee_target_quat = env.get_ee_pose()
                    hand_target = env.get_hand_qpos().copy()
                else:
                    joint_target = np.concatenate(
                        [env.get_arm_qpos(), env.get_hand_qpos()]
                    )

                episode += 1
                step = 0
                ep_reward = 0.0
                reward = 0.0
                terminated = False

            # ================================================================
            # 构造 action 并 step
            # ================================================================
            if is_ee:
                # ee 模式：直接传绝对末端目标和手部目标
                # _apply_ee_action 期望归一化到 [-1,1] 的增量，但我们 scale=1.0 且直接计算差值
                # 实际做法：传当前位置差作为位置增量，传姿态差作为旋转增量，手部传差值
                cur_pos, cur_quat = env.get_ee_pose()
                cur_hand = env.get_hand_qpos()

                pos_delta = ee_target_pos - cur_pos  # (3,) 米

                # 姿态差 → 轴角增量（局部坐标系，与 _apply_ee_action 一致）
                cur_quat_inv = np.zeros(4)
                mujoco.mju_negQuat(cur_quat_inv, cur_quat)
                dq = np.zeros(4)
                mujoco.mju_mulQuat(dq, ee_target_quat, cur_quat_inv)
                rot_delta = np.zeros(3)
                mujoco.mju_quat2Vel(rot_delta, dq, 1.0)  # (3,) 弧度

                hand_delta = hand_target - cur_hand  # (6,) 米

                action = np.concatenate([pos_delta, rot_delta, hand_delta]).astype(
                    np.float32
                )
            else:
                # joint 模式：差值方案
                current_qpos = np.concatenate([env.get_arm_qpos(), env.get_hand_qpos()])
                action = (joint_target - current_qpos).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            # truncated 在键盘模式下永远不触发（max_episode_steps=999999）
            # 但即便万一触发也只打印日志，不自动重置
            if truncated:
                print(f"[警告] 意外截断（步数={step}），请检查 max_episode_steps 设置")

            step += 1
            ep_reward += reward

            # ================================================================
            # 可视化
            # ================================================================
            viewer.user_scn.ngeom = 0
            actual_pos = _get_ee_position(env)
            if actual_pos is not None:
                traj_vis.update(actual_pos, target_pos=ee_target_pos, target_quat=ee_target_quat)
                traj_vis.draw(viewer)

            # ---- 构造面板显示值 ----
            if is_ee:
                rpy_deg = _quat_to_euler_deg(ee_target_quat)
                display_vals = np.concatenate([ee_target_pos, rpy_deg, hand_target])
            else:
                display_vals = joint_target.copy()

            info_extra = {
                k: v
                for k, v in info.items()
                if k not in ("episode_steps", "episode_reward", "episode_count")
                and (not isinstance(v, np.ndarray) or v.size <= 6)
            }

            panel.update_state(
                sel_idx=sel_idx,
                display_vals=display_vals,
                reward=reward,
                ep_reward=ep_reward,
                step=step,
                terminated=terminated,
                episode=episode,
                info_extra=info_extra,
            )

            # ---- 触觉热力图 ----
            heatmap = render_tactile_heatmap(obs)
            cv2.namedWindow("Tactile", cv2.WINDOW_NORMAL)
            cv2.imshow("Tactile", heatmap)
            cv2.resizeWindow("Tactile", 1000, 360)

            # ---- 相机画面 ----
            if "camera_rgb" in obs:
                cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
                mode_str = "EE" if is_ee else "JNT"

                overlay_texts = [
                    f"[{mode_str}] Reward: {reward:+.4f}  Cum: {ep_reward:+.4f}",
                    f"Step: {step}  Ep: {episode}  (R=reset, no timeout)",
                ]

                # ===== 1. 半透明背景条 =====
                overlay = cam_bgr.copy()
                bar_h = 20 * len(overlay_texts) + 10
                cv2.rectangle(overlay, (0, 0), (640, bar_h), (0, 0, 0), -1)
                cam_bgr = cv2.addWeighted(overlay, 0.4, cam_bgr, 0.6, 0)

                # ===== 2. 描边文字 =====
                for li, txt in enumerate(overlay_texts):
                    y = 18 + li * 18

                    # 黑色描边（关键）
                    cv2.putText(
                        cam_bgr,
                        txt,
                        (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )

                    # 白色主文字（比绿色更稳）
                    cv2.putText(
                        cam_bgr,
                        txt,
                        (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # ===== 3. SUCCESS 提示（同样加描边）=====
                if terminated:
                    msg = "TASK SUCCESS!"
                    pos = (60, 130)

                    cv2.putText(cam_bgr, msg, pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 0, 0), 4, cv2.LINE_AA)

                    cv2.putText(cam_bgr, msg, pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 80), 2, cv2.LINE_AA)

                cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
                cv2.imshow("Camera", cam_bgr)
                cv2.resizeWindow("Camera", 640, 480)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False
                break

            viewer.sync()

    cv2.destroyAllWindows()
    env.close()
    print("\n[键盘控制] 已退出。")


# ====================== 入口 ======================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="通用任务环境演示",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_REGISTRY.keys()),
        default="pick_and_place",
        help=(
            "要演示的任务名称：\n"
            + "\n".join(f"  {k}: {v['display_name']}" for k, v in TASK_REGISTRY.items())
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["random", "verify", "benchmark", "keyboard"],
        default="random",
        help=(
            "演示模式：\n"
            "  random    随机策略 + 可视化\n"
            "  verify    观测空间验证\n"
            "  benchmark 无渲染高速基准测试\n"
            "  keyboard  键盘逐关节控制（奖励/终止/截断实时显示）\n"
        ),
    )
    parser.add_argument(
        "--no-render", action="store_true", help="禁用可视化（random 模式）"
    )
    parser.add_argument(
        "--no-traj", action="store_true", help="禁用末端执行器轨迹可视化"
    )
    parser.add_argument("--episodes", type=int, default=3, help="演示回合数")
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
    parser.add_argument(
        "--arm-step",
        type=float,
        default=0.05,
        help="键盘模式：机械臂单次调整步长（弧度），默认 0.05",
    )
    parser.add_argument(
        "--hand-step",
        type=float,
        default=0.001,
        help="键盘模式：灵巧手单次调整步长（米），默认 0.001",
    )
    args = parser.parse_args()

    render = not args.no_render
    show_traj = not args.no_traj

    print(f"\n{'='*65}")
    print(f"  任务:       {TASK_REGISTRY[args.task]['display_name']}")
    print(f"  模式:       {args.mode}")
    if args.mode == "random":
        print(f"  渲染:       {'是' if render else '否'}")
        print(f"  轨迹可视化: {'是' if show_traj else '否'}")
    elif args.mode == "keyboard":
        print(f"  臂步长:     {args.arm_step} rad")
        print(f"  手步长:     {args.hand_step} m")
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
    elif args.mode == "keyboard":
        demo_keyboard_control(
            task_name=args.task,
            action_mode=args.action_mode,
            controller_type=args.controller,
            arm_step=args.arm_step,
            hand_step=args.hand_step,
        )
