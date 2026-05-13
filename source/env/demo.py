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

    def __init__(self, style: TrajectoryVisualStyle, max_history: int = 2000):
        self.style = style
        self.max_history = max_history
        self.actual_pos: Optional[np.ndarray] = None
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.history: list = []

    def update(
        self,
        actual_pos: np.ndarray,
        target_pos: Optional[np.ndarray] = None,
        target_quat: Optional[np.ndarray] = None,
    ):
        self.actual_pos = actual_pos.copy()
        if target_pos is not None:
            self.target_pos = target_pos.copy()
        if target_quat is not None:
            self.target_quat = target_quat.copy()
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

        # 4. 目标朝向坐标轴（局部坐标系三轴）
        if self.target_quat is not None and self.target_pos is not None:
            target_mat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(target_mat, self.target_quat)
            rot_matrix = target_mat.reshape(3, 3)

            axis_len = 0.1
            colors = [
                np.array([1, 0, 0, 1], dtype=np.float32),  # X: 红
                np.array([0, 1, 0, 1], dtype=np.float32),  # Y: 绿
                np.array([0, 0, 1, 1], dtype=np.float32),  # Z: 蓝
            ]

            for i in range(3):
                if viewer.user_scn.ngeom >= max_geoms - safety_margin:
                    break
                axis_dir = rot_matrix[:, i]
                from_pos = self.target_pos.astype(np.float64)
                to_pos = (self.target_pos + axis_dir * axis_len).astype(np.float64)

                geom_id = viewer.user_scn.ngeom
                geom = viewer.user_scn.geoms[geom_id]
                mujoco.mjv_initGeom(
                    geom,
                    type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    size=np.array([0.002, 0.002, axis_len], dtype=np.float64),
                    pos=from_pos,
                    mat=np.eye(3).flatten().astype(np.float64),
                    rgba=colors[i],
                )
                mujoco.mjv_connector(
                    geom,
                    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    0.002,
                    from_pos,
                    to_pos,
                )
                viewer.user_scn.ngeom += 1

    def reset(self):
        self.history.clear()
        self.actual_pos = None
        self.target_pos = None
        self.target_quat = None


# ====================== 【新增】指尖连线中点可视化 ======================


@dataclass
class FingertipMidpointStyle:
    """指尖连线中点可视化样式配置."""
    line_rgba: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.8, 0.0, 0.9])  # 金黄色
    )
    midpoint_rgba: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.2, 0.8, 1.0])  # 品红色
    )
    endpoint_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.8, 1.0, 0.7])  # 浅蓝色
    )
    line_width: float = 0.003  # 3mm
    midpoint_size: float = 0.012  # 12mm
    endpoint_size: float = 0.008  # 8mm


class FingertipMidpointVisualizer:
    """
    绘制 thumb 和 finger_3 指尖连线及其中点.

    使用 env.get_site_pos() 获取两个 fingertip site 的世界坐标，
    在 viewer 中绘制：
      - 两个端点（半透明小球）
      - 连线（圆柱体）
      - 中点（高亮球体）
    """

    def __init__(self, style: Optional[FingertipMidpointStyle] = None):
        self.style = style or FingertipMidpointStyle()
        self.thumb_pos: Optional[np.ndarray] = None
        self.finger3_pos: Optional[np.ndarray] = None
        self.midpoint: Optional[np.ndarray] = None

    def update(self, env: RobotArmEnvBase):
        """从环境获取两个 fingertip 位置并计算中点."""
        try:
            self.thumb_pos = env.get_site_pos("inspirehand_fingertip_thumb").copy()
            self.finger3_pos = env.get_site_pos("inspirehand_fingertip_3").copy()
            self.midpoint = (self.thumb_pos + self.finger3_pos) / 2.0
        except ValueError:
            # 如果 site 不存在则静默跳过
            self.thumb_pos = None
            self.finger3_pos = None
            self.midpoint = None
            print("[Warning] 无法获取 fingertip site 位置，指尖中点可视化将被跳过。")

    def draw(self, viewer) -> None:
        """在 MuJoCo viewer 中绘制几何体."""
        if self.midpoint is None or self.thumb_pos is None or self.finger3_pos is None:
            return

        max_geoms = 1000
        safety_margin = 50

        # 绘制中点（品红色大球，高亮）
        if viewer.user_scn.ngeom < max_geoms - safety_margin:
            geom_id = viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_id],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.style.midpoint_size, 0, 0],
                pos=self.midpoint,
                mat=np.eye(3).flatten(),
                rgba=self.style.midpoint_rgba,
            )
            viewer.user_scn.ngeom += 1

    def reset(self):
        self.thumb_pos = None
        self.finger3_pos = None
        self.midpoint = None


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
    show_fingertip_midpoint: bool = True,  # [新增] 控制是否显示指尖中点
):
    """随机策略演示：仿真窗口 + 触觉热力图 + 任务状态信息 + 末端轨迹可视化 + 指尖中点."""
    reg = TASK_REGISTRY[task_name]
    print("=" * 65)
    print(f" [Demo] 随机策略 | 任务={reg['display_name']}")
    print(f" action_mode={action_mode}, controller={controller_type}")
    print(f" 末端轨迹可视化: {'开启' if show_ee_traj else '关闭'}")
    print(f" 指尖中点可视化: {'开启' if show_fingertip_midpoint else '关闭'}")  # [新增]
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

        # [新增] 初始化指尖中点可视化器
        ft_vis = None
        if show_fingertip_midpoint:
            ft_vis = FingertipMidpointVisualizer()

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

                # [新增] 指尖中点可视化
                if show_fingertip_midpoint and ft_vis is not None:
                    # 注意：如果上面 traj_vis 已经设置了 ngeom=0，这里不需要重复设置
                    # 但如果只开启 fingertip 可视化，需要确保 ngeom 重置
                    if not show_ee_traj:
                        viewer.user_scn.ngeom = 0
                    ft_vis.update(env)
                    ft_vis.draw(viewer)

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
                    print(f"[Episode {episode+1}] {status} | " f"steps={step}")
                    episode += 1
                    step = 0
                    if traj_vis is not None:
                        traj_vis.reset()
                    if ft_vis is not None:  # [新增]
                        ft_vis.reset()
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
    "F2-4 Sync",       # [修改] 同步调节第2、3、4 DOF
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

# ee 旋转轴显示名（面板右侧标签用）
_EE_ROT_NAMES = ["Roll", "Pitch", "Yaw"]

# 按键绑定（键盘模式）
_KEY_INCREASE = "Up"
_KEY_DECREASE = "Down"
_KEY_PREV_JNT = "Left"
_KEY_NEXT_JNT = "Right"
_KEY_RESET = "r"
_KEY_OPEN_HAND = "o"
_KEY_CLOSE_HAND = "c"
_KEY_QUIT = "q"
_KEY_GRIPPER_OPEN = "g"


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

    joint 模式：显示 7 个臂关节目标值 + 7 个手部目标值（共 14 列）
    ee    模式：显示 EE 位置(xyz) + 欧拉角(rpy) + 7 个手部目标值（共 13 列）

    修复项：
    - [FIX-3] 步长参数通过构造函数传入，不再依赖外部直接赋值私有属性
    - [FIX-4] joint 模式按键提示 Label 现在正确调用 .pack()
    - [FIX-8] ee 旋转轴名称改用 _EE_ROT_NAMES 列表，不再用字符串 split
    - [FIX-9] 命令队列消费改为纯 try/except，移除冗余的 empty() 检查
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
        # [FIX-3] 步长通过构造函数注入，不再暴露私有属性给外部赋值
        arm_step: float = 0.05,
        hand_step: float = 0.0005,
        pos_step: float = 0.01,
        rot_step: float = 0.05,
    ):
        self.cmd_queue = cmd_queue
        self.arm_dof = arm_dof
        self.hand_dof = hand_dof
        self.action_mode = action_mode

        self.ee_dof = 6
        self.hand_display_num = hand_dof + 1
        self.ctrl_dof = (arm_dof if action_mode == "joint" else self.ee_dof) + self.hand_display_num

        self._lock = threading.Lock()
        self._sel_idx = 0
        self._display_vals = np.zeros(self.ctrl_dof)
        self._reward = 0.0
        self._ep_reward = 0.0
        self._step = 0
        self._terminated = False
        self._episode = 1
        self._reward_hist: list = []
        self._info_extra = {}

        # [FIX-3] 步长由外部传入，初始值统一在此处设置
        self._pos_step = pos_step
        self._rot_step = rot_step
        self._arm_step = arm_step
        self._hand_step = hand_step

        self._root = None
        self._ready = threading.Event()

    # -------- 公开接口（主线程调用） --------

    def update_state(
        self,
        sel_idx: int,
        display_vals: np.ndarray,
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
        root.title(f"Robot Arm Control [{mode_label}]  ← Click to Focus")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)

        BG = "#1e1e2e"
        BG2 = "#2a2a3e"
        FG = "#cdd6f4"
        ACC = "#89b4fa"
        ARM_COL = "#a6e3a1"
        EE_P_COL = "#89dceb"
        EE_R_COL = "#cba6f7"
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

        # [FIX-6] 焦点提示始终可见
        tk.Label(
            root,
            text="Click the panel and use keyboard to control!",
            font=FS,
            bg="#3e2a00",
            fg=WARN_COL,
        ).pack(fill="x", padx=8, pady=(0, 2))

        # ---- 按键说明
        if self.action_mode == "joint":
            hint = " ←/→ Select Joint, ↑/↓ Adjust, R Reset, O Open, C Close, G Gripper Open, Q Quit "
        else:
            hint = " ←/→ Select DOF, ↑/↓ Adjust, R Reset, O Open, C Close, G Gripper Open, Q Quit "
        tk.Label(root, text=hint, font=FS, bg=BG2, fg=FG).pack(
            fill="x", padx=8, pady=(0, 4)
        )

        # ---- 步长设置 ----
        sf = tk.Frame(root, bg=BG)
        sf.pack(fill="x", padx=8, pady=2)

        if self.action_mode == "joint":
            tk.Label(sf, text="Arm Step (rad):", font=FS, bg=BG, fg=FG).pack(
                side="left"
            )
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
            tk.Label(sf, text="Position Step (m):", font=FS, bg=BG, fg=FG).pack(
                side="left"
            )
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

            tk.Label(sf, text="Rotation Step (rad):", font=FS, bg=BG, fg=FG).pack(
                side="left"
            )
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

        self._ctrl_labels = []
        self._hand_labels = []

        # 左列（臂 / ee）
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
                lc,
                text="── End-Effector Pose [target values] ──",
                font=FS,
                bg=BG2,
                fg=EE_P_COL,
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
            # 旋转 rpy
            for i in range(3):
                lbl = tk.Label(
                    lc,
                    text=f"[{3+i}] EE {_EE_ROT_NAMES[i]:5s} (deg): 0.00°",
                    font=FM,
                    bg=BG2,
                    fg=EE_R_COL,
                    anchor="w",
                    width=30,
                )
                lbl.pack(fill="x")
                self._ctrl_labels.append((lbl, EE_R_COL))

        # 右列（手部）
        rc = tk.Frame(jf, bg=BG2, padx=6, pady=4)
        rc.pack(side="left", fill="y")
        ee_offset = self.arm_dof if self.action_mode == "joint" else self.ee_dof
        tk.Label(
            rc, text="── Hand 6 Dof [target m] ──", font=FS, bg=BG2, fg=HAND_COL
        ).pack()
        for i in range(self.hand_display_num):
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

        self._rwd_var = tk.StringVar(value="Reward:  0.0000")
        self._ep_rwd_var = tk.StringVar(value="Cumulative:  0.0000")
        self._step_var = tk.StringVar(value="Step:  0")
        self._ep_var = tk.StringVar(value="Episode:  1")
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
        """从 Entry 控件读取并更新步长."""
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
        elif key.lower() == _KEY_GRIPPER_OPEN:
            self.cmd_queue.put(("gripper_open", None))

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
                # [FIX-8] 使用列表索引，不再用字符串 split
                text = f"[{i}] EE {_EE_ROT_NAMES[i-3]:5s}: {val:+.2f}°"
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
            # [修改] 同步条目(F2-4 Sync) 显示2,3,4的平均值
            if i == 6:  # F2-4 Sync 条目
                # 显示 F2,F3,F4 的平均值
                v2 = vals[ee_offset + 2] if (ee_offset + 2) < len(vals) else 0.0
                v3 = vals[ee_offset + 3] if (ee_offset + 3) < len(vals) else 0.0
                v4 = vals[ee_offset + 4] if (ee_offset + 4) < len(vals) else 0.0
                avg_val = (v2 + v3 + v4) / 3.0
                text = f"[{idx}] {_HAND_JOINT_NAMES[i]}: {avg_val:+.5f} (avg)"
            else:
                text = f"[{idx}] {_HAND_JOINT_NAMES[i]}: {val:+.5f}"
            lbl.config(
                text=text,
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
            self._status_var.set("Status:  Success — Press R to reset")
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
    hand_step: float = 0.0005,
    pos_step: float = 0.01,
    rot_step: float = 0.05,
    show_fingertip_midpoint: bool = True,  # [新增] 控制是否显示指尖中点
):
    """
    键盘控制模式（joint / ee 双模式，禁用超时，仅手动 R 重置）.

    joint 模式：
      ←/→  切换关节（0-6 臂，7-13 手）  [修改] 手部条目变为7个
      ↑/↓  当前关节 ±arm_step rad（手部 ±hand_step m）

    ee 模式：
      ←/→  切换自由度（0-2 位置xyz，3-5 旋转rpy，6-12 手部） [修改] 手部条目变为7个
      ↑/↓  位置 ±pos_step m，旋转 ±rot_step rad（局部坐标系），手部 ±hand_step m

    通用：
      R    重置回合（唯一触发重置的方式，无超时）
      O/C  张手/握手
      Q    退出

    修复项：
    - [FIX-1] terminated 不再自动触发重置，仅在用户按 R 后重置；
              成功后停在原地，面板显示 "Press R to reset"
    - [FIX-2] ee 旋转增量注释修正为"局部坐标系右乘"，与实现一致
    - [FIX-3] 步长通过构造函数传入 KeyboardControlPanel，不再外部直接赋值私有属性
    """
    reg = TASK_REGISTRY[task_name]
    is_ee = action_mode == "ee"

    print("=" * 65)
    print(f" [Demo] 键盘控制 | 任务={reg['display_name']}  模式={action_mode}")
    print(f" controller={controller_type}  超时=禁用（手动R重置）")
    if is_ee:
        print(
            f" 位置步长={pos_step}m  旋转步长={rot_step}rad（局部系）  手步长={hand_step}m"
        )
    else:
        print(f" 臂步长={arm_step}rad  手步长={hand_step}m")
    print(f" 指尖中点可视化: {'开启' if show_fingertip_midpoint else '关闭'}")  # [新增]
    print("=" * 65)

    robot_cfg = RobotConfig(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=999_999,
        action_scale=1.0,
        action_scale_rot=1.0,
        action_scale_hand=1.0,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)
    HAND_MAX, HAND_MIN = 0.0095, 0.0
    GRIPPER_OPEN_POS = np.array(
        [HAND_MAX, HAND_MAX, HAND_MIN, HAND_MIN, HAND_MIN, HAND_MAX]
    )

    # [FIX-3] 步长统一通过构造函数传入
    cmd_q: queue.Queue = queue.Queue()
    panel = KeyboardControlPanel(
        cmd_q,
        arm_dof=env.ARM_DOF,
        hand_dof=env.HAND_DOF,
        action_mode=action_mode,
        arm_step=arm_step,
        hand_step=hand_step,
        pos_step=pos_step,
        rot_step=rot_step,
    )
    threading.Thread(target=panel.run, daemon=True).start()
    panel.wait_ready(timeout=10.0)

    traj_vis = EETrajectoryVisualizer(TrajectoryVisualStyle(), max_history=40)
    
    # [新增] 初始化指尖中点可视化器
    ft_vis = FingertipMidpointVisualizer() if show_fingertip_midpoint else None
    
    obs, info = env.reset(seed=42)

    # ---- 累积目标初始化 ----
    ee_target_pos, ee_target_quat = None, None
    joint_target = None
    hand_target = None

    def _init_targets_from_env():
        """从环境当前真实状态初始化所有控制目标."""
        nonlocal ee_target_pos, ee_target_quat, joint_target, hand_target
        if is_ee:
            ee_target_pos, ee_target_quat = env.get_ee_pose()
            ee_target_pos = ee_target_pos.copy()
            ee_target_quat = ee_target_quat.copy()
            hand_target = env.get_hand_qpos().copy()
        else:
            joint_target = np.concatenate([env.get_arm_qpos(), env.get_hand_qpos()])

    _init_targets_from_env()

    sel_idx = 0
    episode = 1
    step = 0
    ep_reward = 0.0
    reward = 0.0
    terminated = False  # [FIX-1] 记录成功状态，但不自动重置
    running = True

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135

        while viewer.is_running() and running:

            # ================================================================
            # [FIX-9] 命令队列消费：纯 try/except，移除冗余的 empty() 检查
            # ================================================================
            pending_reset = False

            while True:
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
                    # [修改] 控制条目数 = 臂/ee + 手显示条目7个
                    n_ctrl = (env.ARM_DOF if not is_ee else 6) + 7  # 7 = 6 DOF + 1 Sync
                    sel_idx = (sel_idx + val) % n_ctrl

                elif cmd == "open_hand":
                    if is_ee:
                        hand_target[:] = HAND_MIN
                    else:
                        # [修改] 包括同步条目对应的物理DOF
                        joint_target[env.ARM_DOF : env.ARM_DOF + env.HAND_DOF] = HAND_MIN

                elif cmd == "close_hand":
                    if is_ee:
                        hand_target[:] = HAND_MAX
                    else:
                        # [修改] 包括同步条目对应的物理DOF
                        joint_target[env.ARM_DOF : env.ARM_DOF + env.HAND_DOF] = HAND_MAX
                elif cmd == "gripper_open":
                    if is_ee:
                        hand_target[:] = GRIPPER_OPEN_POS
                    else:
                        # [修改] 包括同步条目对应的物理DOF
                        joint_target[env.ARM_DOF : env.ARM_DOF + env.HAND_DOF] = GRIPPER_OPEN_POS

                elif cmd == "delta":
                    # [FIX-1] 成功后不再响应调整指令，等待 R 重置
                    if terminated:
                        continue

                    if is_ee:
                        if sel_idx < 3:
                            ee_target_pos[sel_idx] += val * panel._pos_step
                        elif sel_idx < 6:
                            # [FIX-2] 局部坐标系右乘旋转增量（注释与实现一致）
                            axis_idx = sel_idx - 3  # 0=X, 1=Y, 2=Z（末端局部系）
                            axis = np.zeros(3)
                            axis[axis_idx] = 1.0
                            angle = val * panel._rot_step
                            dq = np.zeros(4)
                            mujoco.mju_axisAngle2Quat(dq, axis, angle)
                            new_quat = np.zeros(4)
                            mujoco.mju_mulQuat(new_quat, ee_target_quat, dq)
                            norm = np.linalg.norm(new_quat)
                            if norm > 1e-8:
                                ee_target_quat[:] = new_quat / norm
                        elif sel_idx < 13:  # 手部条目 6-12 (6个独立DOF + 1个同步条目)
                            # [修改] 处理手部DOF和同步条目
                            hi = sel_idx - 6  # 0-6
                            if hi < 6:
                                # 普通DOF 0-5
                                hand_target[hi] = np.clip(
                                    hand_target[hi] + val * panel._hand_step,
                                    HAND_MIN,
                                    HAND_MAX,
                                )
                            else:
                                # [新增] 同步条目 hi=6，同时调节 DOF 2,3,4
                                for sync_idx in [2, 3, 4]:
                                    hand_target[sync_idx] = np.clip(
                                        hand_target[sync_idx] + val * panel._hand_step,
                                        HAND_MIN,
                                        HAND_MAX,
                                    )
                    else:
                        if sel_idx < env.ARM_DOF:
                            joint_target[sel_idx] += val * panel._arm_step
                        elif sel_idx < env.ARM_DOF + env.HAND_DOF:
                            # 普通手部DOF
                            joint_target[sel_idx] = np.clip(
                                joint_target[sel_idx] + val * panel._hand_step,
                                HAND_MIN,
                                HAND_MAX,
                            )
                        else:
                            # [新增] 同步条目，同时调节 DOF 2,3,4
                            base_idx = env.ARM_DOF
                            for sync_idx in [base_idx + 2, base_idx + 3, base_idx + 4]:
                                joint_target[sync_idx] = np.clip(
                                    joint_target[sync_idx] + val * panel._hand_step,
                                    HAND_MIN,
                                    HAND_MAX,
                                )

            if not running:
                break

            # ================================================================
            # [FIX-1] 重置：只在用户主动按 R（pending_reset）时触发
            #         terminated 仅作展示，不再自动重置
            # ================================================================
            if pending_reset:
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
                if ft_vis is not None:  # [新增]
                    ft_vis.reset()
                _init_targets_from_env()

                episode += 1
                step = 0
                ep_reward = 0.0
                reward = 0.0
                terminated = False

            # ================================================================
            # 构造 action 并 step
            # [FIX-1] 已成功时跳过 step，原地保持
            # ================================================================
            if not terminated:
                if is_ee:
                    cur_pos, cur_quat = env.get_ee_pose()
                    cur_hand = env.get_hand_qpos()

                    pos_delta = ee_target_pos - cur_pos

                    cur_quat_inv = np.zeros(4)
                    mujoco.mju_negQuat(cur_quat_inv, cur_quat)
                    dq = np.zeros(4)
                    mujoco.mju_mulQuat(dq, ee_target_quat, cur_quat_inv)
                    rot_delta = np.zeros(3)
                    mujoco.mju_quat2Vel(rot_delta, dq, 1.0)

                    hand_delta = hand_target - cur_hand

                    action = np.concatenate([pos_delta, rot_delta, hand_delta]).astype(
                        np.float32
                    )
                else:
                    current_qpos = np.concatenate(
                        [env.get_arm_qpos(), env.get_hand_qpos()]
                    )
                    action = (joint_target - current_qpos).astype(np.float32)

                obs, reward, terminated, truncated, info = env.step(action)

                if truncated:
                    print(
                        f"[警告] 意外截断（步数={step}），请检查 max_episode_steps 设置"
                    )

                step += 1
                ep_reward += reward

                if terminated:
                    print(
                        f"[回合 {episode}] ✅ 任务成功！  步数={step}  累积奖励={ep_reward:.4f}"
                        f"  → 按 R 重置"
                    )

            # ================================================================
            # 可视化
            # ================================================================
            viewer.user_scn.ngeom = 0
            actual_pos = _get_ee_position(env)
            if actual_pos is not None:
                traj_vis.update(
                    actual_pos,
                    target_pos=ee_target_pos,
                    target_quat=ee_target_quat,
                )
                traj_vis.draw(viewer)

            # [新增] 指尖中点可视化
            if ft_vis is not None:
                ft_vis.update(env)
                ft_vis.draw(viewer)

            # ---- 构造面板显示值 ----
            if is_ee:
                rpy_deg = _quat_to_euler_deg(ee_target_quat)
                # [修改] 添加同步条目的显示值（F2,F3,F4的平均值）
                sync_val = np.array([(hand_target[2] + hand_target[3] + hand_target[4]) / 3.0])
                display_vals = np.concatenate([ee_target_pos, rpy_deg, hand_target, sync_val])
            else:
                # [修改] 添加同步条目的显示值（F2,F3,F4的平均值）
                hand_part = joint_target[env.ARM_DOF : env.ARM_DOF + env.HAND_DOF]
                sync_val = np.array([(hand_part[2] + hand_part[3] + hand_part[4]) / 3.0])
                display_vals = np.concatenate([joint_target[:env.ARM_DOF + env.HAND_DOF], sync_val])

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

                overlay = cam_bgr.copy()
                bar_h = 20 * len(overlay_texts) + 10
                cv2.rectangle(overlay, (0, 0), (640, bar_h), (0, 0, 0), -1)
                cam_bgr = cv2.addWeighted(overlay, 0.4, cam_bgr, 0.6, 0)

                for li, txt in enumerate(overlay_texts):
                    y = 18 + li * 18
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

                if terminated:
                    msg = "TASK SUCCESS!"
                    cv2.putText(
                        cam_bgr,
                        msg,
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        cam_bgr,
                        msg,
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 80),
                        2,
                        cv2.LINE_AA,
                    )

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
    # [新增] 控制指尖中点可视化开关
    parser.add_argument(
        "--no-ft-mid", action="store_true", help="禁用指尖连线中点可视化"
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
        default=0.0005,
        help="键盘模式：灵巧手单次调整步长（米），默认 0.0005",
    )
    parser.add_argument(
        "--pos-step",
        type=float,
        default=0.01,
        help="键盘模式（ee）：位置单次调整步长（米），默认 0.01",
    )
    parser.add_argument(
        "--rot-step",
        type=float,
        default=0.05,
        help="键盘模式（ee）：旋转单次调整步长（弧度），默认 0.05",
    )
    args = parser.parse_args()

    render = not args.no_render
    show_traj = not args.no_traj
    show_ft_mid = not args.no_ft_mid  # [新增]

    print(f"\n{'='*65}")
    print(f"  任务:       {TASK_REGISTRY[args.task]['display_name']}")
    print(f"  模式:       {args.mode}")
    if args.mode == "random":
        print(f"  渲染:       {'是' if render else '否'}")
        print(f"  轨迹可视化: {'是' if show_traj else '否'}")
        print(f"  指尖中点:   {'是' if show_ft_mid else '否'}")  # [新增]
    elif args.mode == "keyboard":
        print(f"  臂步长:     {args.arm_step} rad")
        print(f"  手步长:     {args.hand_step} m")
        print(f"  指尖中点:   {'是' if show_ft_mid else '否'}")  # [新增]
    print(f"{'='*65}\n")

    if args.mode == "random":
        demo_random_policy(
            task_name=args.task,
            n_episodes=args.episodes,
            render=render,
            action_mode=args.action_mode,
            controller_type=args.controller,
            show_ee_traj=show_traj,
            show_fingertip_midpoint=show_ft_mid,  # [新增]
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
        )